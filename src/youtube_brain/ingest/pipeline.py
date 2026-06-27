"""Ingestion pipeline orchestrator — wires resolver, transcripts, chunker, embeddings, labeler, and summarizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from youtube_brain.config.settings import get_settings
from youtube_brain.core.enums import BrainStatus, SourceStatus, SourceType, VideoStatus
from youtube_brain.core.models import Brain, Chunk, Source, Video
from youtube_brain.ingest.chunker import chunk_transcript
from youtube_brain.ingest.labeler import label_chunks
from youtube_brain.ingest.resolver import parse_youtube_url, resolve_video_ids
from youtube_brain.ingest.summarizer import summarize_video
from youtube_brain.ingest.transcripts import clean_transcript, fetch_transcript
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.storage.brains import (
    increment_video_count,
    insert_brain,
    update_brain_status,
)
from youtube_brain.storage.chunks import insert_chunks, store_embedding
from youtube_brain.storage.database import chunks as chunks_table, get_session, sources as sources_table
from youtube_brain.storage.videos import insert_video, update_video, video_exists

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of an ingestion pipeline run."""

    brain_id: str
    videos_found: int = 0
    videos_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


async def _insert_source(source: Source) -> None:
    """Insert a Source record directly via the session."""
    async with get_session() as session:
        from sqlalchemy.dialects.sqlite import insert

        stmt = insert(sources_table).values(
            id=str(source.id),
            brain_id=str(source.brain_id),
            source_type=source.source_type.value,
            source_url=source.source_url,
            source_title=source.source_title,
            source_id=source.source_id,
            status=source.status.value,
            created_at=source.created_at,
        ).on_conflict_do_nothing(index_elements=["id"])
        await session.execute(stmt)


def _format_transcript_raw(segments: list[dict]) -> str:
    """Format transcript segments with timestamps for storage."""
    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        minutes = int(start // 60)
        seconds = start % 60
        text = seg.get("text", "")
        lines.append(f"[{minutes:02d}:{seconds:05.2f}] {text}")
    return "\n".join(lines)


def _derive_brain_name(parsed, url: str) -> str:
    """Derive a brain name from the parsed URL info."""
    if parsed.channel_handle:
        return f"@{parsed.channel_handle}"
    if parsed.channel_id:
        return f"Channel {parsed.channel_id}"
    if parsed.playlist_id:
        return f"Playlist {parsed.playlist_id}"
    if parsed.video_id:
        return f"Video {parsed.video_id}"
    return f"Brain from {url}"


def _derive_source_id(parsed) -> str:
    """Derive a source_id string from the parsed URL."""
    if parsed.video_id:
        return parsed.video_id
    if parsed.playlist_id:
        return parsed.playlist_id
    if parsed.channel_handle:
        return parsed.channel_handle
    if parsed.channel_id:
        return parsed.channel_id
    return parsed.original_url


async def ingest_url(
    url: str,
    brain_name: str | None = None,
    limit: int | None = None,
    existing_brain: Brain | None = None,
    existing_source: Source | None = None,
    video_metas: list[dict] | None = None,
    generate_metadata: bool = True,
) -> PipelineResult:
    """Run the full ingestion pipeline for a YouTube URL.

    Flow:
        1. Parse URL
        2. Create Brain and Source records
        3. Resolve video IDs
        4. For each video: fetch transcript, chunk, embed, label, summarize
        5. Update brain status

    Args:
        url: A YouTube URL (video, playlist, or channel).
        brain_name: Optional brain name; derived from URL if not provided.
        limit: Optional cap on the number of videos to process (most recent
            first, as returned by yt-dlp). Useful for bounding cost on large
            channels.

    Returns:
        PipelineResult with counts and any errors.
    """
    settings = get_settings()

    # 1. Parse URL
    parsed = parse_youtube_url(url)

    # 2. Create Brain record (or append to an existing one for a watchlist refresh)
    if existing_brain is not None:
        brain = existing_brain
    else:
        name = brain_name or _derive_brain_name(parsed, url)
        brain = Brain(name=name)
        await insert_brain(brain)
    brain_id = str(brain.id)

    result = PipelineResult(brain_id=brain_id)

    # 3. Create Source record (reuse the brain's existing source on refresh)
    if existing_source is not None:
        source = existing_source
    else:
        source = Source(
            brain_id=brain.id,
            source_type=SourceType(parsed.source_type),
            source_url=url,
            source_id=_derive_source_id(parsed),
            status=SourceStatus.RESOLVING,
        )
        await _insert_source(source)

    # 4. Set brain status to INGESTING
    await update_brain_status(brain.id, BrainStatus.INGESTING)

    # 5. Resolve video IDs (unless an explicit list was supplied, e.g. on refresh)
    if video_metas is None:
        try:
            video_metas = resolve_video_ids(parsed)
        except Exception as exc:
            error_msg = f"Failed to resolve video IDs: {exc}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            await update_brain_status(brain.id, BrainStatus.ERROR)
            return result

    if limit is not None and limit > 0:
        video_metas = video_metas[:limit]

    result.videos_found = len(video_metas)

    # Initialize Gemini client
    client = GeminiClient()

    try:
        # 6. Process each video
        for video_meta in video_metas:
            vid_id = video_meta["video_id"]

            video = None
            try:
                # a. Skip if already exists
                if await video_exists(brain.id, vid_id):
                    logger.info("Video %s already exists in brain %s, skipping", vid_id, brain_id)
                    continue

                # b. Fetch transcript
                transcript_result = await fetch_transcript(vid_id)
                if transcript_result is None:
                    error_msg = f"No transcript available for video {vid_id}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    continue

                # c. Prepare transcript data
                transcript_raw = _format_transcript_raw(transcript_result.text_with_timestamps)
                transcript_clean_text = transcript_result.full_text

                # Determine transcript source and caption kind
                _source_map = {"api": "api", "yt-dlp": "yt_dlp"}
                transcript_source = _source_map.get(transcript_result.source, transcript_result.source)
                caption_kind = "auto" if transcript_result.is_auto_generated else "manual"
                language = transcript_result.language

                # d. Insert video record
                video = Video(
                    brain_id=brain.id,
                    source_id=source.id,
                    video_id=vid_id,
                    title=video_meta.get("title"),
                    channel_name=video_meta.get("channel_name"),
                    published_at=video_meta.get("published_at"),
                    duration_seconds=video_meta.get("duration_seconds"),
                    url=f"https://www.youtube.com/watch?v={vid_id}",
                    transcript_raw=transcript_raw,
                    transcript_clean=transcript_clean_text,
                    transcript_source=transcript_source,
                    transcript_language=language,
                    caption_kind=caption_kind,
                    status=VideoStatus.FETCHED,
                )
                await insert_video(video)
                await increment_video_count(brain.id)

                # e. Chunk transcript
                chunk_dicts = chunk_transcript(
                    transcript_result.text_with_timestamps,
                    window=settings.chunk_window_seconds,
                    overlap=settings.chunk_overlap_seconds,
                )

                # f. Insert chunks
                chunk_models = [
                    Chunk(
                        video_id=video.id,
                        brain_id=brain.id,
                        start_time=cd["start_time"],
                        end_time=cd["end_time"],
                        text=cd["text"],
                    )
                    for cd in chunk_dicts
                ]
                await insert_chunks(chunk_models)
                result.chunks_created += len(chunk_models)

                await update_video(video.id, status=VideoStatus.CHUNKED)

                # g. Embed all chunk texts
                try:
                    chunk_texts = [cm.text for cm in chunk_models]
                    if chunk_texts:
                        embeddings = await client.embed_texts(chunk_texts)
                        for cm, emb in zip(chunk_models, embeddings):
                            await store_embedding(
                                cm.id,
                                client.embed_model,
                                client.embed_dims,
                                emb,
                            )
                except Exception as exc:
                    error_msg = f"Embedding failed for video {vid_id}: {exc}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

                # h. Label chunks (skipped in lite mode — labels feed Q&A
                #    retrieval, not the intelligence reports, and are the
                #    biggest generate cost per video)
                if not generate_metadata:
                    result.videos_processed += 1
                    continue
                try:
                    labels = await label_chunks(
                        client, chunk_dicts, video_meta.get("title") or "", video_meta.get("channel_name") or ""
                    )
                    for cm, label in zip(chunk_models, labels):
                        if label:
                            update_values = {}
                            if label.get("topics"):
                                topics = label["topics"]
                                update_values["topics"] = topics if isinstance(topics, list) else [topics]
                            if label.get("business_type"):
                                bt = label["business_type"]
                                update_values["business_type"] = bt if isinstance(bt, list) else [bt]
                            if label.get("advice_category"):
                                ac = label["advice_category"]
                                update_values["advice_category"] = ac if isinstance(ac, list) else [ac]
                            if label.get("stage"):
                                st = label["stage"]
                                update_values["stage"] = st if isinstance(st, list) else [st]
                            if label.get("asset_type"):
                                at = label["asset_type"]
                                update_values["asset_type"] = at if isinstance(at, list) else [at]
                            if update_values:
                                await _update_chunk_metadata(cm.id, update_values)
                except Exception as exc:
                    error_msg = f"Labeling failed for video {vid_id}: {exc}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

                # i. Summarize video
                try:
                    summary = await summarize_video(
                        client,
                        transcript_clean_text,
                        video_meta.get("title") or "",
                        video_meta.get("channel_name") or "",
                    )
                    if summary:
                        await update_video(
                            video.id,
                            video_summary=summary.get("video_summary"),
                            key_points=summary.get("key_points", []),
                            businesses_mentioned=summary.get("businesses_mentioned", []),
                            people_mentioned=summary.get("people_mentioned", []),
                            main_topics=summary.get("main_topics", []),
                            status=VideoStatus.SUMMARIZED,
                        )
                    else:
                        await update_video(video.id, status=VideoStatus.SUMMARIZED)
                except Exception as exc:
                    error_msg = f"Summarization failed for video {vid_id}: {exc}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

                # j. Track processed count
                result.videos_processed += 1

                # k. Partially ready threshold
                if result.videos_processed == settings.partially_ready_threshold:
                    await update_brain_status(brain.id, BrainStatus.PARTIALLY_READY)

            except Exception as exc:
                error_msg = f"Error processing video {vid_id}: {exc}"
                logger.error(error_msg, exc_info=True)
                result.errors.append(error_msg)
                # Try to mark the video as errored if it was inserted
                try:
                    if video is not None:
                        await update_video(video.id, status=VideoStatus.ERROR, failure_reason=str(exc))
                except Exception:
                    pass

        # 7. Set final brain status
        if result.videos_processed > 0:
            await update_brain_status(brain.id, BrainStatus.READY)
        else:
            await update_brain_status(brain.id, BrainStatus.ERROR)

    finally:
        # 8. Close Gemini client
        await client.close()

    return result


async def _update_chunk_metadata(chunk_id, values: dict) -> None:
    """Update metadata fields on a chunk row."""
    from sqlalchemy import update

    async with get_session() as session:
        stmt = (
            update(chunks_table)
            .where(chunks_table.c.id == str(chunk_id))
            .values(**values)
        )
        await session.execute(stmt)
