"""Ingestion pipeline orchestrator: resolver, transcripts, chunker, FTS index.

No external API. Transcripts are fetched, chunked, and stored; the FTS5 index is
kept in sync by triggers, so chunks are immediately retrievable by keyword. All
semantic work (summaries, observations, answers) is done by Claude in-loop via
the skill bridge, not here.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from youtube_brain.config.settings import get_settings
from youtube_brain.core.enums import BrainStatus, SourceStatus, SourceType, VideoStatus
from youtube_brain.core.models import Brain, Chunk, Source, Video
from youtube_brain.embed import EMBED_DIMS, EMBED_MODEL, embed_texts
from youtube_brain.ingest.chunker import chunk_transcript
from youtube_brain.ingest.resolver import fetch_video_stats, parse_youtube_url, resolve_video_ids
from youtube_brain.ingest.transcripts import fetch_transcript
from youtube_brain.storage.brains import (
    increment_video_count,
    insert_brain,
    update_brain_status,
)
from youtube_brain.storage.chunks import insert_chunks, store_embedding
from youtube_brain.storage.database import get_session, sources as sources_table
from youtube_brain.storage.videos import (
    get_videos_by_brain,
    insert_video,
    update_video,
    video_exists,
)

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
) -> PipelineResult:
    """Ingest a YouTube URL into a brain: transcript -> chunks -> FTS index.

    For each video: fetch the transcript, store it, chunk it, and insert the
    chunks (which the FTS5 triggers index). No embeddings, no LLM calls.

    Args:
        url: A YouTube URL (video, playlist, or channel).
        brain_name: Optional brain name; derived from the URL if not provided.
        limit: Optional cap on videos to process (most recent first).
        existing_brain / existing_source: reuse when appending to a brain.
        video_metas: explicit video list (skips resolution).

    Returns:
        PipelineResult with counts and any errors.
    """
    settings = get_settings()

    # 1. Parse URL
    parsed = parse_youtube_url(url)

    # 2. Create Brain record (or append to an existing one)
    if existing_brain is not None:
        brain = existing_brain
    else:
        name = brain_name or _derive_brain_name(parsed, url)
        brain = Brain(name=name)
        await insert_brain(brain)
    brain_id = str(brain.id)

    result = PipelineResult(brain_id=brain_id)

    # 3. Create Source record (reuse the brain's existing source on append)
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

    # 5. Resolve video IDs (unless an explicit list was supplied)
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

            _source_map = {"api": "api", "yt-dlp": "yt_dlp"}
            transcript_source = _source_map.get(transcript_result.source, transcript_result.source)
            caption_kind = "auto" if transcript_result.is_auto_generated else "manual"
            language = transcript_result.language

            # c2. Backfill view/like/comment/subscriber counts. --flat-playlist
            #    enumeration (channels/playlists) never returns these; a single-
            #    video resolve already has them. Best-effort: failures leave the
            #    stats as None rather than blocking ingestion.
            if "view_count" not in video_meta:
                video_meta = {**video_meta, **fetch_video_stats(vid_id)}

            # d. Insert video record
            video = Video(
                brain_id=brain.id,
                source_id=source.id,
                video_id=vid_id,
                title=video_meta.get("title"),
                channel_name=video_meta.get("channel_name"),
                published_at=video_meta.get("published_at"),
                duration_seconds=video_meta.get("duration_seconds"),
                view_count=video_meta.get("view_count"),
                like_count=video_meta.get("like_count"),
                comment_count=video_meta.get("comment_count"),
                channel_follower_count=video_meta.get("channel_follower_count"),
                stats_fetched_at=video_meta.get("stats_fetched_at"),
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

            # f. Insert chunks (FTS5 triggers index them for keyword retrieval)
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

            # g. Embed chunk texts for dense retrieval (local fastembed, no API).
            #    Off the event loop since embedding is CPU-bound. A failure here
            #    leaves the chunks keyword-searchable (FTS), just not dense.
            try:
                texts = [cm.text for cm in chunk_models]
                if texts:
                    vecs = await asyncio.to_thread(embed_texts, texts)
                    for cm, vec in zip(chunk_models, vecs):
                        await store_embedding(cm.id, EMBED_MODEL, EMBED_DIMS, vec)
            except Exception as exc:
                logger.warning("Embedding failed for video %s: %s", vid_id, exc)
                result.errors.append(f"Embedding failed for {vid_id}: {exc}")

            result.videos_processed += 1

            if result.videos_processed == settings.partially_ready_threshold:
                await update_brain_status(brain.id, BrainStatus.PARTIALLY_READY)

        except Exception as exc:
            error_msg = f"Error processing video {vid_id}: {exc}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
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

    return result


async def refresh_video_stats(brain_id: str) -> dict:
    """Re-fetch and store current view/like/comment/subscriber counts for
    every video in a brain.

    On-demand only, never run automatically: stats are otherwise frozen at
    whatever moment each video was first ingested (or last refreshed here), so
    they drift out of sync across videos pulled at different times. Best-
    effort per video: a failed fetch leaves that video's existing stats
    untouched rather than blanking them with Nones.
    """
    videos = await get_videos_by_brain(brain_id, limit=None)
    refreshed = 0
    failed = 0
    for video in videos:
        stats = fetch_video_stats(video.video_id)
        if stats.get("view_count") is None:
            failed += 1
            continue
        await update_video(
            video.id,
            view_count=stats["view_count"],
            like_count=stats["like_count"],
            comment_count=stats["comment_count"],
            channel_follower_count=stats["channel_follower_count"],
            stats_fetched_at=stats["stats_fetched_at"],
        )
        refreshed += 1
    return {"videos_total": len(videos), "videos_refreshed": refreshed, "videos_failed": failed}
