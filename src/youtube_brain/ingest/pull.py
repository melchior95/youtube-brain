"""Channel-aware, zero-generate pull — the shared ingest path for a creator.

A *pull* lite-ingests a video / channel / playlist (transcript + chunk + embed,
ZERO Gemini *generate* quota) and groups single videos under their channel's
brain so a creator accumulates across pulls regardless of how YouTube renders
the display name or handle. The stable identity is the **channel_id**, which is
stamped onto the brain's source row (`sources.source_id`) so future pulls — and
the category page's channel_id matching — converge on ONE brain.

Both the skill bridge (`scripts/skill_bridge.py`) and the web API
(`POST /api/brains/ingest`) call :func:`pull_creator`; presentation (JSON
emission, transcript read-back, logging) belongs to the callers. This module is
a pure library: it never prints to stdout/stderr or emits JSON.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, update

from youtube_brain.core.enums import SourceStatus, SourceType
from youtube_brain.core.models import Source
from youtube_brain.ingest.pipeline import ingest_url
from youtube_brain.ingest.resolver import (
    _resolve_single_video,
    parse_youtube_url,
    resolve_video_ids,
)
from youtube_brain.storage.brains import get_brain, list_brains
from youtube_brain.storage.database import get_session, init_database, sources as sources_table
from youtube_brain.storage.sources import get_sources_by_brain

logger = logging.getLogger(__name__)

# Default cap — bound cost on channel/playlist pulls.
DEFAULT_CHANNEL_LIMIT = 6


async def _match_brain(channel_id: str | None, brain_name: str):
    """Find an existing brain by STABLE channel_id (preferred) or name (legacy).

    channel_id is stored on the brain's source row (sources.source_id), so a
    video-pull and a channel-pull of the same creator converge on ONE brain
    regardless of how YouTube renders the display name or handle. Returns
    (existing_brain, existing_source) — both None when a new brain is needed.
    """
    if channel_id:
        async with get_session() as session:
            row = (await session.execute(
                select(sources_table).where(sources_table.c.source_id == channel_id)
            )).fetchone()
        if row is not None:
            brain = await get_brain(row.brain_id)
            if brain is not None:
                srcs = await get_sources_by_brain(brain.id)
                return brain, (srcs[0] if srcs else None)

    for brain in await list_brains():  # legacy fallback: human-readable name
        if brain.name == brain_name:
            srcs = await get_sources_by_brain(brain.id)
            return brain, (srcs[0] if srcs else None)
    return None, None


async def _tag_channel_id(brain_id: str, channel_id: str) -> None:
    """Stamp the stable channel_id onto the brain's source(s) for future matching."""
    async with get_session() as session:
        await session.execute(
            update(sources_table)
            .where(sources_table.c.brain_id == brain_id)
            .values(source_id=channel_id)
        )


async def pull_creator(url: str, limit: int | None = None, topic_brain: str | None = None) -> dict:
    """Lite-ingest a video / channel / playlist under its channel's brain.

    When ``topic_brain`` is set, the videos go into a named TOPIC brain instead
    (matched/created by name, spanning many channels) — used by question-driven
    research. A topic brain is never keyed/stamped by channel_id, so a later
    creator pull can't wrongly merge into it.

    Resolves the video metas, applies the channel/playlist cap (default
    :data:`DEFAULT_CHANNEL_LIMIT`), determines the stable channel_id (enriching a
    flat-playlist enumeration by fully resolving the first video when needed),
    matches or creates the brain by channel_id, lite-ingests with
    ``generate_metadata=False`` (ZERO generate), and stamps the channel_id onto
    the source so future pulls merge here.

    Returns a dict with: ``brain_id``, ``brain_name``, ``channel_id``,
    ``source_type``, ``videos_found``, ``videos_processed``, ``chunks_created``,
    ``errors``, ``targeted_ids`` (the youtube video ids this pull targeted). On
    failure to resolve any videos, returns ``{"error": "no_videos_resolved",
    "url": url}``.
    """
    await init_database()
    parsed = parse_youtube_url(url)

    metas = resolve_video_ids(parsed)
    if not metas:
        return {"error": "no_videos_resolved", "url": url}

    if parsed.source_type in ("channel", "playlist"):
        cap = limit if (limit and limit > 0) else DEFAULT_CHANNEL_LIMIT
        metas = metas[:cap]

    # Stable identity = channel_id. Single-video dumps carry it directly;
    # flat-playlist channel enumeration does NOT, so recover it (and the display
    # name) by fully resolving the first video — one extra yt-dlp call.
    channel_id = next((m.get("channel_id") for m in metas if m.get("channel_id")), None)
    channel_name = next((m.get("channel_name") for m in metas if m.get("channel_name")), None)
    if (channel_id is None or channel_name is None) and metas:
        try:
            full = _resolve_single_video(metas[0]["video_id"])
            channel_id = channel_id or full.get("channel_id")
            channel_name = channel_name or full.get("channel_name")
        except Exception as exc:
            logger.debug("channel enrichment failed: %s", exc)

    if topic_brain:
        brain_name = topic_brain
    elif channel_name:
        brain_name = channel_name
    elif parsed.channel_handle:
        brain_name = f"@{parsed.channel_handle}"
    elif parsed.playlist_id:
        brain_name = f"Playlist {parsed.playlist_id}"
    elif parsed.video_id:
        brain_name = f"Video {parsed.video_id}"
    else:
        brain_name = f"Brain from {url}"

    # Topic brains span many channels — match/create by NAME only, never by
    # channel_id (which would wrongly merge a later creator pull of one of them).
    existing_brain, existing_source = await _match_brain(
        None if topic_brain else channel_id, brain_name
    )

    # When appending to an existing brain that has no source row, mint one
    # (keyed by channel_id for creator brains, by topic name for topic brains).
    if existing_brain is not None and existing_source is None:
        existing_source = Source(
            brain_id=existing_brain.id,
            source_type=SourceType(parsed.source_type),
            source_url=url,
            source_id=topic_brain if topic_brain else (
                channel_id or parsed.video_id or parsed.playlist_id
                or parsed.channel_handle or url),
            status=SourceStatus.RESOLVING,
        )

    targeted_ids = [m["video_id"] for m in metas]

    result = await ingest_url(
        url,
        brain_name=brain_name,
        existing_brain=existing_brain,
        existing_source=existing_source,
        video_metas=metas,
        generate_metadata=False,
    )

    # Stamp the stable channel_id so future pulls of this creator merge here —
    # but never for a topic brain (it spans channels; stamping would mis-merge).
    if channel_id and not topic_brain:
        await _tag_channel_id(result.brain_id, channel_id)

    return {
        "brain_id": result.brain_id,
        "brain_name": brain_name,
        "channel_id": channel_id,
        "source_type": parsed.source_type,
        "videos_found": result.videos_found,
        "videos_processed": result.videos_processed,
        "chunks_created": result.chunks_created,
        "errors": result.errors,
        "targeted_ids": targeted_ids,
    }
