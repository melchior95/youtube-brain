"""CRUD operations for the videos table."""

from enum import Enum
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.enums import TranscriptSource, VideoStatus
from youtube_brain.core.models import Video
from youtube_brain.storage.database import videos, get_session


def _uuid(val: UUID | str) -> str:
    """Convert a UUID or string to a plain string for storage."""
    return str(val)


async def get_published_dates_by_brain(brain_id: UUID | str) -> dict:
    """Return {youtube_video_id: published_at} for a brain's dated videos."""
    async with get_session() as session:
        stmt = select(videos.c.video_id, videos.c.published_at).where(
            videos.c.brain_id == _uuid(brain_id),
            videos.c.published_at.isnot(None),
        )
        result = await session.execute(stmt)
        return {row.video_id: row.published_at for row in result.fetchall()}


def _row_to_video(row) -> Video:
    """Convert a database row to a Video model."""
    return Video(
        id=row.id,
        brain_id=row.brain_id,
        source_id=row.source_id,
        video_id=row.video_id,
        title=row.title,
        channel_name=row.channel_name,
        published_at=row.published_at,
        duration_seconds=row.duration_seconds,
        view_count=row.view_count,
        like_count=row.like_count,
        comment_count=row.comment_count,
        channel_follower_count=row.channel_follower_count,
        stats_fetched_at=row.stats_fetched_at,
        url=row.url,
        transcript_raw=row.transcript_raw,
        transcript_clean=row.transcript_clean,
        transcript_source=(
            TranscriptSource(row.transcript_source)
            if row.transcript_source is not None
            else None
        ),
        transcript_language=row.transcript_language,
        caption_kind=row.caption_kind,
        transcript_quality_score=row.transcript_quality_score,
        failure_reason=row.failure_reason,
        video_summary=row.video_summary,
        key_points=row.key_points or [],
        businesses_mentioned=row.businesses_mentioned or [],
        people_mentioned=row.people_mentioned or [],
        main_topics=row.main_topics or [],
        status=VideoStatus(row.status),
        created_at=row.created_at,
    )


async def insert_video(video: Video) -> bool:
    """Insert a video with on_conflict_do_nothing. Returns True if inserted."""
    async with get_session() as session:
        stmt = insert(videos).values(
            id=_uuid(video.id),
            brain_id=_uuid(video.brain_id),
            source_id=_uuid(video.source_id),
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            published_at=video.published_at,
            duration_seconds=video.duration_seconds,
            view_count=video.view_count,
            like_count=video.like_count,
            comment_count=video.comment_count,
            channel_follower_count=video.channel_follower_count,
            stats_fetched_at=video.stats_fetched_at,
            url=video.url,
            transcript_raw=video.transcript_raw,
            transcript_clean=video.transcript_clean,
            transcript_source=(
                video.transcript_source.value
                if video.transcript_source is not None
                else None
            ),
            transcript_language=video.transcript_language,
            caption_kind=video.caption_kind,
            transcript_quality_score=video.transcript_quality_score,
            failure_reason=video.failure_reason,
            video_summary=video.video_summary,
            key_points=video.key_points,
            businesses_mentioned=video.businesses_mentioned,
            people_mentioned=video.people_mentioned,
            main_topics=video.main_topics,
            status=video.status.value,
            created_at=video.created_at,
        ).on_conflict_do_nothing(index_elements=["id"])
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_videos_by_brain(
    brain_id: UUID | str,
    status: VideoStatus | None = None,
    limit: int = 100,
) -> list[Video]:
    """Get videos for a brain, optionally filtered by status."""
    async with get_session() as session:
        stmt = select(videos).where(videos.c.brain_id == _uuid(brain_id))
        if status is not None:
            stmt = stmt.where(videos.c.status == status.value)
        stmt = stmt.order_by(videos.c.created_at).limit(limit)
        result = await session.execute(stmt)
        return [_row_to_video(row) for row in result.fetchall()]


async def update_video(video_id: UUID | str, **kwargs) -> None:
    """Update arbitrary fields on a video. Enum values are converted automatically."""
    values = {}
    for key, val in kwargs.items():
        if isinstance(val, Enum):
            values[key] = val.value
        else:
            values[key] = val
    async with get_session() as session:
        stmt = (
            update(videos)
            .where(videos.c.id == _uuid(video_id))
            .values(**values)
        )
        await session.execute(stmt)


async def video_exists(brain_id: UUID | str, yt_video_id: str) -> bool:
    """Check if a video with the given YouTube video_id exists within a brain."""
    async with get_session() as session:
        stmt = (
            select(videos.c.id)
            .where(videos.c.brain_id == _uuid(brain_id))
            .where(videos.c.video_id == yt_video_id)
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.fetchone() is not None
