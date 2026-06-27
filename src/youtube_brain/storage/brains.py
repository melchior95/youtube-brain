"""CRUD operations for the brains table."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.enums import BrainStatus
from youtube_brain.core.models import Brain
from youtube_brain.storage.database import brains, get_session


def _uuid(val: UUID | str) -> str:
    """Convert a UUID or string to a plain string for storage."""
    return str(val)


def _row_to_brain(row) -> Brain:
    """Convert a database row to a Brain model."""
    return Brain(
        id=row.id,
        name=row.name,
        owner_user_id=row.owner_user_id,
        visibility=row.visibility,
        canonical_brain_id=row.canonical_brain_id,
        recency_weight=row.recency_weight,
        video_count=row.video_count,
        status=BrainStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def insert_brain(brain: Brain) -> bool:
    """Insert a brain with on_conflict_do_nothing. Returns True if inserted."""
    async with get_session() as session:
        stmt = insert(brains).values(
            id=_uuid(brain.id),
            name=brain.name,
            owner_user_id=brain.owner_user_id,
            visibility=brain.visibility,
            canonical_brain_id=_uuid(brain.canonical_brain_id) if brain.canonical_brain_id else None,
            recency_weight=brain.recency_weight,
            video_count=brain.video_count,
            status=brain.status.value,
            created_at=brain.created_at,
            updated_at=brain.updated_at,
        ).on_conflict_do_nothing(index_elements=["id"])
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_brain(brain_id: UUID | str) -> Brain | None:
    """Get a brain by ID."""
    async with get_session() as session:
        stmt = select(brains).where(brains.c.id == _uuid(brain_id))
        result = await session.execute(stmt)
        row = result.fetchone()
        if row is None:
            return None
        return _row_to_brain(row)


async def list_brains() -> list[Brain]:
    """List all brains ordered by updated_at descending."""
    async with get_session() as session:
        stmt = select(brains).order_by(brains.c.updated_at.desc())
        result = await session.execute(stmt)
        return [_row_to_brain(row) for row in result.fetchall()]


async def update_brain_status(brain_id: UUID | str, status: BrainStatus) -> None:
    """Update a brain's status and updated_at timestamp."""
    async with get_session() as session:
        stmt = (
            update(brains)
            .where(brains.c.id == _uuid(brain_id))
            .values(status=status.value, updated_at=datetime.now(timezone.utc))
        )
        await session.execute(stmt)


async def increment_video_count(brain_id: UUID | str, count: int = 1) -> None:
    """Atomically increment a brain's video_count."""
    async with get_session() as session:
        stmt = (
            update(brains)
            .where(brains.c.id == _uuid(brain_id))
            .values(video_count=brains.c.video_count + count)
        )
        await session.execute(stmt)
