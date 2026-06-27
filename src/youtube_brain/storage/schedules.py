"""CRUD for watchlist_schedules — per-brain auto-refresh configuration."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.storage.database import get_session, watchlist_schedules


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row) -> dict:
    return {
        "brain_id": row.brain_id,
        "enabled": bool(row.enabled),
        "interval_hours": row.interval_hours,
        "max_videos": row.max_videos,
        "last_refreshed_at": row.last_refreshed_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


async def upsert_schedule(
    brain_id: str,
    enabled: bool = True,
    interval_hours: int = 24,
    max_videos: int | None = None,
) -> None:
    """Create or update a brain's watchlist schedule."""
    now = _now()
    async with get_session() as session:
        stmt = insert(watchlist_schedules).values(
            brain_id=brain_id,
            enabled=enabled,
            interval_hours=interval_hours,
            max_videos=max_videos,
            last_refreshed_at=None,
            created_at=now,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["brain_id"],
            set_={
                "enabled": enabled,
                "interval_hours": interval_hours,
                "max_videos": max_videos,
                "updated_at": now,
            },
        )
        await session.execute(stmt)


async def get_schedule(brain_id: str) -> dict | None:
    async with get_session() as session:
        stmt = select(watchlist_schedules).where(
            watchlist_schedules.c.brain_id == brain_id
        )
        row = (await session.execute(stmt)).fetchone()
        return _row_to_dict(row) if row else None


async def list_schedules() -> list[dict]:
    async with get_session() as session:
        result = await session.execute(select(watchlist_schedules))
        return [_row_to_dict(r) for r in result.fetchall()]


async def mark_refreshed(brain_id: str, when: datetime | None = None) -> None:
    async with get_session() as session:
        await session.execute(
            update(watchlist_schedules)
            .where(watchlist_schedules.c.brain_id == brain_id)
            .values(last_refreshed_at=when or _now(), updated_at=_now())
        )
