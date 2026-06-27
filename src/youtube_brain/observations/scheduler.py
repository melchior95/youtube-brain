"""Watchlist scheduler — decide which brains are due and refresh them.

`is_due` is pure (testable). `run_due` is the one-shot the user wires into an
OS scheduler (Windows Task Scheduler / cron); `watch loop` polls it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def is_due(schedule: dict, now: datetime) -> bool:
    """True if an enabled schedule's interval has elapsed since last refresh."""
    if not schedule.get("enabled"):
        return False
    last = _as_utc(schedule.get("last_refreshed_at"))
    if last is None:
        return True
    interval = timedelta(hours=schedule.get("interval_hours", 24))
    return now >= last + interval


async def run_due(now: datetime | None = None) -> list[dict]:
    """Refresh every brain whose schedule is due. Returns a per-brain summary."""
    from youtube_brain.observations.refresh import refresh_brain
    from youtube_brain.storage.schedules import list_schedules, mark_refreshed

    now = now or datetime.now(timezone.utc)
    results: list[dict] = []
    for sched in await list_schedules():
        if not is_due(sched, now):
            continue
        bid = sched["brain_id"]
        try:
            out = await refresh_brain(bid, max_videos=sched.get("max_videos"))
            await mark_refreshed(bid, now)
            results.append({"brain_id": bid, **out})
        except Exception as exc:  # noqa: BLE001 - one bad brain shouldn't stop the rest
            logger.error("Refresh failed for brain %s: %s", bid, exc)
            results.append({"brain_id": bid, "error": str(exc)})
    return results
