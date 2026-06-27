"""Tests for the watchlist scheduler due-logic (pure)."""

from datetime import datetime, timedelta, timezone

from youtube_brain.observations.scheduler import is_due

NOW = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)


def _sched(**kw):
    base = {"enabled": True, "interval_hours": 24, "last_refreshed_at": None}
    base.update(kw)
    return base


def test_disabled_is_never_due():
    assert is_due(_sched(enabled=False), NOW) is False


def test_never_refreshed_is_due():
    assert is_due(_sched(last_refreshed_at=None), NOW) is True


def test_recently_refreshed_not_due():
    last = NOW - timedelta(hours=2)
    assert is_due(_sched(interval_hours=24, last_refreshed_at=last), NOW) is False


def test_interval_elapsed_is_due():
    last = NOW - timedelta(hours=25)
    assert is_due(_sched(interval_hours=24, last_refreshed_at=last), NOW) is True


def test_naive_last_refreshed_treated_as_utc():
    naive = (NOW - timedelta(hours=25)).replace(tzinfo=None)
    assert is_due(_sched(last_refreshed_at=naive), NOW) is True
