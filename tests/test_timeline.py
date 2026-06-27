"""Tests for the theme timeline engine (pure logic)."""

from datetime import datetime, timezone

from youtube_brain.core.models import Observation
from youtube_brain.observations.timeline import bucket_key, build_timeline


def test_bucket_keys():
    dt = datetime(2026, 5, 17, tzinfo=timezone.utc)
    assert bucket_key(dt, "month") == "2026-05"
    assert bucket_key(dt, "quarter") == "2026-Q2"
    assert bucket_key(dt, "week").startswith("2026-W")


def _obs(yt, kws_value, otype="tool"):
    return Observation(brain_id="b", youtube_id=yt, creator=yt,
                       obs_type=otype, claim=kws_value, value=kws_value)


def test_timeline_cumulative_founders_grow():
    # Two founders in April mention Supabase, a third in May.
    obs = [
        _obs("a", "Supabase"),
        _obs("b", "Supabase"),
        _obs("c", "Supabase"),
    ]
    published = {
        "a": datetime(2026, 4, 5, tzinfo=timezone.utc),
        "b": datetime(2026, 4, 20, tzinfo=timezone.utc),
        "c": datetime(2026, 5, 10, tzinfo=timezone.utc),
    }
    tl = build_timeline(obs, published, "month")
    assert tl["periods"] == ["2026-04", "2026-05"]
    # Cumulative distinct founders: 2 by April, 3 by May.
    assert tl["series"]["Tools"]["Supabase"] == [2, 3]
    assert tl["founders_cumulative"] == [2, 3]
    # The May gain shows up as a trend.
    assert any(t["entity"] == "Supabase" and t["to"] == 3 for t in tl["trends"])


def test_timeline_empty_without_dates():
    obs = [_obs("a", "Supabase")]
    tl = build_timeline(obs, {}, "month")
    assert tl["periods"] == []
    assert tl["series"] == {}
