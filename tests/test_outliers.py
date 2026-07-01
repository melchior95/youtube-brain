"""Tests for deterministic view-count outlier detection."""

from uuid import uuid4

from youtube_brain.core.models import Video
from youtube_brain.observations.outliers import compute_outliers

_BRAIN_ID = uuid4()


def _video(video_id: str, view_count: int | None) -> Video:
    return Video(
        brain_id=_BRAIN_ID,
        source_id=_BRAIN_ID,
        video_id=video_id,
        title=f"Title {video_id}",
        url=f"https://youtube.com/watch?v={video_id}",
        view_count=view_count,
    )


def test_flags_video_far_above_median():
    videos = [
        _video("v1", 1000),
        _video("v2", 1200),
        _video("v3", 900),
        _video("v4", 50000),  # way above median
    ]
    outliers = compute_outliers(videos, ratio_threshold=3.0, min_videos=3)
    assert len(outliers) == 1
    assert outliers[0]["video_id"] == "v4"
    assert outliers[0]["ratio"] > 3.0


def test_no_outliers_when_views_are_similar():
    videos = [_video("v1", 1000), _video("v2", 1100), _video("v3", 950)]
    assert compute_outliers(videos) == []


def test_skips_brain_with_too_few_scored_videos():
    videos = [_video("v1", 1000), _video("v2", 50000)]
    assert compute_outliers(videos, min_videos=3) == []


def test_ignores_videos_with_no_view_count():
    videos = [_video("v1", None), _video("v2", None), _video("v3", 1000)]
    assert compute_outliers(videos, min_videos=3) == []


def test_zero_median_returns_no_outliers():
    videos = [_video("v1", 0), _video("v2", 0), _video("v3", 0)]
    assert compute_outliers(videos) == []


def test_sorted_by_ratio_descending():
    videos = [
        _video("v1", 1000),
        _video("v2", 1000),
        _video("v3", 1000),
        _video("v4", 10000),
        _video("v5", 5000),
    ]
    outliers = compute_outliers(videos, ratio_threshold=3.0, min_videos=3)
    assert [o["video_id"] for o in outliers] == ["v4", "v5"]
    assert outliers[0]["ratio"] > outliers[1]["ratio"]
