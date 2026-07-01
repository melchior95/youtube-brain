"""Deterministic outlier detection: which of a brain's videos wildly overperform.

Grouped by brain, not channel_name: channel_name is null for channel/playlist-
ingested videos (yt-dlp's --flat-playlist enumeration never returns it), while
brain_id is always populated and one brain is already one creator's channel in
this project's ingestion model. Counts and ratios are computed from persisted
view_count, never guessed, mirroring the trust rule in observations/report.py.
"""

from __future__ import annotations

import statistics

from youtube_brain.core.models import Video


def compute_outliers(
    videos: list[Video],
    ratio_threshold: float = 3.0,
    min_videos: int = 3,
) -> list[dict]:
    """Videos whose view_count is >= ratio_threshold times the brain's median.

    Returns [] when fewer than min_videos have a known view_count (no
    reliable baseline to compare against). Sorted by ratio descending.
    """
    counts = [v.view_count for v in videos if v.view_count is not None]
    if len(counts) < min_videos:
        return []
    median = statistics.median(counts)
    if median <= 0:
        return []

    outliers = [
        {
            "video_id": v.video_id,
            "title": v.title,
            "view_count": v.view_count,
            "median_views": median,
            "ratio": round(v.view_count / median, 2),
        }
        for v in videos
        if v.view_count is not None and v.view_count / median >= ratio_threshold
    ]
    outliers.sort(key=lambda o: -o["ratio"])
    return outliers
