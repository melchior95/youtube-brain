"""Theme timeline, how consensus builds over time.

The differentiated view: for each tracked entity, the CUMULATIVE number of
distinct founders who had expressed it by each period ("by March, 9 founders
used Reddit"). Requires a time dimension, the source video's published_at.

`bucket_key` and `build_timeline` are pure (testable with injected dates).
"""

from __future__ import annotations

import re
from datetime import datetime

from youtube_brain.core.models import Observation
from youtube_brain.observations.rollup import CATEGORIES


def bucket_key(dt: datetime, granularity: str = "month") -> str:
    if granularity == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    if granularity == "week":
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return f"{dt.year}-{dt.month:02d}"  # month (default)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def build_timeline(
    observations: list[Observation],
    video_published: dict[str, datetime],
    granularity: str = "month",
) -> dict:
    """Cumulative distinct-founder counts per tracked entity over time.

    Args:
        observations: the brain's observations.
        video_published: youtube_id -> publish datetime.
        granularity: "week" | "month" | "quarter".
    """
    # Period for each observation (skip those whose video has no date).
    def period_of(o: Observation) -> str | None:
        dt = video_published.get(o.youtube_id or "")
        return bucket_key(dt, granularity) if dt else None

    periods = sorted({p for o in observations if (p := period_of(o))})
    if not periods:
        return {"granularity": granularity, "periods": [], "series": {},
                "volume": [], "founders_cumulative": [], "trends": []}
    pidx = {p: i for i, p in enumerate(periods)}

    # New-observation volume + cumulative distinct founders per period.
    volume = [0] * len(periods)
    founder_first: dict[str, str] = {}
    for o in observations:
        p = period_of(o)
        if not p:
            continue
        volume[pidx[p]] += 1
        f = o.youtube_id or o.creator
        if f and (f not in founder_first or p < founder_first[f]):
            founder_first[f] = p
    founders_cumulative = []
    running = 0
    for i, p in enumerate(periods):
        running += sum(1 for fp in founder_first.values() if pidx[fp] == i)
        founders_cumulative.append(running)

    # Per-entity: each founder's earliest period mentioning it -> cumulative series.
    series: dict[str, dict[str, list[int]]] = {}
    trends: list[dict] = []
    last = periods[-1]
    for cat_name, (types, ents) in CATEGORIES.items():
        cat_obs = [o for o in observations if o.obs_type in types and period_of(o)]
        cat_series: dict[str, list[int]] = {}
        for canonical, kws in ents.items():
            first_seen: dict[str, str] = {}  # founder -> earliest period
            for o in cat_obs:
                blob = _norm(f"{o.value or ''} {o.claim or ''} {o.evidence_quote or ''}")
                if any(k in blob for k in kws):
                    f = o.youtube_id or o.creator
                    p = period_of(o)
                    if f and (f not in first_seen or p < first_seen[f]):
                        first_seen[f] = p
            if not first_seen:
                continue
            cumulative = []
            run = 0
            for i, p in enumerate(periods):
                run += sum(1 for fp in first_seen.values() if pidx[fp] == i)
                cumulative.append(run)
            cat_series[canonical] = cumulative
            gained = [f for f, p in first_seen.items() if p == last]
            if gained and len(periods) > 1:
                trends.append({
                    "category": cat_name, "entity": canonical,
                    "from": cumulative[-2], "to": cumulative[-1],
                    "gained": len(gained), "period": last,
                })
        if cat_series:
            series[cat_name] = cat_series

    trends.sort(key=lambda t: -t["gained"])
    return {
        "granularity": granularity,
        "periods": periods,
        "series": series,
        "volume": volume,
        "founders_cumulative": founders_cumulative,
        "trends": trends,
    }
