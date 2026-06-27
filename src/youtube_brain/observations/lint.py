"""Lint persisted observations for tension — the dual of the consensus report.

`report` surfaces clusters where >=2 sources AGREE. ``lint_candidates`` surfaces
the opposite: entities where sources DIVERGE, or where a claim has been
superseded over time. It is the deterministic *narrowing* step — it cannot
decide whether a pair truly contradicts (claims are natural language), so it
groups observations by shared entity, orders them by date, and emits only groups
with real tension potential (>=2 distinct sources, or spanning >=2 dates).

Claude adjudicates each candidate in-loop (contradiction / evolution / stale /
consistent), exactly as it answers questions from retrieved chunks. Zero
generate — this module only reshapes already-persisted observations.

Entity (exact, normalized) is the grouping axis because it is the most reliable
signal that two observations are *about the same thing*. Semantic grouping via
``Observation.cluster_id`` is a possible future axis (catches paraphrased
entities) but is intentionally out of scope here to keep the pass cheap and pure.
"""

from __future__ import annotations

from collections import defaultdict

from youtube_brain.core.models import Observation


def _yt_link(youtube_id: str | None, start: float | None) -> str | None:
    if not youtube_id or start is None:
        return None
    return f"https://youtu.be/{youtube_id}?t={int(start)}"


def _norm(entity: str) -> str:
    """Case/space-insensitive entity key so 'NVDA' and ' nvda ' group together."""
    return " ".join(entity.lower().split())


def lint_candidates(
    observations: list[Observation],
    date_map: dict[str, str | None],
    max_groups: int = 40,
) -> tuple[list[dict], int]:
    """Group observations by shared entity and return tension candidates.

    ``date_map`` maps youtube_id -> published date ("YYYY-MM-DD") or None; it
    orders members chronologically and detects time-spanning evolution. A group
    qualifies as a candidate only when it has >=2 members AND either spans >=2
    distinct sources (cross-source contradiction) or >=2 distinct dates
    (single-source evolution / staleness) — single mentions and lone restatements
    are dropped.

    Returns ``(candidates, total)`` where ``candidates`` is capped at
    ``max_groups`` (most-contested first) and ``total`` is the full count before
    capping, so the caller can report silent truncation.
    """
    groups: dict[str, list[Observation]] = defaultdict(list)
    display: dict[str, str] = {}
    for o in observations:
        for e in o.entities or []:
            if not e or not e.strip():
                continue
            key = _norm(e)
            if not key:
                continue
            groups[key].append(o)
            display.setdefault(key, e.strip())

    candidates: list[dict] = []
    for key, obs in groups.items():
        if len(obs) < 2:
            continue
        sources = {o.youtube_id or o.creator for o in obs}
        dates = {date_map.get(o.youtube_id) for o in obs if date_map.get(o.youtube_id)}
        if len(sources) < 2 and len(dates) < 2:
            continue
        members = sorted(obs, key=lambda o: (date_map.get(o.youtube_id) or "", o.creator or ""))
        candidates.append({
            "entity": display[key],
            "distinct_sources": len(sources),
            "distinct_dates": len(dates),
            "values": sorted({o.value for o in obs if o.value}),
            "observations": [
                {
                    "creator": o.creator,
                    "type": o.obs_type,
                    "claim": o.claim,
                    "value": o.value,
                    "published": date_map.get(o.youtube_id),
                    "confidence": o.confidence,
                    "evidence_quote": o.evidence_quote,
                    "citation": _yt_link(o.youtube_id, o.start_time),
                }
                for o in members
            ],
        })

    total = len(candidates)
    candidates.sort(
        key=lambda c: (-c["distinct_sources"], -c["distinct_dates"], -len(c["observations"]))
    )
    return candidates[:max_groups], total
