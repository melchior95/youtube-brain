"""Generate an Intelligence Report from clustered observations.

Trust rule (mirrors our retrieval discipline): counts are computed from the
clustered observation data, never asked of an LLM. Every theme is backed by
attributable evidence (creator + timestamp + link). Prose only wraps numbers
that are real.

`build_intelligence` returns a structured payload (for the API / frontend);
`build_report` renders the same data as Markdown.
"""

from __future__ import annotations

from collections import defaultdict

from youtube_brain.core.models import Observation
from youtube_brain.observations.rollup import compute_absences, entity_rollup


def _ts(seconds: float | None) -> str:
    if seconds is None:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _yt_link(youtube_id: str | None, start: float | None) -> str:
    if not youtube_id:
        return ""
    t = f"?t={int(start)}" if start is not None else ""
    return f"https://youtu.be/{youtube_id}{t}"


def _ident(o: Observation) -> str:
    """Stable per-source identity for counting (one video = one founder)."""
    return o.youtube_id or o.creator or ""


def _display(o: Observation) -> str:
    return o.creator or o.youtube_id or "?"


def _norm_entity(entity: str) -> str:
    """Case/space-insensitive entity key so 'NVDA' and ' nvda ' group together."""
    return " ".join(entity.lower().split())


def consensus_themes(observations: list[Observation], ident=_ident) -> list[dict]:
    """Entity-grouped consensus: entities that >= 2 distinct sources talk about.

    Deterministic and embedding-free. Observations are grouped by the canonical
    `entities` Claude extracted at save time; a theme qualifies when >= 2 distinct
    sources land on the same entity. `ident` is the unit of agreement (default:
    one video = one source; cross-creator passes a per-channel ident). Counts are
    computed from persisted fields, never guessed.
    """
    total = len({ident(o) for o in observations})
    groups: dict[str, list[Observation]] = defaultdict(list)
    display: dict[str, str] = {}
    for o in observations:
        for e in o.entities or []:
            if not e or not e.strip():
                continue
            key = _norm_entity(e)
            if not key:
                continue
            groups[key].append(o)
            display.setdefault(key, e.strip())

    themes = []
    for key, obs in groups.items():
        # One evidence entry per distinct source (highest confidence each).
        by_source: dict[str, Observation] = {}
        for o in obs:
            k = ident(o)
            if k not in by_source or (o.confidence or 0) > (by_source[k].confidence or 0):
                by_source[k] = o
        if len(by_source) < 2:
            continue
        evidence = [
            {
                "creator": _display(o),
                "quote": (o.evidence_quote or "").strip(),
                "youtube_id": o.youtube_id,
                "start_time": o.start_time,
                "obs_type": o.obs_type,
            }
            for _, o in sorted(by_source.items(), key=lambda kv: _display(kv[1]))
        ]
        themes.append({
            "label": display[key],
            "founders": len(by_source),
            "total_founders": total,
            "evidence": evidence,
        })
    themes.sort(key=lambda t: (-t["founders"], -len(t["evidence"])))
    return themes


def build_intelligence(brain_name: str, observations: list[Observation]) -> dict:
    """Structured intelligence payload for the API / frontend."""
    by_type: dict[str, int] = defaultdict(int)
    for o in observations:
        by_type[o.obs_type] += 1
    return {
        "brain_name": brain_name,
        "total_observations": len(observations),
        "founders": len({_ident(o) for o in observations}),
        "consensus": consensus_themes(observations),
        "rollups": entity_rollup(observations),
        "absences": compute_absences(observations),
        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
    }


def build_report(brain_name: str, observations: list[Observation]) -> str:
    """Render a Markdown intelligence report from clustered observations."""
    intel = build_intelligence(brain_name, observations)
    n_f = intel["founders"]
    lines: list[str] = [
        f"# {brain_name}: Intelligence Report",
        "",
        f"*Generated from {intel['total_observations']} attributable observations "
        f"across {n_f} founders.*",
        "",
        "## Consensus: what multiple founders independently said",
        "",
    ]

    if not intel["consensus"]:
        lines.append("_No multi-founder themes found._")
    for theme in intel["consensus"]:
        lines.append(f"### {theme['label']}: {theme['founders']}/{n_f} founders")
        for e in theme["evidence"]:
            quote = e["quote"]
            if len(quote) > 160:
                quote = quote[:157] + "…"
            link = _yt_link(e["youtube_id"], e["start_time"])
            loc = f" ([{_ts(e['start_time'])}]({link}))" if link else ""
            lines.append(f"- **{e['creator']}**: \"{quote}\"{loc}")
        lines.append("")

    # Entity rollups, exact counts for concrete entities.
    for cat, rows in intel["rollups"].items():
        if not rows:
            continue
        lines.append(f"## {cat} by adoption")
        lines.append("")
        for r in rows:
            creators = ", ".join(e["creator"] for e in r["evidence"])
            lines.append(f"- **{r['value']}**, {r['founders']} founder"
                         f"{'s' if r['founders'] != 1 else ''} ({creators})")
        lines.append("")

    lines.append("## Observation coverage by type")
    lines.append("")
    for t, c in intel["by_type"].items():
        lines.append(f"- **{t}**: {c}")
    lines.append("")
    return "\n".join(lines)
