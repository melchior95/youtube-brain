"""Generate an Intelligence Report from clustered observations.

Trust rule (mirrors our retrieval discipline): counts are computed from the
clustered observation data — never asked of an LLM. Every theme is backed by
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


def consensus_themes(observations: list[Observation]) -> list[dict]:
    """Clusters spanning >= 2 distinct founders, as structured data."""
    total_founders = len({_ident(o) for o in observations})
    clusters: dict[int, list[Observation]] = defaultdict(list)
    for o in observations:
        clusters[o.cluster_id if o.cluster_id is not None else -1].append(o)

    themes = []
    for obs in clusters.values():
        creators = {_ident(o) for o in obs}
        if len(creators) < 2:
            continue

        types = [o.obs_type for o in obs]
        metric_frac = sum(1 for t in types if t == "metric") / len(types)
        if metric_frac >= 0.6:
            label = "Revenue & traction milestones (every founder is a success story)"
        else:
            rep = max(obs, key=lambda o: ((o.confidence or 0), -len(o.claim)))
            label = rep.claim

        # One evidence entry per distinct founder (highest confidence each).
        by_creator: dict[str, Observation] = {}
        for o in obs:
            key = _ident(o)
            if key not in by_creator or (o.confidence or 0) > (by_creator[key].confidence or 0):
                by_creator[key] = o
        evidence = [
            {
                "creator": _display(o),
                "quote": (o.evidence_quote or "").strip(),
                "youtube_id": o.youtube_id,
                "start_time": o.start_time,
                "obs_type": o.obs_type,
            }
            for _, o in sorted(by_creator.items(), key=lambda kv: _display(kv[1]))
        ]
        themes.append({
            "label": label,
            "founders": len(creators),
            "total_founders": total_founders,
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
        f"# {brain_name} — Intelligence Report",
        "",
        f"*Generated from {intel['total_observations']} attributable observations "
        f"across {n_f} founders.*",
        "",
        "## Consensus — what multiple founders independently said",
        "",
    ]

    if not intel["consensus"]:
        lines.append("_No multi-founder themes found._")
    for theme in intel["consensus"]:
        lines.append(f"### {theme['label']}  —  {theme['founders']}/{n_f} founders")
        for e in theme["evidence"]:
            quote = e["quote"]
            if len(quote) > 160:
                quote = quote[:157] + "…"
            link = _yt_link(e["youtube_id"], e["start_time"])
            loc = f" ([{_ts(e['start_time'])}]({link}))" if link else ""
            lines.append(f"- **{e['creator']}**: \"{quote}\"{loc}")
        lines.append("")

    # Entity rollups — exact counts for concrete entities.
    for cat, rows in intel["rollups"].items():
        if not rows:
            continue
        lines.append(f"## {cat} by adoption")
        lines.append("")
        for r in rows:
            creators = ", ".join(e["creator"] for e in r["evidence"])
            lines.append(f"- **{r['value']}** — {r['founders']} founder"
                         f"{'s' if r['founders'] != 1 else ''} ({creators})")
        lines.append("")

    lines.append("## Observation coverage by type")
    lines.append("")
    for t, c in intel["by_type"].items():
        lines.append(f"- **{t}**: {c}")
    lines.append("")
    return "\n".join(lines)
