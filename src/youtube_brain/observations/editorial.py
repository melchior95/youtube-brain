"""Editorial layer — turn counts into a narrative people want to read.

The raw Intelligence Report is the audit layer (proves truthfulness). This is
the product layer: a magazine-style article with a lede, signal tiers, "what
surprised me" (including notable ABSENCES), where founders disagree, and an
evidence-derived playbook.

Discipline preserved: every number comes from the clustered observation data.
The deterministic scaffolding (tiers, absences, tension candidates) is computed
here; the LLM only narrates facts it is given — it must not invent counts.
"""

from __future__ import annotations

import math

from youtube_brain.core.models import Observation
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.report import build_intelligence
from youtube_brain.observations.rollup import compute_absences  # re-exported for callers


def signal_tier(founders: int, total: int) -> str:
    """Map a founder count to a labelled strength tier."""
    if total <= 0:
        return "Outlier"
    ratio = founders / total
    if ratio >= 0.45:
        return "Strong signal"
    if founders >= max(2, math.ceil(0.25 * total)):
        return "Emerging pattern"
    return "Outlier"


def build_brief(brain_name: str, observations: list[Observation]) -> dict:
    """Structured editorial brief: the only facts the writer is allowed to use."""
    intel = build_intelligence(brain_name, observations)
    total = intel["founders"]

    consensus = [
        {"theme": t["label"], "founders": t["founders"],
         "tier": signal_tier(t["founders"], total),
         "voices": [e["creator"] for e in t["evidence"]]}
        for t in intel["consensus"]
    ]

    rollups = {
        cat: [{"value": r["value"], "founders": r["founders"]} for r in rows]
        for cat, rows in intel["rollups"].items()
    }

    # Opinion-bearing observations help the writer find genuine tensions.
    opinions = [
        {"creator": o.creator or o.youtube_id, "type": o.obs_type,
         "claim": o.claim, "value": o.value}
        for o in observations
        if o.obs_type in ("principle", "mistake", "tactic", "business_model")
    ]

    return {
        "brain_name": brain_name,
        "total_founders": total,
        "total_observations": intel["total_observations"],
        "consensus": consensus,
        "rollups": rollups,
        "absences": compute_absences(observations),
        "opinions": opinions[:60],
    }


EDITORIAL_SYSTEM = """You are a sharp magazine writer turning research findings into a feature article.

You are given a JSON brief of findings mined from founder interviews: consensus
themes (with how many founders independently expressed each and a signal tier),
exact tool/channel/monetization adoption counts, notably ABSENT topics, and a
list of opinion-bearing observations.

Write a compelling Markdown article with these sections:

1. A title (magazine-style, specific — e.g. "What 10 Founders Who Built $20K–$3M
   Products Actually Agree On"). Then a strong lede that names the single
   STRONGEST pattern as a story, not a statistic.
2. ## The consensus — narrate the top themes. Lead with the most surprising one.
   Use the signal tiers (Strong signal / Emerging pattern / Outlier) and cite the
   founder counts. Make each theme a small story, not a bullet.
3. ## What surprised me — 4-6 memorable findings. INCLUDE the notable absences
   (what founders did NOT talk about) — these are often the most interesting.
4. ## Where founders disagree — find 2-3 GENUINE tensions in the opinions list
   (e.g. validate-before-building vs ship-to-validate; chase trends vs boring
   evergreen niche; use AI for everything vs craft by hand). Name the founders on
   each side. If you cannot find a real tension, say consensus was unusually broad.
5. ## If you were starting tomorrow — a tight, numbered playbook derived ONLY from
   the strongest recurring advice.

STRICT: Use ONLY facts and numbers present in the brief. Never invent a count, a
founder, a tool, or a quote. It is fine to interpret and synthesize, but every
number must trace to the brief. Keep it crisp and quotable."""


async def generate_editorial(
    client: GeminiClient, brain_name: str, observations: list[Observation]
) -> str:
    """Generate the editorial article from observations via an LLM."""
    import json

    brief = build_brief(brain_name, observations)
    prompt = "Findings brief (JSON):\n" + json.dumps(brief, ensure_ascii=False)
    return await client.generate(prompt, system=EDITORIAL_SYSTEM, temperature=0.6)
