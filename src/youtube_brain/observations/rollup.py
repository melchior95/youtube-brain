"""Entity rollups — exact "N founders" counts for concrete entities.

Embedding clustering groups by sentence-topic (good for fuzzy strategy themes)
but blurs distinct entities ("uses Cursor" vs "uses Supabase" both read as
"uses tool X"). For tools / channels / monetization the useful count is by the
specific entity, so we tally distinct founders per canonical value via keyword
matching. This complements (does not replace) embedding clustering.
"""

from __future__ import annotations

import re

from youtube_brain.core.models import Observation

TOOLS = {
    "Claude Code / Claude": ["claude code", "cloud code", "claude as", "had claude", "claude do"],
    "Cursor": ["cursor"],
    "Replit": ["replit"],
    "Supabase": ["supabase", "superbase"],
    "Firebase": ["firebase"],
    "RevenueCat": ["revenuecat", "revenue cat", "revenue cap"],
    "Superwall": ["superwall"],
    "Mixpanel": ["mixpanel", "mix panel"],
    "Bubble": ["bubble"],
    "Vercel": ["vercel"],
    "Figma": ["figma"],
    "ChatGPT / OpenAI": ["chatgpt", "chat gpt", "openai", "open ai", "chad gbt", "height gpt"],
}

CHANNELS = {
    "Reddit": ["reddit"],
    "Influencer marketing": ["influencer"],
    "UGC / faceless content": ["ugc", "faceless"],
    "Paid ads (Meta)": ["paid ads", "meta ads"],
    "SEO": ["seo"],
    "Product Hunt": ["product hunt"],
    "Word of mouth": ["word of mouth"],
    "Organic / App Store search": ["organic", "app store"],
    "TikTok / Instagram": ["tiktok", "tik tok", "instagram"],
    "Cold outreach / personal network": ["cold messag", "cold mail", "personal network", "dm "],
}

MONETIZATION = {
    "Subscription": ["subscription", "monthly", "yearly", "weekly", "mrr"],
    "Hard paywall": ["hard paywall"],
    "Usage / credit-based": ["usage-based", "usage based", "credit"],
    "Lifetime deal": ["lifetime deal"],
    "Free trial": ["free trial"],
    "Paid-only (no free plan)": ["no free plan", "zero free", "paid users only", "paid plans only"],
}

CATEGORIES = {
    "Tools": (("tool",), TOOLS),
    "Acquisition channels": (("acquisition_channel", "tactic"), CHANNELS),
    "Monetization": (("monetization", "metric"), MONETIZATION),
}

# Topics a reader would *expect* from "startup founders" — surfacing their
# absence is often more memorable than any present theme.
EXPECTED_TOPICS = {
    "Raising venture capital": ["venture", " vc ", "raise a round", "investor", "seed round", "fundrais"],
    "Hiring a team": ["hire", "employees", "headcount", "team of "],
    "App Store Optimization (ASO)": ["aso", "app store optimization", "keyword ranking"],
    "Cold email outreach": ["cold email", "cold outreach"],
    "Patents / moats / defensibility": ["patent", "moat", "defensib"],
    "Profitability / unit economics": ["unit economics", "burn rate", "runway", "gross margin"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def compute_absences(observations: list[Observation]) -> list[dict]:
    """Expected topics that are absent or rare (< 2 founders)."""
    out = []
    for topic, kws in EXPECTED_TOPICS.items():
        founders = set()
        for o in observations:
            blob = _norm(f"{o.value or ''} {o.claim or ''} {o.evidence_quote or ''}")
            if any(k in blob for k in kws):
                founders.add(o.youtube_id or o.creator)
        if len(founders) < 2:
            out.append({"topic": topic, "founders": len(founders)})
    return out


def entity_rollup(observations: list[Observation]) -> dict[str, list[dict]]:
    """Return {category: [{value, founders, evidence:[...]}]} sorted by founders."""
    out: dict[str, list[dict]] = {}
    for cat_name, (types, ents) in CATEGORIES.items():
        cat_obs = [o for o in observations if o.obs_type in types]
        rows: list[dict] = []
        for canonical, kws in ents.items():
            by_creator: dict[str, Observation] = {}
            for o in cat_obs:
                blob = _norm(f"{o.value or ''} {o.claim or ''} {o.evidence_quote or ''}")
                if any(k in blob for k in kws):
                    key = o.youtube_id or o.creator or "?"  # one video = one founder
                    if key not in by_creator or (o.confidence or 0) > (by_creator[key].confidence or 0):
                        by_creator[key] = o
            if by_creator:
                rows.append({
                    "value": canonical,
                    "founders": len(by_creator),
                    "evidence": [
                        {
                            "creator": o.creator or o.youtube_id,
                            "quote": (o.evidence_quote or "").strip(),
                            "youtube_id": o.youtube_id,
                            "start_time": o.start_time,
                        }
                        for o in sorted(by_creator.values(), key=lambda x: (x.creator or x.youtube_id or ""))
                    ],
                })
        rows.sort(key=lambda r: -r["founders"])
        out[cat_name] = rows
    return out
