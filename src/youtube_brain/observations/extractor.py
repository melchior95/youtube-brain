"""Extract typed observations from a video's transcript via an LLM.

The quality-sensitive step. Returns Observation models with each evidence
quote attributed back to a chunk (recovering chunk_id + timestamp). The LLM
is a quality accelerator, not required infrastructure — this module is the
Gemini path; a transcript can equally be handed to a local model or to Claude.
"""

from __future__ import annotations

import logging
import re

from youtube_brain.core.models import Observation
from youtube_brain.llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

OBS_TYPES = [
    "acquisition_channel", "business_model", "monetization", "metric",
    "mistake", "tactic", "tool", "principle", "market",
]

SYSTEM = f"""You extract structured "observations" from startup founder interview transcripts.

An observation is a single, concrete, REUSABLE insight or fact the founder actually
stated — the kind of thing you could count across many interviews (e.g. "used Reddit
for first users", "charges a monthly subscription", "made $30K/month").

Return a JSON array. Each object MUST have:
  type           one of: {", ".join(OBS_TYPES)}
  claim          a short, canonical statement (normalize wording so the same idea
                 from different founders would read the same)
  entity         the subject (the founder, their app/company, or the method)
  value          the specific value if applicable (e.g. "Reddit", "$30K/month",
                 "subscription"); "" if none
  evidence_quote a VERBATIM snippet from the transcript that supports the claim
                 (copy it exactly, <=200 chars)
  confidence     0.0-1.0

STRICT RULES:
- Extract ONLY claims actually present in the transcript. Never infer or invent.
- evidence_quote must be copied verbatim from the transcript text provided.
- One observation per distinct idea; do not split one idea into near-duplicates.
- Aim for the 5-15 most concrete, countable observations. Skip filler.
Return ONLY the JSON array, no commentary."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def attribute(quote: str, chunks: list[dict]) -> dict | None:
    """Map an evidence quote back to the chunk that contains it (normalized)."""
    q = _norm(quote)
    if len(q) < 15:
        return None
    for frag_len in (len(q), 60, 40, 25):
        frag = q[:frag_len]
        if len(frag) < 15:
            break
        for c in chunks:
            if frag in _norm(c["text"]):
                return {"chunk_id": c["id"], "start_time": c["start_time"]}
    return None


async def extract_observations(
    client: GeminiClient,
    *,
    brain_id: str,
    video_id: str,
    youtube_id: str,
    creator: str,
    transcript: str,
    chunks: list[dict],
    domain: str = "founders",
) -> list[Observation]:
    """Extract and attribute observations for one video. Empty list on failure."""
    prompt = f"Video: {creator}\n\nTranscript:\n{transcript[:14000]}"
    try:
        result = await client.generate_json(prompt, system=SYSTEM, temperature=0.1)
    except Exception as e:
        logger.warning("Observation extraction failed for %s: %s", youtube_id, e)
        return []
    if not isinstance(result, list):
        return []

    out: list[Observation] = []
    for o in result:
        if not isinstance(o, dict) or not o.get("claim"):
            continue
        attr = attribute(o.get("evidence_quote", ""), chunks)
        out.append(Observation(
            brain_id=brain_id,
            video_id=video_id,
            youtube_id=youtube_id,
            creator=creator,
            obs_type=o.get("type", "other"),
            claim=o["claim"],
            value=o.get("value") or None,
            entities=[o["entity"]] if o.get("entity") else [],
            evidence_quote=o.get("evidence_quote"),
            chunk_id=attr["chunk_id"] if attr else None,
            start_time=attr["start_time"] if attr else None,
            confidence=o.get("confidence"),
            domain=domain,
        ))
    return out
