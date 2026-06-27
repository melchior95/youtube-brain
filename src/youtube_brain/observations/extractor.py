"""Attribute an evidence quote back to the chunk it came from.

Observation *extraction* is Claude's job (it reads the transcript and writes the
observations JSON that the bridge's `save` persists). This module only does the
deterministic part: mapping a verbatim evidence quote back to the chunk that
contains it, which recovers the chunk_id and a `youtu.be?t=` timestamp. No
external API.
"""

from __future__ import annotations

import re

OBS_TYPES = [
    "acquisition_channel", "business_model", "monetization", "metric",
    "mistake", "tactic", "tool", "principle", "market",
]


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
