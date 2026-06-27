"""Observation-extraction SPIKE — founders domain.

Standalone experiment (NOT wired into the production pipeline) to validate
whether we can reliably extract typed, countable, attributable "observations"
from founder-interview transcripts. Evaluate the output against three gates:

  1. Faithfulness — is each observation actually supported by the transcript?
  2. Attribution  — does its evidence_quote map back to a real chunk+timestamp?
  3. Granularity  — do near-duplicate claims collapse, or inflate counts?

Usage:
    python scripts/extract_observations.py [BRAIN_ID] [--limit N] [--out PATH]

Reads videos+chunks straight from the SQLite DB; writes a JSON array plus a
console summary. Costs ~1 Gemini generate call per video.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so prints never crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from youtube_brain.config.settings import get_settings
from youtube_brain.llm.gemini import GeminiClient

DEFAULT_BRAIN = "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"  # Starter Story, 10 videos

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
- Extract ONLY claims actually present in the transcript. Never infer, generalize,
  or invent. If unsure, leave it out.
- evidence_quote must be copied verbatim from the transcript text provided.
- One observation per distinct idea. Do not split one idea into near-duplicates.
- Aim for the 5-15 most concrete, countable observations per video. Skip filler.
Return ONLY the JSON array, no commentary."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load_videos(db_path: str, brain_id: str, limit: int | None) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    vids = con.execute(
        "SELECT id, video_id, title, transcript_clean FROM videos "
        "WHERE brain_id=? AND transcript_clean IS NOT NULL ORDER BY created_at",
        (brain_id,),
    ).fetchall()
    if limit:
        vids = vids[:limit]
    out = []
    for v in vids:
        chunks = con.execute(
            "SELECT id, start_time, text FROM chunks WHERE video_id=? ORDER BY start_time",
            (v["id"],),
        ).fetchall()
        out.append({
            "id": v["id"],
            "youtube_id": v["video_id"],
            "title": v["title"],
            "transcript": v["transcript_clean"],
            "chunks": [{"id": c["id"], "start_time": c["start_time"], "text": c["text"]}
                       for c in chunks],
        })
    con.close()
    return out


def attribute(quote: str, chunks: list[dict]) -> dict | None:
    """Map an evidence_quote back to the chunk that contains it (normalized)."""
    q = _norm(quote)
    if not q:
        return None
    # Try a shrinking prefix so minor tail differences still match.
    for frag_len in (len(q), 80, 50, 30):
        frag = q[:frag_len]
        if len(frag) < 20:
            break
        for c in chunks:
            if frag in _norm(c["text"]):
                return {"chunk_id": c["id"], "start_time": c["start_time"]}
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("brain_id", nargs="?", default=DEFAULT_BRAIN)
    ap.add_argument("--limit", type=int, default=None, help="Max videos to process")
    ap.add_argument("--out", default="data/observations.json")
    args = ap.parse_args()

    db_path = str(get_settings().database_path)
    videos = load_videos(db_path, args.brain_id, args.limit)
    print(f"Loaded {len(videos)} videos from brain {args.brain_id[:8]}")

    client = GeminiClient()
    all_obs: list[dict] = []
    attributed = 0
    try:
        for v in videos:
            transcript = v["transcript"][:14000]
            prompt = f"Video title: {v['title']}\n\nTranscript:\n{transcript}"
            try:
                result = await client.generate_json(prompt, system=SYSTEM, temperature=0.1)
            except Exception as e:
                print(f"  ! {v['title'][:50]}: extraction failed - {e}")
                continue
            if not isinstance(result, list):
                print(f"  ! {v['title'][:50]}: non-list result, skipping")
                continue
            for obs in result:
                if not isinstance(obs, dict):
                    continue
                attr = attribute(obs.get("evidence_quote", ""), v["chunks"])
                if attr:
                    attributed += 1
                obs["video_title"] = v["title"]
                obs["youtube_id"] = v["youtube_id"]
                obs["chunk_id"] = attr["chunk_id"] if attr else None
                obs["start_time"] = attr["start_time"] if attr else None
                obs["attributed"] = attr is not None
                all_obs.append(obs)
            print(f"  + {v['title'][:50]}: {len(result)} observations")
    finally:
        await client.close()

    Path(args.out).write_text(json.dumps(all_obs, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    by_type: dict[str, int] = {}
    for o in all_obs:
        by_type[o.get("type", "?")] = by_type.get(o.get("type", "?"), 0) + 1
    print(f"\n{'='*55}")
    print(f"Total observations: {len(all_obs)}")
    if all_obs:
        print(f"Attributed to a chunk+timestamp: {attributed}/{len(all_obs)} "
              f"({100*attributed//len(all_obs)}%)")
    print("By type:")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:22s} {n}")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
