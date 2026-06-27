"""Attribute observations to chunk timestamps and cluster them across founders.

Reads observations_claude.json, maps each evidence_quote back to a real chunk
(recovering youtube_id + timestamp), writes observations_all.json, and prints
the cross-founder clustering that is the whole point: "mentioned by N founders".
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict

from youtube_brain.config.settings import get_settings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BRAIN = "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
IN = "data/observations_claude.json"
OUT = "data/observations_all.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load_chunks_by_video(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.video_id yt, v.title title, c.id cid, c.start_time st, c.text txt "
        "FROM chunks c JOIN videos v ON v.id=c.video_id WHERE c.brain_id=?",
        (BRAIN,),
    ).fetchall()
    con.close()
    by_vid: dict = defaultdict(lambda: {"title": "", "chunks": []})
    for r in rows:
        by_vid[r["yt"]]["title"] = r["title"]
        by_vid[r["yt"]]["chunks"].append({"id": r["cid"], "start": r["st"], "text": _norm(r["txt"])})
    return by_vid


def attribute(quote: str, chunks: list[dict]) -> dict | None:
    q = _norm(quote)
    if len(q) < 15:
        return None
    for frag_len in (len(q), 60, 40, 25):
        frag = q[:frag_len]
        if len(frag) < 15:
            break
        for c in chunks:
            if frag in c["text"]:
                return {"chunk_id": c["id"], "start_time": c["start"]}
    return None


# Cross-founder theme clusters (keyword stand-in for embedding clustering).
THEMES = {
    "Tool: Claude Code / Claude": ["claude code", "cloud code", "claude as", "claude do"],
    "Tool: Cursor": ["cursor"],
    "Tool: Replit": ["replit"],
    "Tool: Supabase": ["supabase", "superbase"],
    "Tool: RevenueCat": ["revenuecat", "revenue cat", "revenue cap"],
    "Tool: Superwall": ["superwall"],
    "Channel: Reddit": ["reddit"],
    "Channel: Influencer marketing": ["influencer"],
    "Channel: UGC / faceless content": ["ugc", "faceless"],
    "Channel: Paid ads (Meta)": ["paid ads", "meta ads"],
    "Channel: SEO": ["seo"],
    "Channel: Word of mouth / organic": ["word of mouth", "organic"],
    "Channel: Product Hunt": ["product hunt"],
    "Strategy: Copy a proven idea / niche down": ["copy", "niche it down", "niche down", "proven", "10% better", "reinvent"],
    "Strategy: Validate before building": ["validate", "commitment metric", "mom test", "buy button"],
    "Strategy: Ship fast / keep shipping": ["keep shipping", "ship fast", "roll the dice", "number of things you ship", "ship this one out"],
    "Strategy: Distribution/marketing > product": ["distribution", "start from the marketing", "execution over", "didn't market"],
    "Strategy: Boring/unsexy or hyper-niche market": ["boring", "unsexy", "hidden", "nobody is building", "niche app", "niche category"],
    "Monetization: Subscription": ["subscription", "monthly", "yearly", "hard paywall", "mrr"],
}


def main() -> None:
    db = str(get_settings().database_path)
    by_vid = load_chunks_by_video(db)
    obs = json.loads(open(IN, encoding="utf-8").read())

    attributed = 0
    for o in obs:
        v = by_vid.get(o["youtube_id"], {"title": "", "chunks": []})
        o["video_title"] = v["title"]
        a = attribute(o.get("evidence_quote", ""), v["chunks"])
        o["chunk_id"] = a["chunk_id"] if a else None
        o["start_time"] = a["start_time"] if a else None
        o["attributed"] = a is not None
        if a:
            attributed += 1

    open(OUT, "w", encoding="utf-8").write(json.dumps(obs, indent=2, ensure_ascii=False))

    n = len(obs)
    print(f"Observations: {n}   attributed: {attributed}/{n} ({100*attributed//n}%)")
    print(f"Videos covered: {len(set(o['youtube_id'] for o in obs))}")

    by_type: dict = defaultdict(int)
    for o in obs:
        by_type[o["type"]] += 1
    print("\nBy type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {c}")

    print("\n" + "=" * 60)
    print("CROSS-FOUNDER CLUSTERS  (distinct founders mentioning each theme)")
    print("=" * 60)
    rows = []
    for theme, kws in THEMES.items():
        vids = set()
        for o in obs:
            blob = _norm(o.get("claim", "") + " " + str(o.get("value", "")) + " " + o.get("evidence_quote", ""))
            if any(k in blob for k in kws):
                vids.add(o["youtube_id"])
        rows.append((len(vids), theme))
    for n_founders, theme in sorted(rows, reverse=True):
        if n_founders == 0:
            continue
        bar = "#" * n_founders
        print(f"  {n_founders:2d}/10 {bar:11s} {theme}")

    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
