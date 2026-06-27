"""Dump a brain's video transcripts to JSON for in-session (Claude) extraction.

Keeps the file small (title + truncated transcript only) — chunk attribution
is done separately against the DB by attribute_observations.py.
"""

import json
import sqlite3
import sys

from youtube_brain.config.settings import get_settings

BRAIN = sys.argv[1] if len(sys.argv) > 1 else "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
OUT = "data/transcripts_for_extraction.json"
MAXLEN = 14000

con = sqlite3.connect(str(get_settings().database_path))
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT video_id, title, transcript_clean FROM videos "
    "WHERE brain_id=? AND transcript_clean IS NOT NULL ORDER BY created_at",
    (BRAIN,),
).fetchall()
con.close()

data = [
    {"youtube_id": r["video_id"], "title": r["title"],
     "transcript": (r["transcript_clean"] or "")[:MAXLEN]}
    for r in rows
]
open(OUT, "w", encoding="utf-8").write(json.dumps(data, indent=2, ensure_ascii=False))
print(f"Dumped {len(data)} transcripts to {OUT}")
for d in data:
    print(f"  {d['youtube_id']}  {len(d['transcript'])} chars  {d['title'][:50]}")
