"""Backfill videos.published_at for a brain via yt-dlp (no LLM cost).

New ingests capture publish dates automatically; this fixes brains ingested
before that. Run: python scripts/backfill_published_dates.py [BRAIN_ID]
"""

import json
import sqlite3
import subprocess
import sys

from youtube_brain.config.settings import get_settings
from youtube_brain.ingest.resolver import _parse_published

BRAIN = sys.argv[1] if len(sys.argv) > 1 else "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"


def main() -> None:
    db = str(get_settings().database_path)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, video_id FROM videos WHERE brain_id=? AND published_at IS NULL",
        (BRAIN,),
    ).fetchall()
    updated = 0
    for v in rows:
        try:
            out = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-warnings", "--quiet", "--skip-download",
                 f"https://www.youtube.com/watch?v={v['video_id']}"],
                capture_output=True, text=True, timeout=60, check=True,
            )
            entry = json.loads(out.stdout.strip().splitlines()[0])
            pub = _parse_published(entry)
            if pub:
                con.execute("UPDATE videos SET published_at=? WHERE id=?", (pub, v["id"]))
                updated += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {v['video_id']}: {e}")
    con.commit()
    con.close()
    print(f"Backfilled published_at for {updated}/{len(rows)} videos.")


if __name__ == "__main__":
    main()
