"""Lite-ingest a spread of back-catalog videos to stretch the timeline.

Samples across the channel's back-catalog (newest-first, so sampling by
position maximizes date span), lite-ingests them (transcript + chunk + embed,
ZERO Gemini generate), then backfills publish dates. Observations are added
separately, Claude-in-loop.

Run: python scripts/ingest_backcatalog.py [N]
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys

from youtube_brain.config.settings import get_settings
from youtube_brain.ingest.pipeline import ingest_url
from youtube_brain.ingest.resolver import _parse_published
from youtube_brain.observations.refresh import find_new_videos
from youtube_brain.storage.brains import get_brain
from youtube_brain.storage.database import init_database

BRAIN = "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def backfill_dates(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, video_id FROM videos WHERE brain_id=? AND published_at IS NULL",
        (BRAIN,),
    ).fetchall()
    n = 0
    for v in rows:
        try:
            out = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-warnings", "--quiet", "--skip-download",
                 f"https://www.youtube.com/watch?v={v['video_id']}"],
                capture_output=True, text=True, timeout=60, check=True,
            )
            pub = _parse_published(json.loads(out.stdout.strip().splitlines()[0]))
            if pub:
                con.execute("UPDATE videos SET published_at=? WHERE id=?", (pub, v["id"]))
                n += 1
        except Exception:
            pass
    con.commit()
    con.close()
    return n


async def main() -> None:
    await init_database()
    db_path = str(get_settings().database_path)

    new, source = await find_new_videos(BRAIN)
    if not new:
        print("No back-catalog videos available.")
        return

    step = max(1, len(new) // N)
    sample = new[::step][:N]
    print(f"Back-catalog: {len(new)} available; sampling {len(sample)} across history (step {step}).")

    brain = await get_brain(BRAIN)
    res = await ingest_url(
        source.source_url, existing_brain=brain, existing_source=source,
        video_metas=sample, generate_metadata=False,
    )
    print(f"Lite-ingested: {res.videos_processed} videos, {res.chunks_created} chunks "
          f"(zero generate calls). Errors: {len(res.errors)}")

    dated = backfill_dates(db_path)
    print(f"Backfilled dates for {dated} videos.")


if __name__ == "__main__":
    asyncio.run(main())
