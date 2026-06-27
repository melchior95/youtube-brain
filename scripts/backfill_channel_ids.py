"""One-off: stamp each existing brain's source with its stable channel_id.

Brains created before channel_id keying carry a handle/video-id in
sources.source_id. This resolves the real channel_id from one of each brain's
videos (yt-dlp) and writes it back, so future pulls of the same creator merge
by id instead of by brittle display-name/handle matching.

Usage:  python scripts/backfill_channel_ids.py [--apply]
"""

from __future__ import annotations

import sqlite3
import sys

from youtube_brain.config.settings import get_settings
from youtube_brain.ingest.resolver import _resolve_single_video

APPLY = "--apply" in sys.argv


def main() -> None:
    con = sqlite3.connect(str(get_settings().database_path))
    con.row_factory = sqlite3.Row
    brains = con.execute("SELECT id, name FROM brains ORDER BY name").fetchall()

    for b in brains:
        vid = con.execute(
            "SELECT video_id FROM videos WHERE brain_id=? LIMIT 1", (b["id"],)
        ).fetchone()
        if not vid:
            print(f"  {b['name']:24} — no videos, skip")
            continue
        meta = _resolve_single_video(vid["video_id"])
        cid = meta.get("channel_id")
        if not cid:
            print(f"  {b['name']:24} — could not resolve channel_id, skip")
            continue
        cur = con.execute(
            "SELECT source_id FROM sources WHERE brain_id=? LIMIT 1", (b["id"],)
        ).fetchone()
        already = cur and cur["source_id"] == cid
        print(f"  {b['name']:24} -> {cid}" + ("  (already set)" if already else ""))
        if APPLY and not already:
            con.execute("UPDATE sources SET source_id=? WHERE brain_id=?", (cid, b["id"]))

    if APPLY:
        con.commit()
        print("\nApplied.")
    else:
        print("\n[dry-run] re-run with --apply to write channel_ids.")
    con.close()


if __name__ == "__main__":
    main()
