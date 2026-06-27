"""One-off maintenance: delete junk/test brains and cascade their dependent rows.

Junk = brains with zero videos, or known test-fixture names. Cascades across
every brain-scoped table (and the embedding child tables keyed by chunk/obs id)
so nothing is orphaned. The chunks delete fires the FTS5 'delete' trigger, so
chunks_fts stays consistent.

Usage:
    python scripts/cleanup_brains.py            # dry-run: print what WOULD go
    python scripts/cleanup_brains.py --apply    # actually delete
"""

from __future__ import annotations

import sqlite3
import sys

from youtube_brain.config.settings import get_settings

JUNK_NAMES = ("My Brain", "Video dQw4w9WgXcQ", "Rick Astley")
APPLY = "--apply" in sys.argv


def main() -> None:
    con = sqlite3.connect(str(get_settings().database_path))
    con.row_factory = sqlite3.Row

    placeholders = ",".join("?" for _ in JUNK_NAMES)
    # Junk = no videos, known fixture names, or the legacy "Video <id>" per-video
    # fallback naming (pre channel-grouping). Real channel brains never match.
    cond = f"(video_count = 0 OR name IN ({placeholders}) OR name LIKE 'Video %')"
    junk = con.execute(
        f"SELECT id, name, video_count, status FROM brains WHERE {cond}",
        JUNK_NAMES,
    ).fetchall()

    keep = con.execute(
        f"SELECT id, name, video_count FROM brains WHERE NOT {cond} "
        f"ORDER BY video_count DESC",
        JUNK_NAMES,
    ).fetchall()

    print(f"DELETE {len(junk)} junk brains; KEEP {len(keep)}.")
    print("\nKEEPING:")
    for b in keep:
        print(f"  {b['video_count']:>3} videos  {b['name']}  ({b['id']})")

    if not junk:
        print("\nNothing to delete.")
        con.close()
        return

    ids = [b["id"] for b in junk]
    idq = ",".join("?" for _ in ids)

    if not APPLY:
        print(f"\n[dry-run] Re-run with --apply to delete {len(ids)} brains and their rows.")
        con.close()
        return

    cur = con.cursor()
    # Children first (embedding tables keyed by chunk/observation id), then the
    # brain-scoped tables, then the brains themselves.
    cur.execute(
        f"DELETE FROM chunk_embeddings WHERE chunk_id IN "
        f"(SELECT id FROM chunks WHERE brain_id IN ({idq}))", ids)
    cur.execute(
        f"DELETE FROM observation_embeddings WHERE observation_id IN "
        f"(SELECT id FROM observations WHERE brain_id IN ({idq}))", ids)
    for table in ("chunks", "observations", "videos", "sources", "articles",
                  "watchlist_schedules", "brains"):
        col = "id" if table == "brains" else "brain_id"
        cur.execute(f"DELETE FROM {table} WHERE {col} IN ({idq})", ids)

    con.commit()
    remaining = con.execute("SELECT COUNT(*) AS n FROM brains").fetchone()["n"]
    print(f"\nDeleted {len(ids)} brains. Remaining brains: {remaining}.")
    con.close()


if __name__ == "__main__":
    main()
