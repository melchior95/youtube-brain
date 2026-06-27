"""Resolve channel_ids for categories.json and report pulled vs pending.

Usage:  python scripts/seed_categories.py            # report only
        python scripts/seed_categories.py --write     # also fill missing channel_ids
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from youtube_brain.categories import _CONFIG_PATH, brains_by_channel_id, load_categories
from youtube_brain.ingest.resolver import _resolve_single_video, parse_youtube_url, resolve_video_ids
from youtube_brain.storage.database import init_database

WRITE = "--write" in sys.argv


def _resolve_channel_id(url: str) -> str | None:
    try:
        metas = resolve_video_ids(parse_youtube_url(url))
        if not metas:
            return None
        full = _resolve_single_video(metas[0]["video_id"])
        return full.get("channel_id")
    except Exception:
        return None


async def main() -> None:
    await init_database()
    raw = json.loads(Path(_CONFIG_PATH).read_text(encoding="utf-8"))
    for cat in raw:
        for cr in cat["creators"]:
            if not cr.get("channel_id"):
                cid = _resolve_channel_id(cr["url"])
                if cid:
                    cr["channel_id"] = cid
                    print(f"  resolved {cr['handle']} -> {cid}")
    if WRITE:
        Path(_CONFIG_PATH).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote channel_ids back to categories.json")

    cats = load_categories()
    cids = [cr.channel_id for c in cats for cr in c.creators if cr.channel_id]
    resolved = await brains_by_channel_id(cids)
    for c in cats:
        print(f"\n{c.name}:")
        for cr in c.creators:
            state = "PULLED " if cr.channel_id and cr.channel_id in resolved else "pending"
            print(f"  [{state}] {cr.handle}")


if __name__ == "__main__":
    asyncio.run(main())
