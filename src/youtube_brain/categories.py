"""Category config loader + creator->brain matching (no DB tables)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import text

from youtube_brain.storage.database import get_session

_CONFIG_PATH = Path(__file__).parent / "config" / "categories.json"


class Creator(BaseModel):
    handle: str
    url: str
    channel_id: str | None = None


class Category(BaseModel):
    slug: str
    name: str
    description: str = ""
    creators: list[Creator] = []


def load_categories(path: Path | None = None) -> list[Category]:
    p = path or _CONFIG_PATH
    if not p.exists():
        return []
    return [Category(**c) for c in json.loads(p.read_text(encoding="utf-8"))]


def get_category(slug: str, path: Path | None = None) -> Category | None:
    return next((c for c in load_categories(path) if c.slug == slug), None)


async def brains_by_channel_id(channel_ids: list[str]) -> dict[str, dict]:
    """{channel_id: brain summary} for channel_ids that have a pulled brain."""
    ids = [c for c in channel_ids if c]
    if not ids:
        return {}
    binds = {f"c{i}": c for i, c in enumerate(ids)}
    inc = ",".join(f":c{i}" for i in range(len(ids)))
    sql = text(
        "SELECT s.source_id AS cid, b.id AS bid, b.name AS name, b.status AS status, "
        "       b.video_count AS video_count, "
        "       (SELECT v.video_id FROM videos v WHERE v.brain_id = b.id "
        "        ORDER BY v.published_at DESC, v.created_at DESC LIMIT 1) AS latest_video "
        f"FROM sources s JOIN brains b ON b.id = s.brain_id WHERE s.source_id IN ({inc})"
    )
    async with get_session() as session:
        rows = (await session.execute(sql, binds)).fetchall()
    return {
        r.cid: {"brain_id": r.bid, "name": r.name, "status": r.status,
                "video_count": r.video_count, "latest_video": r.latest_video}
        for r in rows
    }
