# tests/test_categories.py
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from youtube_brain.core.models import Brain, Source
from youtube_brain.core.enums import SourceType, SourceStatus
from youtube_brain.storage.brains import insert_brain
from youtube_brain.categories import load_categories, get_category, brains_by_channel_id


async def _seed_brain_with_channel(channel_id, name):
    brain = Brain(name=name)
    await insert_brain(brain)
    src = Source(brain_id=brain.id, source_type=SourceType.CHANNEL,
                 source_url="x", source_id=channel_id, status=SourceStatus.RESOLVED)
    from youtube_brain.storage.database import get_session, sources as sources_table
    from sqlalchemy.dialects.sqlite import insert
    async with get_session() as s:
        await s.execute(insert(sources_table).values(
            id=str(src.id), brain_id=str(brain.id), source_type="channel",
            source_url="x", source_id=channel_id, status="ready",
            created_at=src.created_at))
    return str(brain.id)


def test_load_categories_from_path():
    d = Path(tempfile.mkdtemp(prefix="ytbrain-cat-"))
    try:
        p = d / "categories.json"
        p.write_text(json.dumps([{"slug": "s", "name": "S", "description": "d",
                                  "creators": [{"handle": "@a", "url": "u", "channel_id": "UC1"}]}]))
        cats = load_categories(p)
        assert cats[0].slug == "s"
        assert cats[0].creators[0].channel_id == "UC1"
        assert get_category("s", p).name == "S"
    finally:
        shutil.rmtree(d, ignore_errors=True)


async def test_brains_by_channel_id_matches_pulled():
    bid = await _seed_brain_with_channel("UCZZZ", "Creator Z")
    found = await brains_by_channel_id(["UCZZZ", "UCNOPE"])
    assert "UCZZZ" in found and found["UCZZZ"]["brain_id"] == bid
    assert "UCNOPE" not in found
