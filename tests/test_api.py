"""Tests for the FastAPI endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from youtube_brain.api.app import create_app
from youtube_brain.storage.database import init_database


@pytest.fixture
async def client(tmp_settings):
    await init_database(tmp_settings)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_list_brains_empty(client):
    resp = await client.get("/api/brains")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_brain_not_found(client):
    resp = await client.get("/api/brains/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Brain not found"


async def test_ingest_returns_200(client):
    resp = await client.post(
        "/api/brains/ingest",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingesting"
    assert data["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


async def test_ingest_with_name(client):
    resp = await client.post(
        "/api/brains/ingest",
        json={"url": "https://www.youtube.com/watch?v=test123", "name": "My Brain"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingesting"
    assert data["url"] == "https://www.youtube.com/watch?v=test123"


async def test_ask_brain_not_found(client):
    resp = await client.post(
        "/api/brains/00000000-0000-0000-0000-000000000000/ask",
        json={"query": "What is this about?"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Brain not found"


async def test_intelligence_not_found(client):
    resp = await client.get(
        "/api/brains/00000000-0000-0000-0000-000000000000/intelligence"
    )
    assert resp.status_code == 404


async def test_intelligence_empty(client):
    from youtube_brain.core.models import Brain
    from youtube_brain.storage.brains import insert_brain

    brain = Brain(name="Empty Brain")
    await insert_brain(brain)
    resp = await client.get(f"/api/brains/{brain.id}/intelligence")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_observations"] == 0
    assert data["consensus"] == []
    assert "rollups" in data and "by_type" in data


async def test_categories_endpoints(client, monkeypatch):
    import shutil
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp(prefix="ytbrain-cat-"))
    try:
        cfg = d / "categories.json"
        cfg.write_text(json.dumps([{
            "slug": "ai-money", "name": "AI Money", "description": "d",
            "creators": [
                {"handle": "@a", "url": "u", "channel_id": "UCPULLED"},
                {"handle": "@b", "url": "u2", "channel_id": "UCPENDING"},
            ],
        }]))
        import youtube_brain.categories as cats_mod
        monkeypatch.setattr(cats_mod, "_CONFIG_PATH", cfg)

        from sqlalchemy.dialects.sqlite import insert as sqlins

        from youtube_brain.core.models import Brain
        from youtube_brain.storage.brains import insert_brain
        from youtube_brain.storage.database import get_session
        from youtube_brain.storage.database import sources as st

        brain = Brain(name="Creator A")
        await insert_brain(brain)
        async with get_session() as s:
            await s.execute(sqlins(st).values(
                id="s-a", brain_id=str(brain.id), source_type="channel", source_url="u",
                source_id="UCPULLED", status="ready", created_at=brain.created_at))

        lst = (await client.get("/api/categories")).json()
        assert lst[0]["slug"] == "ai-money"
        assert lst[0]["creator_count"] == 2 and lst[0]["pulled_count"] == 1

        detail = (await client.get("/api/categories/ai-money")).json()
        by_handle = {c["handle"]: c for c in detail["creators"]}
        assert by_handle["@a"]["pulled"] is True and by_handle["@a"]["brain_id"] == str(brain.id)
        assert by_handle["@b"]["pulled"] is False

        assert (await client.get("/api/categories/nope")).status_code == 404
    finally:
        shutil.rmtree(d, ignore_errors=True)
