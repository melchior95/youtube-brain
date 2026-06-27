import pytest
from sqlalchemy import text

from youtube_brain.storage.database import init_database, get_session, close_database


@pytest.fixture
async def db(tmp_settings):
    await init_database(tmp_settings)
    yield
    await close_database()


async def test_database_creates_tables(db, tmp_settings):
    assert tmp_settings.database_path.exists()
    async with get_session() as session:
        result = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        )
        tables = [row[0] for row in result.fetchall()]
    assert "brains" in tables
    assert "sources" in tables
    assert "videos" in tables
    assert "chunks" in tables
    assert "chunk_embeddings" in tables
    assert "observations" in tables
    assert "articles" in tables


async def test_fts5_table_exists(db):
    async with get_session() as session:
        result = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        )
        assert result.fetchone() is not None


async def test_fts5_triggers_exist(db):
    async with get_session() as session:
        result = await session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
            )
        )
        triggers = [row[0] for row in result.fetchall()]
    assert "chunks_ai" in triggers
    assert "chunks_ad" in triggers
    assert "chunks_au" in triggers
