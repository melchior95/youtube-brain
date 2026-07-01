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


async def test_migration_adds_stat_columns_to_existing_videos_table(tmp_settings):
    """A videos table created before the stat columns existed should still get
    them on init, since create_all() only creates missing tables."""
    from sqlalchemy import create_engine

    tmp_settings.ensure_dirs()
    db_path = str(tmp_settings.database_path).replace("\\", "/")
    sync_engine = create_engine(f"sqlite:///{db_path}")
    with sync_engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE videos (id TEXT PRIMARY KEY, brain_id TEXT, video_id TEXT, url TEXT)")
        )
        conn.commit()
    sync_engine.dispose()

    await init_database(tmp_settings)

    async with get_session() as session:
        result = await session.execute(text("PRAGMA table_info(videos)"))
        columns = {row[1] for row in result.fetchall()}
    await close_database()

    assert {
        "view_count", "like_count", "comment_count",
        "channel_follower_count", "stats_fetched_at",
    } <= columns
