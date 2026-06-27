"""SQLite database setup with SQLAlchemy async."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from youtube_brain.config.settings import Settings, get_settings

# Metadata for all tables
metadata = MetaData()

# -----------------------------------------------------------------------------
# Table Definitions
# -----------------------------------------------------------------------------

brains = Table(
    "brains",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("name", Text, nullable=False),
    Column("owner_user_id", Text, nullable=True),
    Column("visibility", String(20), default="private"),
    Column("canonical_brain_id", String(36), nullable=True),
    Column("recency_weight", Float, default=0.1),
    Column("video_count", Integer, default=0),
    Column("status", String(20), default="pending"),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

sources = Table(
    "sources",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("source_type", String(20), nullable=False),
    Column("source_url", Text, nullable=False),
    Column("source_title", Text, nullable=True),
    Column("source_id", Text, nullable=False),
    Column("status", String(20), default="pending"),
    Column("created_at", DateTime, nullable=False),
)

videos = Table(
    "videos",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("source_id", String(36), nullable=False, index=True),
    Column("video_id", String(20), nullable=False, index=True),
    Column("title", Text, nullable=True),
    Column("channel_name", Text, nullable=True),
    Column("published_at", DateTime, nullable=True),
    Column("duration_seconds", Integer, nullable=True),
    Column("url", Text, nullable=False),
    Column("transcript_raw", Text, nullable=True),
    Column("transcript_clean", Text, nullable=True),
    Column("transcript_source", String(30), nullable=True),
    Column("transcript_language", String(10), nullable=True),
    Column("caption_kind", String(10), nullable=True),
    Column("transcript_quality_score", Float, nullable=True),
    Column("failure_reason", Text, nullable=True),
    Column("video_summary", Text, nullable=True),
    Column("key_points", JSON, nullable=True),
    Column("businesses_mentioned", JSON, nullable=True),
    Column("people_mentioned", JSON, nullable=True),
    Column("main_topics", JSON, nullable=True),
    Column("status", String(20), default="pending"),
    Column("created_at", DateTime, nullable=False),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("video_id", String(36), nullable=False, index=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("start_time", Float, nullable=False),
    Column("end_time", Float, nullable=False),
    Column("text", Text, nullable=False),
    Column("topics", JSON, nullable=True),
    Column("business_type", JSON, nullable=True),
    Column("advice_category", JSON, nullable=True),
    Column("stage", JSON, nullable=True),
    Column("asset_type", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

chunk_embeddings = Table(
    "chunk_embeddings",
    metadata,
    Column("chunk_id", String(36), primary_key=True),
    Column("model", String(50), nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("embedding", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

observations = Table(
    "observations",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("video_id", String(36), nullable=True, index=True),  # FK -> videos.id
    Column("youtube_id", String(20), nullable=True),
    Column("creator", Text, nullable=True),
    Column("obs_type", String(40), nullable=False, index=True),
    Column("claim", Text, nullable=False),
    Column("value", Text, nullable=True),
    Column("entities", JSON, nullable=True),
    Column("evidence_quote", Text, nullable=True),
    Column("chunk_id", String(36), nullable=True),
    Column("start_time", Float, nullable=True),
    Column("confidence", Float, nullable=True),
    Column("domain", String(30), nullable=False, default="founders"),
    Column("cluster_id", Integer, nullable=True, index=True),
    Column("created_at", DateTime, nullable=False),
)

observation_embeddings = Table(
    "observation_embeddings",
    metadata,
    Column("observation_id", String(36), primary_key=True),
    Column("model", String(50), nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("embedding", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

watchlist_schedules = Table(
    "watchlist_schedules",
    metadata,
    Column("brain_id", String(36), primary_key=True),
    Column("enabled", Boolean, nullable=False, default=True),
    Column("interval_hours", Integer, nullable=False, default=24),
    Column("max_videos", Integer, nullable=True),
    Column("last_refreshed_at", DateTime, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

articles = Table(
    "articles",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("title", Text, nullable=False),
    Column("body", Text, nullable=False),
    Column("article_type", String(20), nullable=False),
    Column("source_chunk_ids", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

# -----------------------------------------------------------------------------
# FTS5 Virtual Table and Trigger SQL
# -----------------------------------------------------------------------------

_FTS5_CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content=chunks, content_rowid=rowid
);
"""

_FTS5_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
    END;
    """,
]

# -----------------------------------------------------------------------------
# Database Engine & Session Management
# -----------------------------------------------------------------------------

_engine = None
_async_session_factory = None


async def init_database(settings: Settings | None = None) -> None:
    """Initialize the database and create tables."""
    global _engine, _async_session_factory

    if settings is None:
        settings = get_settings()

    settings.ensure_dirs()

    db_path = str(settings.database_path).replace("\\", "/")
    async_url = f"sqlite+aiosqlite:///{db_path}"
    sync_url = f"sqlite:///{db_path}"

    _engine = create_async_engine(
        async_url,
        echo=False,
        future=True,
    )

    @event.listens_for(_engine.sync_engine, "connect")
    def set_async_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables using sync engine (SQLite requires sync for DDL)
    sync_engine = create_engine(sync_url)

    @event.listens_for(sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    metadata.create_all(sync_engine)

    # Create FTS5 virtual table and triggers
    with sync_engine.connect() as conn:
        conn.execute(text(_FTS5_CREATE))
        for trigger_sql in _FTS5_TRIGGERS:
            conn.execute(text(trigger_sql))
        conn.commit()

    sync_engine.dispose()


async def close_database() -> None:
    """Close the database connection."""
    global _engine, _async_session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.

    Auto-commits on success, rolls back on exception.
    Lazy-initializes the database if needed.
    """
    global _async_session_factory

    if _async_session_factory is None:
        await init_database()

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
