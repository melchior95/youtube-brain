# YouTube Brain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid-RAG app that turns YouTube channels into searchable advisors with timestamp-cited answers.

**Architecture:** FastAPI backend with SQLite/FTS5/sqlite-vec, Gemini for embeddings + generation, React PWA frontend. Ingestion pipeline: resolve URL → fetch transcripts → chunk → embed → FTS5 index → metadata label → video summary. Retrieval: 4-lane hybrid search (FTS5 chunks + vector chunks + FTS5 summaries + vector summaries) → diversity select → rerank → Gemini answer with citations.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy (async, Core style), aiosqlite, sqlite-vec, Gemini API (google-genai), youtube-transcript-api, yt-dlp, pydantic-settings, React 18, TypeScript, Vite

**Reference project patterns:** a prior internal project — Core SQLAlchemy table definitions, async session management, Pydantic models, Click CLI, pydantic-settings config, FTS5 with triggers, hybrid BM25+vector scoring.

**Design doc:** `docs/plans/2026-06-02-youtube-brain-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/youtube_brain/__init__.py`
- Create: `src/youtube_brain/config/__init__.py`
- Create: `src/youtube_brain/config/settings.py`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create `pyproject.toml`**

```toml
[project]
name = "youtube-brain"
version = "0.1.0"
description = "Turn YouTube channels into searchable advisors"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20.0",
    "sqlite-vec>=0.1.0",
    "google-genai>=1.0.0",
    "youtube-transcript-api>=0.6.0",
    "yt-dlp>=2024.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx[http2]>=0.27.0",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.5.0",
]

[project.scripts]
ytbrain = "youtube_brain.cli:main"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

**Step 2: Create settings module**

```python
# src/youtube_brain/config/settings.py
from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YTBRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    database_path: Path = Field(default=Path("data/youtube_brain.db"))

    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="text-embedding-004")
    gemini_embedding_dimensions: int = Field(default=768)

    chunk_window_seconds: float = Field(default=150.0)
    chunk_overlap_seconds: float = Field(default=30.0)

    max_concurrent_fetches: int = Field(default=5)
    partially_ready_threshold: int = Field(default=5)

    http_timeout: int = Field(default=30)
    http_max_retries: int = Field(default=3)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
```

**Step 3: Create `.env.example`**

```
YTBRAIN_GEMINI_API_KEY=your-gemini-api-key
YTBRAIN_DATABASE_PATH=data/youtube_brain.db
```

**Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
.env
data/
*.db
node_modules/
dist/
.vite/
*.egg-info/
.ruff_cache/
.pytest_cache/
```

**Step 5: Create `tests/conftest.py`**

```python
# tests/conftest.py
import asyncio
import pytest
from pathlib import Path
from youtube_brain.config.settings import Settings


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "test.db",
        gemini_api_key="test-key",
    )
```

**Step 6: Create empty `__init__.py` files**

Create `src/youtube_brain/__init__.py` and `src/youtube_brain/config/__init__.py` and `tests/__init__.py` as empty files.

**Step 7: Install project in dev mode**

Run: `pip install -e ".[dev]"`
Expected: Clean install, no errors.

**Step 8: Verify settings load**

Run: `python -c "from youtube_brain.config.settings import get_settings; s = get_settings(); print(s.database_path)"`
Expected: `data\youtube_brain.db`

**Step 9: Commit**

```bash
git init
git add pyproject.toml src/ tests/ .env.example .gitignore
git commit -m "feat: project scaffolding with settings and dev tooling"
```

---

## Task 2: Database Schema

**Files:**
- Create: `src/youtube_brain/storage/__init__.py`
- Create: `src/youtube_brain/storage/database.py`
- Create: `tests/test_database.py`

**Step 1: Write the failing test**

```python
# tests/test_database.py
import pytest
from youtube_brain.storage.database import init_database, get_session


@pytest.fixture
async def db(tmp_settings):
    await init_database(tmp_settings)
    yield


async def test_database_creates_tables(db, tmp_settings):
    assert tmp_settings.database_path.exists()
    async with get_session() as session:
        result = await session.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
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
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert result.fetchone() is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write database module**

```python
# src/youtube_brain/storage/database.py
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from sqlalchemy import (
    JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
    create_engine, event, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from youtube_brain.config.settings import Settings, get_settings

metadata = MetaData()

brains = Table(
    "brains", metadata,
    Column("id", String(36), primary_key=True),
    Column("name", Text, nullable=False),
    Column("owner_user_id", Text, nullable=True),
    Column("visibility", String(20), nullable=False, default="private"),
    Column("canonical_brain_id", String(36), nullable=True),
    Column("recency_weight", Float, nullable=False, default=0.1),
    Column("video_count", Integer, nullable=False, default=0),
    Column("status", String(20), nullable=False, default="pending"),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

sources = Table(
    "sources", metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("source_type", String(20), nullable=False),
    Column("source_url", Text, nullable=False),
    Column("source_title", Text, nullable=True),
    Column("source_id", Text, nullable=False),
    Column("status", String(20), nullable=False, default="pending"),
    Column("created_at", DateTime, nullable=False),
)

videos = Table(
    "videos", metadata,
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
    Column("status", String(20), nullable=False, default="pending"),
    Column("created_at", DateTime, nullable=False),
)

chunks = Table(
    "chunks", metadata,
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
    "chunk_embeddings", metadata,
    Column("chunk_id", String(36), primary_key=True),
    Column("model", String(50), nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("embedding", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

observations = Table(
    "observations", metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("observation", Text, nullable=False),
    Column("source_chunk_ids", JSON, nullable=True),
    Column("topic", Text, nullable=True),
    Column("confidence", Float, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

articles = Table(
    "articles", metadata,
    Column("id", String(36), primary_key=True),
    Column("brain_id", String(36), nullable=False, index=True),
    Column("title", Text, nullable=False),
    Column("body", Text, nullable=False),
    Column("article_type", String(20), nullable=False),
    Column("source_chunk_ids", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

_FTS5_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content=chunks, content_rowid=rowid
);
"""

_FTS5_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

_engine = None
_async_session_factory = None


async def init_database(settings: Settings | None = None) -> None:
    global _engine, _async_session_factory

    if settings is None:
        settings = get_settings()
    settings.ensure_dirs()

    db_url = f"sqlite+aiosqlite:///{settings.database_path}"
    _engine = create_async_engine(db_url, echo=False, future=True)
    _async_session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False,
    )

    sync_url = f"sqlite:///{settings.database_path}"
    sync_engine = create_engine(sync_url)

    @event.listens_for(sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    metadata.create_all(sync_engine)

    with sync_engine.connect() as conn:
        for stmt in _FTS5_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        for stmt in _FTS5_TRIGGERS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()

    sync_engine.dispose()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _async_session_factory is None:
        await init_database()

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

Note: FTS5 uses `content=chunks` (external content table). The triggers keep FTS5 in sync automatically when rows are inserted/updated/deleted in `chunks`. The `rowid` used for `content_rowid` is SQLite's implicit integer rowid, which exists on every table unless you use `WITHOUT ROWID`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_database.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/storage/ tests/test_database.py
git commit -m "feat: database schema with FTS5 and all MVP tables"
```

---

## Task 3: Pydantic Models and Enums

**Files:**
- Create: `src/youtube_brain/core/__init__.py`
- Create: `src/youtube_brain/core/enums.py`
- Create: `src/youtube_brain/core/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
from youtube_brain.core.enums import BrainStatus, SourceType, CaptionKind, BusinessType
from youtube_brain.core.models import Brain, Source, Video, Chunk


def test_brain_defaults():
    brain = Brain(name="Test Brain")
    assert brain.id is not None
    assert brain.status == BrainStatus.PENDING
    assert brain.visibility == "private"
    assert brain.recency_weight == 0.1


def test_source_types():
    assert SourceType.CHANNEL.value == "channel"
    assert SourceType.PLAYLIST.value == "playlist"
    assert SourceType.VIDEO.value == "video"


def test_controlled_taxonomy():
    assert BusinessType.SAAS.value == "saas"
    assert BusinessType.ECOMMERCE.value == "ecommerce"


def test_chunk_with_metadata():
    chunk = Chunk(
        video_id="vid-1",
        brain_id="brain-1",
        start_time=0.0,
        end_time=150.0,
        text="Some transcript text",
        topics=["marketing"],
        business_type=["saas"],
    )
    assert chunk.start_time == 0.0
    assert chunk.topics == ["marketing"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write enums module**

```python
# src/youtube_brain/core/enums.py
from enum import Enum


class BrainStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    ERROR = "error"


class SourceType(str, Enum):
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    VIDEO = "video"


class SourceStatus(str, Enum):
    PENDING = "pending"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ERROR = "error"


class VideoStatus(str, Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    CHUNKED = "chunked"
    SUMMARIZED = "summarized"
    ERROR = "error"


class CaptionKind(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


class TranscriptSource(str, Enum):
    MANUAL = "manual"
    OFFICIAL_CAPTION = "official_caption"
    AUTO_CAPTION = "auto_caption"
    YT_DLP = "yt_dlp"
    API = "api"


class BusinessType(str, Enum):
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    AGENCY = "agency"
    MARKETPLACE = "marketplace"
    CONTENT = "content"
    PHYSICAL_PRODUCT = "physical_product"
    SERVICE = "service"
    MOBILE_APP = "mobile_app"
    OTHER = "other"


class AdviceCategory(str, Enum):
    MARKETING = "marketing"
    DISTRIBUTION = "distribution"
    PRICING = "pricing"
    HIRING = "hiring"
    FUNDRAISING = "fundraising"
    PRODUCT = "product"
    OPERATIONS = "operations"
    CUSTOMER_ACQUISITION = "customer_acquisition"
    RETENTION = "retention"
    MONETIZATION = "monetization"
    LAUNCH = "launch"
    GROWTH = "growth"
    TECHNICAL = "technical"
    LEGAL = "legal"
    OTHER = "other"


class Stage(str, Enum):
    IDEA = "idea"
    PRE_LAUNCH = "pre_launch"
    EARLY_STAGE = "early_stage"
    GROWTH = "growth"
    SCALING = "scaling"
    MATURE = "mature"
    EXIT = "exit"
    OTHER = "other"


class AssetType(str, Enum):
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    COMMENTARY = "commentary"
    CASE_STUDY = "case_study"
    EARNINGS_CALL = "earnings_call"
    LECTURE = "lecture"
    PANEL = "panel"
    OTHER = "other"


class ArticleType(str, Enum):
    SUMMARY = "summary"
    PLAYBOOK = "playbook"
    FAQ = "faq"
    COMPARISON = "comparison"
```

**Step 4: Write models module**

```python
# src/youtube_brain/core/models.py
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from youtube_brain.core.enums import (
    BrainStatus, SourceType, SourceStatus, VideoStatus,
    CaptionKind, TranscriptSource, ArticleType,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Brain(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    owner_user_id: str | None = None
    visibility: str = "private"
    canonical_brain_id: UUID | None = None
    recency_weight: float = 0.1
    video_count: int = 0
    status: BrainStatus = BrainStatus.PENDING
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Source(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    brain_id: UUID
    source_type: SourceType
    source_url: str
    source_title: str | None = None
    source_id: str
    status: SourceStatus = SourceStatus.PENDING
    created_at: datetime = Field(default_factory=_now)


class Video(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    brain_id: UUID
    source_id: UUID
    video_id: str
    title: str | None = None
    channel_name: str | None = None
    published_at: datetime | None = None
    duration_seconds: int | None = None
    url: str
    transcript_raw: str | None = None
    transcript_clean: str | None = None
    transcript_source: TranscriptSource | None = None
    transcript_language: str | None = None
    caption_kind: CaptionKind | None = None
    transcript_quality_score: float | None = None
    failure_reason: str | None = None
    video_summary: str | None = None
    key_points: list[str] | None = None
    businesses_mentioned: list[str] | None = None
    people_mentioned: list[str] | None = None
    main_topics: list[str] | None = None
    status: VideoStatus = VideoStatus.PENDING
    created_at: datetime = Field(default_factory=_now)


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    video_id: UUID | str
    brain_id: UUID | str
    start_time: float
    end_time: float
    text: str
    topics: list[str] | None = None
    business_type: list[str] | None = None
    advice_category: list[str] | None = None
    stage: list[str] | None = None
    asset_type: list[str] | None = None
    created_at: datetime = Field(default_factory=_now)


class ChunkEmbedding(BaseModel):
    chunk_id: UUID | str
    model: str
    dimensions: int
    embedding: list[float]
    created_at: datetime = Field(default_factory=_now)


class Observation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    brain_id: UUID | str
    observation: str
    source_chunk_ids: list[str] | None = None
    topic: str | None = None
    confidence: float | None = None
    created_at: datetime = Field(default_factory=_now)


class Article(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    brain_id: UUID | str
    title: str
    body: str
    article_type: ArticleType
    source_chunk_ids: list[str] | None = None
    created_at: datetime = Field(default_factory=_now)
```

**Step 5: Run tests**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/youtube_brain/core/ tests/test_models.py
git commit -m "feat: pydantic models and controlled taxonomy enums"
```

---

## Task 4: Storage CRUD Layer

**Files:**
- Create: `src/youtube_brain/storage/brains.py`
- Create: `src/youtube_brain/storage/videos.py`
- Create: `src/youtube_brain/storage/chunks.py`
- Create: `tests/test_storage.py`

**Step 1: Write the failing test**

```python
# tests/test_storage.py
import pytest
from uuid import uuid4
from youtube_brain.storage.database import init_database, get_session
from youtube_brain.storage.brains import insert_brain, get_brain, update_brain_status
from youtube_brain.storage.videos import insert_video, get_videos_by_brain, update_video
from youtube_brain.storage.chunks import insert_chunks, get_chunks_by_video, search_fts
from youtube_brain.core.models import Brain, Video, Chunk, Source
from youtube_brain.core.enums import BrainStatus, SourceType, VideoStatus


@pytest.fixture
async def db(tmp_settings):
    await init_database(tmp_settings)
    yield


async def test_insert_and_get_brain(db):
    brain = Brain(name="Test Brain")
    await insert_brain(brain)
    fetched = await get_brain(brain.id)
    assert fetched is not None
    assert fetched.name == "Test Brain"
    assert fetched.status == BrainStatus.PENDING


async def test_update_brain_status(db):
    brain = Brain(name="Status Brain")
    await insert_brain(brain)
    await update_brain_status(brain.id, BrainStatus.INGESTING)
    fetched = await get_brain(brain.id)
    assert fetched.status == BrainStatus.INGESTING


async def test_insert_and_search_chunks(db):
    brain = Brain(name="Chunk Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id, source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=abc", source_id="abc",
    )
    video = Video(
        brain_id=brain.id, source_id=source.id,
        video_id="abc", url="https://youtube.com/watch?v=abc",
    )
    await insert_video(video)
    chunk = Chunk(
        video_id=video.id, brain_id=brain.id,
        start_time=0.0, end_time=150.0,
        text="Reddit was our best marketing channel for SaaS growth",
    )
    await insert_chunks([chunk])
    results = await search_fts("Reddit marketing", str(brain.id), limit=10)
    assert len(results) >= 1
    assert "Reddit" in results[0].text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write `storage/brains.py`**

```python
# src/youtube_brain/storage/brains.py
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.enums import BrainStatus
from youtube_brain.core.models import Brain
from youtube_brain.storage.database import brains, get_session


def _uuid(val: UUID | str) -> str:
    return str(val)


def _row_to_brain(row) -> Brain:
    return Brain(
        id=UUID(row.id),
        name=row.name,
        owner_user_id=row.owner_user_id,
        visibility=row.visibility,
        canonical_brain_id=UUID(row.canonical_brain_id) if row.canonical_brain_id else None,
        recency_weight=row.recency_weight,
        video_count=row.video_count,
        status=BrainStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def insert_brain(brain: Brain) -> bool:
    async with get_session() as session:
        stmt = insert(brains).values(
            id=_uuid(brain.id),
            name=brain.name,
            owner_user_id=brain.owner_user_id,
            visibility=brain.visibility,
            canonical_brain_id=_uuid(brain.canonical_brain_id) if brain.canonical_brain_id else None,
            recency_weight=brain.recency_weight,
            video_count=brain.video_count,
            status=brain.status.value,
            created_at=brain.created_at,
            updated_at=brain.updated_at,
        ).on_conflict_do_nothing(index_elements=["id"])
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_brain(brain_id: UUID | str) -> Brain | None:
    async with get_session() as session:
        stmt = select(brains).where(brains.c.id == _uuid(brain_id))
        result = await session.execute(stmt)
        row = result.fetchone()
        return _row_to_brain(row) if row else None


async def list_brains() -> list[Brain]:
    async with get_session() as session:
        stmt = select(brains).order_by(brains.c.updated_at.desc())
        result = await session.execute(stmt)
        return [_row_to_brain(row) for row in result.fetchall()]


async def update_brain_status(brain_id: UUID | str, status: BrainStatus) -> None:
    async with get_session() as session:
        stmt = (
            update(brains)
            .where(brains.c.id == _uuid(brain_id))
            .values(status=status.value, updated_at=datetime.now(timezone.utc))
        )
        await session.execute(stmt)


async def increment_video_count(brain_id: UUID | str, count: int = 1) -> None:
    async with get_session() as session:
        stmt = (
            update(brains)
            .where(brains.c.id == _uuid(brain_id))
            .values(
                video_count=brains.c.video_count + count,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.execute(stmt)
```

**Step 4: Write `storage/videos.py`**

```python
# src/youtube_brain/storage/videos.py
from uuid import UUID
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.enums import VideoStatus, CaptionKind, TranscriptSource
from youtube_brain.core.models import Video
from youtube_brain.storage.database import videos, get_session


def _uuid(val: UUID | str) -> str:
    return str(val)


def _row_to_video(row) -> Video:
    return Video(
        id=UUID(row.id),
        brain_id=UUID(row.brain_id),
        source_id=UUID(row.source_id),
        video_id=row.video_id,
        title=row.title,
        channel_name=row.channel_name,
        published_at=row.published_at,
        duration_seconds=row.duration_seconds,
        url=row.url,
        transcript_raw=row.transcript_raw,
        transcript_clean=row.transcript_clean,
        transcript_source=TranscriptSource(row.transcript_source) if row.transcript_source else None,
        transcript_language=row.transcript_language,
        caption_kind=CaptionKind(row.caption_kind) if row.caption_kind else None,
        transcript_quality_score=row.transcript_quality_score,
        failure_reason=row.failure_reason,
        video_summary=row.video_summary,
        key_points=row.key_points,
        businesses_mentioned=row.businesses_mentioned,
        people_mentioned=row.people_mentioned,
        main_topics=row.main_topics,
        status=VideoStatus(row.status),
        created_at=row.created_at,
    )


async def insert_video(video: Video) -> bool:
    async with get_session() as session:
        stmt = insert(videos).values(
            id=_uuid(video.id),
            brain_id=_uuid(video.brain_id),
            source_id=_uuid(video.source_id),
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            published_at=video.published_at,
            duration_seconds=video.duration_seconds,
            url=video.url,
            transcript_raw=video.transcript_raw,
            transcript_clean=video.transcript_clean,
            transcript_source=video.transcript_source.value if video.transcript_source else None,
            transcript_language=video.transcript_language,
            caption_kind=video.caption_kind.value if video.caption_kind else None,
            transcript_quality_score=video.transcript_quality_score,
            failure_reason=video.failure_reason,
            video_summary=video.video_summary,
            key_points=video.key_points,
            businesses_mentioned=video.businesses_mentioned,
            people_mentioned=video.people_mentioned,
            main_topics=video.main_topics,
            status=video.status.value,
            created_at=video.created_at,
        ).on_conflict_do_nothing(index_elements=["id"])
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_videos_by_brain(
    brain_id: UUID | str, status: VideoStatus | None = None, limit: int = 100,
) -> list[Video]:
    async with get_session() as session:
        stmt = select(videos).where(videos.c.brain_id == _uuid(brain_id))
        if status:
            stmt = stmt.where(videos.c.status == status.value)
        stmt = stmt.order_by(videos.c.created_at).limit(limit)
        result = await session.execute(stmt)
        return [_row_to_video(row) for row in result.fetchall()]


async def update_video(video_id: UUID | str, **kwargs) -> None:
    async with get_session() as session:
        values = {}
        for key, val in kwargs.items():
            if hasattr(val, "value"):
                values[key] = val.value
            else:
                values[key] = val
        stmt = update(videos).where(videos.c.id == _uuid(video_id)).values(**values)
        await session.execute(stmt)


async def video_exists(brain_id: UUID | str, yt_video_id: str) -> bool:
    async with get_session() as session:
        stmt = (
            select(videos.c.id)
            .where(videos.c.brain_id == _uuid(brain_id))
            .where(videos.c.video_id == yt_video_id)
        )
        result = await session.execute(stmt)
        return result.fetchone() is not None
```

**Step 5: Write `storage/chunks.py`**

```python
# src/youtube_brain/storage/chunks.py
from uuid import UUID
from sqlalchemy import select, text
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.models import Chunk
from youtube_brain.storage.database import chunks, chunk_embeddings, get_session


def _uuid(val: UUID | str) -> str:
    return str(val)


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=UUID(row.id),
        video_id=UUID(row.video_id),
        brain_id=UUID(row.brain_id),
        start_time=row.start_time,
        end_time=row.end_time,
        text=row.text,
        topics=row.topics,
        business_type=row.business_type,
        advice_category=row.advice_category,
        stage=row.stage,
        asset_type=row.asset_type,
        created_at=row.created_at,
    )


async def insert_chunks(chunk_list: list[Chunk]) -> int:
    if not chunk_list:
        return 0
    async with get_session() as session:
        count = 0
        for chunk in chunk_list:
            stmt = insert(chunks).values(
                id=_uuid(chunk.id),
                video_id=_uuid(chunk.video_id),
                brain_id=_uuid(chunk.brain_id),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=chunk.text,
                topics=chunk.topics,
                business_type=chunk.business_type,
                advice_category=chunk.advice_category,
                stage=chunk.stage,
                asset_type=chunk.asset_type,
                created_at=chunk.created_at,
            ).on_conflict_do_nothing(index_elements=["id"])
            result = await session.execute(stmt)
            count += result.rowcount
        return count


async def get_chunks_by_video(video_id: UUID | str) -> list[Chunk]:
    async with get_session() as session:
        stmt = (
            select(chunks)
            .where(chunks.c.video_id == _uuid(video_id))
            .order_by(chunks.c.start_time)
        )
        result = await session.execute(stmt)
        return [_row_to_chunk(row) for row in result.fetchall()]


async def search_fts(query: str, brain_id: str, limit: int = 50) -> list[Chunk]:
    terms = query.split()
    match_expr = " OR ".join(f'"{t}"' for t in terms if t.strip())
    if not match_expr:
        return []

    async with get_session() as session:
        sql = text("""
            SELECT c.* FROM chunks c
            JOIN chunks_fts ON chunks_fts.rowid = c.rowid
            WHERE chunks_fts MATCH :query
            AND c.brain_id = :brain_id
            ORDER BY bm25(chunks_fts)
            LIMIT :limit
        """)
        result = await session.execute(
            sql, {"query": match_expr, "brain_id": brain_id, "limit": limit}
        )
        return [_row_to_chunk(row) for row in result.fetchall()]


async def store_embedding(chunk_id: UUID | str, model: str, dims: int, embedding: list[float]) -> None:
    import json
    async with get_session() as session:
        stmt = insert(chunk_embeddings).values(
            chunk_id=_uuid(chunk_id),
            model=model,
            dimensions=dims,
            embedding=json.dumps(embedding),
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        ).on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={"model": model, "dimensions": dims, "embedding": json.dumps(embedding)},
        )
        await session.execute(stmt)
```

**Step 6: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/youtube_brain/storage/brains.py src/youtube_brain/storage/videos.py src/youtube_brain/storage/chunks.py tests/test_storage.py
git commit -m "feat: CRUD storage layer for brains, videos, and chunks"
```

---

## Task 5: Transcript Fetcher

**Files:**
- Create: `src/youtube_brain/ingest/__init__.py`
- Create: `src/youtube_brain/ingest/transcripts.py`
- Create: `tests/test_transcripts.py`

**Step 1: Write the test (uses real YouTube API — integration test)**

```python
# tests/test_transcripts.py
import pytest
from youtube_brain.ingest.transcripts import fetch_transcript, TranscriptResult


def test_transcript_result_model():
    result = TranscriptResult(
        text_with_timestamps=[
            {"start": 0.0, "duration": 5.0, "text": "Hello world"}
        ],
        full_text="Hello world",
        language="en",
        is_auto_generated=False,
        source="api",
    )
    assert result.full_text == "Hello world"
    assert result.source == "api"


@pytest.mark.integration
async def test_fetch_real_transcript():
    """Integration test — requires internet. Run with: pytest -m integration"""
    result = await fetch_transcript("dQw4w9WgXcQ")
    assert result is not None
    assert len(result.full_text) > 100
    assert result.language is not None
```

**Step 2: Run test**

Run: `pytest tests/test_transcripts.py -v -k "not integration"`
Expected: PASS (unit test only)

**Step 3: Write transcript fetcher**

```python
# src/youtube_brain/ingest/transcripts.py
import re
import logging
from dataclasses import dataclass

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    start: float
    duration: float
    text: str


class TranscriptResult(BaseModel):
    text_with_timestamps: list[dict]
    full_text: str
    language: str | None = None
    is_auto_generated: bool = False
    source: str = "api"


async def fetch_transcript(video_id: str) -> TranscriptResult | None:
    result = _try_youtube_transcript_api(video_id)
    if result:
        return result

    result = await _try_yt_dlp(video_id)
    if result:
        return result

    logger.warning("All transcript methods failed for %s", video_id)
    return None


def _try_youtube_transcript_api(video_id: str) -> TranscriptResult | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
            is_auto = False
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
                is_auto = True
            except Exception:
                transcripts = list(transcript_list)
                if not transcripts:
                    return None
                transcript = transcripts[0]
                is_auto = transcript.is_generated

        segments = transcript.fetch()
        entries = [
            {"start": s.start, "duration": s.duration, "text": s.text}
            for s in segments
        ]
        full_text = " ".join(s.text for s in segments)

        return TranscriptResult(
            text_with_timestamps=entries,
            full_text=full_text,
            language=transcript.language_code,
            is_auto_generated=is_auto,
            source="api",
        )

    except Exception as e:
        logger.debug("youtube-transcript-api failed for %s: %s", video_id, e)
        return None


async def _try_yt_dlp(video_id: str) -> TranscriptResult | None:
    import asyncio
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--write-auto-subs", "--skip-download",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--output", "-",
            f"https://youtube.com/watch?v={video_id}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.debug("yt-dlp failed for %s: %s", video_id, stderr.decode())
            return None

        import json
        data = json.loads(stdout.decode())
        events = data.get("events", [])

        entries = []
        for ev in events:
            if "segs" in ev:
                text = "".join(seg.get("utf8", "") for seg in ev["segs"]).strip()
                if text:
                    entries.append({
                        "start": ev.get("tStartMs", 0) / 1000.0,
                        "duration": ev.get("dDurationMs", 0) / 1000.0,
                        "text": text,
                    })

        full_text = " ".join(e["text"] for e in entries)
        return TranscriptResult(
            text_with_timestamps=entries,
            full_text=full_text,
            language="en",
            is_auto_generated=True,
            source="yt_dlp",
        )

    except Exception as e:
        logger.debug("yt-dlp failed for %s: %s", video_id, e)
        return None


def clean_transcript(raw_text: str) -> str:
    text = re.sub(r"\[Music\]|\[Applause\]|\[Laughter\]", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Step 4: Run tests**

Run: `pytest tests/test_transcripts.py -v -k "not integration"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/ingest/ tests/test_transcripts.py
git commit -m "feat: transcript fetcher with youtube-transcript-api and yt-dlp fallback"
```

---

## Task 6: URL Resolver (Source Discovery)

**Files:**
- Create: `src/youtube_brain/ingest/resolver.py`
- Create: `tests/test_resolver.py`

**Step 1: Write the failing test**

```python
# tests/test_resolver.py
from youtube_brain.ingest.resolver import parse_youtube_url, UrlParseResult


def test_parse_video_url():
    result = parse_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result.source_type == "video"
    assert result.video_id == "dQw4w9WgXcQ"


def test_parse_short_url():
    result = parse_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert result.source_type == "video"
    assert result.video_id == "dQw4w9WgXcQ"


def test_parse_playlist_url():
    result = parse_youtube_url(
        "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
    )
    assert result.source_type == "playlist"
    assert result.playlist_id == "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"


def test_parse_channel_url():
    result = parse_youtube_url("https://www.youtube.com/@starterstory")
    assert result.source_type == "channel"
    assert result.channel_handle == "starterstory"


def test_parse_channel_id_url():
    result = parse_youtube_url("https://www.youtube.com/channel/UC123abc")
    assert result.source_type == "channel"
    assert result.channel_id == "UC123abc"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolver.py -v`
Expected: FAIL

**Step 3: Write URL resolver**

```python
# src/youtube_brain/ingest/resolver.py
import re
import logging
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UrlParseResult(BaseModel):
    source_type: str  # video / playlist / channel
    video_id: str | None = None
    playlist_id: str | None = None
    channel_id: str | None = None
    channel_handle: str | None = None
    original_url: str = ""


def parse_youtube_url(url: str) -> UrlParseResult:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    # youtu.be short link
    if parsed.hostname and "youtu.be" in parsed.hostname:
        video_id = parsed.path.lstrip("/").split("/")[0]
        return UrlParseResult(source_type="video", video_id=video_id, original_url=url)

    path = parsed.path.rstrip("/")

    # /playlist?list=...
    if "list" in qs and "/playlist" in path:
        return UrlParseResult(
            source_type="playlist", playlist_id=qs["list"][0], original_url=url,
        )

    # /watch?v=...
    if "v" in qs:
        return UrlParseResult(source_type="video", video_id=qs["v"][0], original_url=url)

    # /@handle
    match = re.match(r"/@([\w\-\.]+)", path)
    if match:
        return UrlParseResult(
            source_type="channel", channel_handle=match.group(1), original_url=url,
        )

    # /channel/UC...
    match = re.match(r"/channel/(UC[\w\-]+)", path)
    if match:
        return UrlParseResult(
            source_type="channel", channel_id=match.group(1), original_url=url,
        )

    # /c/CustomName
    match = re.match(r"/c/([\w\-]+)", path)
    if match:
        return UrlParseResult(
            source_type="channel", channel_handle=match.group(1), original_url=url,
        )

    raise ValueError(f"Cannot parse YouTube URL: {url}")


async def resolve_video_ids(parse_result: UrlParseResult) -> list[dict]:
    if parse_result.source_type == "video":
        return [{"video_id": parse_result.video_id, "title": None}]

    if parse_result.source_type == "playlist":
        return await _resolve_playlist(parse_result.playlist_id)

    if parse_result.source_type == "channel":
        return await _resolve_channel(parse_result)

    return []


async def _resolve_playlist(playlist_id: str) -> list[dict]:
    import asyncio
    import json

    proc = await asyncio.create_subprocess_exec(
        "yt-dlp", "--flat-playlist", "--dump-json",
        f"https://www.youtube.com/playlist?list={playlist_id}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    results = []
    for line in stdout.decode().strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        results.append({
            "video_id": entry.get("id"),
            "title": entry.get("title"),
            "channel_name": entry.get("channel") or entry.get("uploader"),
            "duration_seconds": entry.get("duration"),
        })
    return results


async def _resolve_channel(parse_result: UrlParseResult) -> list[dict]:
    import asyncio
    import json

    if parse_result.channel_handle:
        url = f"https://www.youtube.com/@{parse_result.channel_handle}/videos"
    else:
        url = f"https://www.youtube.com/channel/{parse_result.channel_id}/videos"

    proc = await asyncio.create_subprocess_exec(
        "yt-dlp", "--flat-playlist", "--dump-json", url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    results = []
    for line in stdout.decode().strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        results.append({
            "video_id": entry.get("id"),
            "title": entry.get("title"),
            "channel_name": entry.get("channel") or entry.get("uploader"),
            "duration_seconds": entry.get("duration"),
        })
    return results
```

**Step 4: Run tests**

Run: `pytest tests/test_resolver.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/ingest/resolver.py tests/test_resolver.py
git commit -m "feat: YouTube URL parser and video ID resolver via yt-dlp"
```

---

## Task 7: Chunking Engine

**Files:**
- Create: `src/youtube_brain/ingest/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing test**

```python
# tests/test_chunker.py
from youtube_brain.ingest.chunker import chunk_transcript


def test_basic_chunking():
    segments = [
        {"start": i * 10.0, "duration": 10.0, "text": f"Sentence number {i}."}
        for i in range(60)
    ]
    chunks = chunk_transcript(segments, window=150.0, overlap=30.0)
    assert len(chunks) > 1
    assert chunks[0]["start_time"] == 0.0
    assert chunks[0]["end_time"] <= 155.0


def test_overlap_exists():
    segments = [
        {"start": i * 10.0, "duration": 10.0, "text": f"Word {i}."}
        for i in range(60)
    ]
    chunks = chunk_transcript(segments, window=150.0, overlap=30.0)
    if len(chunks) >= 2:
        assert chunks[1]["start_time"] < chunks[0]["end_time"]


def test_short_video_single_chunk():
    segments = [
        {"start": 0.0, "duration": 5.0, "text": "Short video."}
    ]
    chunks = chunk_transcript(segments, window=150.0, overlap=30.0)
    assert len(chunks) == 1


def test_sentence_boundary_snap():
    segments = [
        {"start": 0.0, "duration": 50.0, "text": "First part of a sentence"},
        {"start": 50.0, "duration": 50.0, "text": "that continues here."},
        {"start": 100.0, "duration": 50.0, "text": "New sentence starts."},
        {"start": 150.0, "duration": 50.0, "text": "Another one here."},
    ]
    chunks = chunk_transcript(segments, window=150.0, overlap=30.0)
    assert chunks[0]["text"].endswith(".")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL

**Step 3: Write chunker**

```python
# src/youtube_brain/ingest/chunker.py
import re


def chunk_transcript(
    segments: list[dict],
    window: float = 150.0,
    overlap: float = 30.0,
) -> list[dict]:
    if not segments:
        return []

    total_duration = segments[-1]["start"] + segments[-1].get("duration", 0)
    step = window - overlap
    chunks = []
    start = 0.0

    while start < total_duration:
        end = start + window

        chunk_segments = [
            s for s in segments
            if s["start"] + s.get("duration", 0) > start and s["start"] < end
        ]

        if not chunk_segments:
            start += step
            continue

        text = " ".join(s["text"] for s in chunk_segments).strip()
        text = _snap_to_sentence_end(text)

        actual_start = chunk_segments[0]["start"]
        last_seg = chunk_segments[-1]
        actual_end = last_seg["start"] + last_seg.get("duration", 0)

        if text:
            chunks.append({
                "start_time": actual_start,
                "end_time": actual_end,
                "text": text,
            })

        start += step

    return chunks


def _snap_to_sentence_end(text: str) -> str:
    sentence_ends = list(re.finditer(r'[.!?](?:\s|$)', text))
    if sentence_ends:
        last_end = sentence_ends[-1]
        return text[:last_end.end()].strip()
    return text
```

**Step 4: Run tests**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/ingest/chunker.py tests/test_chunker.py
git commit -m "feat: timestamp-aware chunker with overlap and sentence boundary snapping"
```

---

## Task 8: Gemini Client

**Files:**
- Create: `src/youtube_brain/llm/__init__.py`
- Create: `src/youtube_brain/llm/gemini.py`
- Create: `tests/test_gemini.py`

**Step 1: Write the test**

```python
# tests/test_gemini.py
import pytest
from youtube_brain.llm.gemini import GeminiClient


def test_client_init():
    client = GeminiClient(api_key="test-key")
    assert client.model == "gemini-2.5-flash"


@pytest.mark.integration
async def test_embed_real():
    """Run with: pytest -m integration"""
    import os
    key = os.environ.get("YTBRAIN_GEMINI_API_KEY")
    if not key:
        pytest.skip("No Gemini API key")
    client = GeminiClient(api_key=key)
    vectors = await client.embed_texts(["Hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
```

**Step 2: Write Gemini client**

```python
# src/youtube_brain/llm/gemini.py
import json
import logging
from typing import Any

import httpx

from youtube_brain.config.settings import get_settings

logger = logging.getLogger(__name__)

_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
_BATCH_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.embed_model = settings.gemini_embedding_model
        self.embed_dims = settings.gemini_embedding_dimensions
        self._http = httpx.AsyncClient(timeout=60)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        response_json: bool = False,
    ) -> str:
        url = _GENERATE_URL.format(model=self.model)
        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if response_json:
            body["generationConfig"]["responseMimeType"] = "application/json"

        resp = await self._http.post(
            url, params={"key": self.api_key}, json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.3,
    ) -> dict | list:
        text = await self.generate(
            prompt, system=system, temperature=temperature, response_json=True,
        )
        return json.loads(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 1:
            return [await self._embed_single(texts[0])]
        return await self._embed_batch(texts)

    async def _embed_single(self, text: str) -> list[float]:
        url = _EMBED_URL.format(model=self.embed_model)
        body = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.embed_dims,
        }
        resp = await self._http.post(url, params={"key": self.api_key}, json=body)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = _BATCH_EMBED_URL.format(model=self.embed_model)
        requests = [
            {
                "model": f"models/{self.embed_model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.embed_dims,
            }
            for t in texts
        ]

        all_embeddings = []
        batch_size = 100
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            resp = await self._http.post(
                url, params={"key": self.api_key},
                json={"requests": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(
                emb["values"] for emb in data["embeddings"]
            )

        return all_embeddings

    async def close(self):
        await self._http.aclose()
```

**Step 3: Run tests**

Run: `pytest tests/test_gemini.py -v -k "not integration"`
Expected: PASS

**Step 4: Commit**

```bash
git add src/youtube_brain/llm/ tests/test_gemini.py
git commit -m "feat: Gemini client for generation and batch embeddings"
```

---

## Task 9: Ingestion Pipeline Orchestrator

**Files:**
- Create: `src/youtube_brain/ingest/pipeline.py`
- Create: `src/youtube_brain/ingest/labeler.py`
- Create: `src/youtube_brain/ingest/summarizer.py`
- Create: `tests/test_pipeline.py`

This task wires together Tasks 4-8 into the full pipeline: resolve → fetch → chunk → embed → FTS5 → label → summarize.

**Step 1: Write metadata labeler**

```python
# src/youtube_brain/ingest/labeler.py
import logging
from youtube_brain.llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

LABEL_SYSTEM = """You are a metadata labeler for YouTube transcript chunks.
Extract structured metadata using ONLY these controlled values.

business_type: saas, ecommerce, agency, marketplace, content, physical_product, service, mobile_app, other
advice_category: marketing, distribution, pricing, hiring, fundraising, product, operations, customer_acquisition, retention, monetization, launch, growth, technical, legal, other
stage: idea, pre_launch, early_stage, growth, scaling, mature, exit, other
asset_type: interview, tutorial, review, commentary, case_study, earnings_call, lecture, panel, other

topics: free-form list of specific topics (2-5 keywords)

Return JSON: {"topics": [...], "business_type": [...], "advice_category": [...], "stage": [...], "asset_type": [...]}
Use empty arrays if a category doesn't apply. Only use values from the controlled lists above."""


async def label_chunks(
    client: GeminiClient,
    chunks: list[dict],
    video_title: str,
    channel_name: str,
) -> list[dict]:
    results = []
    batch_size = 5

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_text = "\n\n---\n\n".join(
            f"[Chunk {j+1} | {c.get('start_time', 0):.0f}s-{c.get('end_time', 0):.0f}s]\n{c['text']}"
            for j, c in enumerate(batch)
        )
        prompt = f"Video: {video_title}\nChannel: {channel_name}\n\nChunks:\n{batch_text}\n\nLabel each chunk. Return a JSON array with one object per chunk."

        try:
            labels = await client.generate_json(prompt, system=LABEL_SYSTEM)
            if isinstance(labels, dict):
                labels = [labels]
            results.extend(labels)
        except Exception as e:
            logger.warning("Labeling failed for batch %d: %s", i, e)
            results.extend([{}] * len(batch))

    return results
```

**Step 2: Write video summarizer**

```python
# src/youtube_brain/ingest/summarizer.py
import logging
from youtube_brain.llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = """Summarize this YouTube video transcript. Return JSON with:
{
  "video_summary": "100-200 word summary",
  "key_points": ["point 1", "point 2", ...],
  "businesses_mentioned": ["company or product names"],
  "people_mentioned": ["person names"],
  "main_topics": ["topic 1", "topic 2", ...]
}
Be concise and factual. Only include information present in the transcript."""


async def summarize_video(
    client: GeminiClient,
    transcript_clean: str,
    video_title: str,
    channel_name: str,
) -> dict:
    truncated = transcript_clean[:8000]
    prompt = f"Video: {video_title}\nChannel: {channel_name}\n\nTranscript:\n{truncated}"

    try:
        return await client.generate_json(prompt, system=SUMMARY_SYSTEM)
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        return {}
```

**Step 3: Write pipeline orchestrator**

```python
# src/youtube_brain/ingest/pipeline.py
import logging
from dataclasses import dataclass, field
from uuid import uuid4
from datetime import datetime, timezone

from youtube_brain.config.settings import get_settings
from youtube_brain.core.enums import BrainStatus, SourceStatus, VideoStatus, SourceType
from youtube_brain.core.models import Brain, Source, Video, Chunk
from youtube_brain.ingest.resolver import parse_youtube_url, resolve_video_ids
from youtube_brain.ingest.transcripts import fetch_transcript, clean_transcript
from youtube_brain.ingest.chunker import chunk_transcript
from youtube_brain.ingest.labeler import label_chunks
from youtube_brain.ingest.summarizer import summarize_video
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.storage.brains import insert_brain, update_brain_status, increment_video_count
from youtube_brain.storage.videos import insert_video, update_video, video_exists
from youtube_brain.storage.chunks import insert_chunks, store_embedding

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    brain_id: str = ""
    videos_found: int = 0
    videos_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


async def ingest_url(url: str, brain_name: str | None = None) -> PipelineResult:
    result = PipelineResult()
    settings = get_settings()
    client = GeminiClient()

    try:
        parsed = parse_youtube_url(url)
    except ValueError as e:
        result.errors.append(str(e))
        return result

    brain = Brain(
        name=brain_name or parsed.channel_handle or parsed.playlist_id or parsed.video_id or "Brain",
    )
    await insert_brain(brain)
    result.brain_id = str(brain.id)

    source = Source(
        brain_id=brain.id,
        source_type=SourceType(parsed.source_type),
        source_url=url,
        source_id=parsed.video_id or parsed.playlist_id or parsed.channel_id or parsed.channel_handle or "",
        status=SourceStatus.RESOLVING,
    )
    from youtube_brain.storage.database import get_session
    from youtube_brain.storage.database import sources as sources_table
    async with get_session() as session:
        from sqlalchemy.dialects.sqlite import insert
        await session.execute(insert(sources_table).values(
            id=str(source.id), brain_id=str(source.brain_id),
            source_type=source.source_type.value, source_url=source.source_url,
            source_title=source.source_title, source_id=source.source_id,
            status=source.status.value, created_at=source.created_at,
        ))

    await update_brain_status(brain.id, BrainStatus.INGESTING)

    video_entries = await resolve_video_ids(parsed)
    result.videos_found = len(video_entries)

    for i, entry in enumerate(video_entries):
        vid_id = entry["video_id"]
        if not vid_id:
            continue

        if await video_exists(brain.id, vid_id):
            continue

        video = Video(
            brain_id=brain.id,
            source_id=source.id,
            video_id=vid_id,
            title=entry.get("title"),
            channel_name=entry.get("channel_name"),
            duration_seconds=entry.get("duration_seconds"),
            url=f"https://www.youtube.com/watch?v={vid_id}",
        )

        try:
            transcript = await fetch_transcript(vid_id)
            if not transcript:
                video.status = VideoStatus.ERROR
                video.failure_reason = "No transcript available"
                await insert_video(video)
                continue

            video.transcript_raw = "\n".join(
                f"[{s['start']:.1f}] {s['text']}" for s in transcript.text_with_timestamps
            )
            video.transcript_clean = clean_transcript(transcript.full_text)
            video.transcript_source = transcript.source
            video.transcript_language = transcript.language
            video.caption_kind = "auto" if transcript.is_auto_generated else "manual"
            video.status = VideoStatus.FETCHED
            await insert_video(video)
            await increment_video_count(brain.id)

            raw_chunks = chunk_transcript(
                transcript.text_with_timestamps,
                window=settings.chunk_window_seconds,
                overlap=settings.chunk_overlap_seconds,
            )

            chunk_models = [
                Chunk(
                    video_id=video.id,
                    brain_id=brain.id,
                    start_time=rc["start_time"],
                    end_time=rc["end_time"],
                    text=rc["text"],
                )
                for rc in raw_chunks
            ]
            await insert_chunks(chunk_models)
            result.chunks_created += len(chunk_models)

            texts = [c.text for c in chunk_models]
            if texts:
                embeddings = await client.embed_texts(texts)
                for chunk_model, emb in zip(chunk_models, embeddings):
                    await store_embedding(
                        chunk_model.id, settings.gemini_embedding_model,
                        settings.gemini_embedding_dimensions, emb,
                    )

            labels = await label_chunks(
                client, raw_chunks,
                video_title=video.title or vid_id,
                channel_name=video.channel_name or "",
            )
            for chunk_model, label in zip(chunk_models, labels):
                if label:
                    from youtube_brain.storage.database import chunks as chunks_table
                    async with get_session() as session:
                        from sqlalchemy import update as sql_update
                        await session.execute(
                            sql_update(chunks_table)
                            .where(chunks_table.c.id == str(chunk_model.id))
                            .values(
                                topics=label.get("topics"),
                                business_type=label.get("business_type"),
                                advice_category=label.get("advice_category"),
                                stage=label.get("stage"),
                                asset_type=label.get("asset_type"),
                            )
                        )

            summary = await summarize_video(
                client,
                video.transcript_clean,
                video_title=video.title or vid_id,
                channel_name=video.channel_name or "",
            )
            await update_video(
                video.id,
                video_summary=summary.get("video_summary"),
                key_points=summary.get("key_points"),
                businesses_mentioned=summary.get("businesses_mentioned"),
                people_mentioned=summary.get("people_mentioned"),
                main_topics=summary.get("main_topics"),
                status=VideoStatus.SUMMARIZED,
            )

            result.videos_processed += 1

            if result.videos_processed >= settings.partially_ready_threshold:
                from youtube_brain.storage.brains import get_brain
                brain_current = await get_brain(brain.id)
                if brain_current and brain_current.status == BrainStatus.INGESTING:
                    await update_brain_status(brain.id, BrainStatus.PARTIALLY_READY)

        except Exception as e:
            logger.error("Error processing video %s: %s", vid_id, e)
            result.errors.append(f"{vid_id}: {e}")
            await update_video(video.id, status=VideoStatus.ERROR, failure_reason=str(e))

    final_status = BrainStatus.READY if result.videos_processed > 0 else BrainStatus.ERROR
    await update_brain_status(brain.id, final_status)

    await client.close()
    return result
```

**Step 4: Write pipeline test**

```python
# tests/test_pipeline.py
from youtube_brain.ingest.chunker import chunk_transcript
from youtube_brain.ingest.transcripts import clean_transcript


def test_full_chunk_flow():
    segments = [
        {"start": i * 10.0, "duration": 10.0, "text": f"This is sentence {i}."}
        for i in range(30)
    ]
    chunks = chunk_transcript(segments, window=150.0, overlap=30.0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["text"]
        assert c["start_time"] >= 0
        assert c["end_time"] > c["start_time"]


def test_clean_transcript():
    raw = "Hello [Music] world [Applause]  how  are   you"
    cleaned = clean_transcript(raw)
    assert "[Music]" not in cleaned
    assert "[Applause]" not in cleaned
    assert "  " not in cleaned
```

**Step 5: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/youtube_brain/ingest/pipeline.py src/youtube_brain/ingest/labeler.py src/youtube_brain/ingest/summarizer.py tests/test_pipeline.py
git commit -m "feat: full ingestion pipeline — resolve, fetch, chunk, embed, label, summarize"
```

---

## Task 10: Retrieval Engine

**Files:**
- Create: `src/youtube_brain/retrieval/__init__.py`
- Create: `src/youtube_brain/retrieval/search.py`
- Create: `src/youtube_brain/retrieval/reranker.py`
- Create: `tests/test_retrieval.py`

**Step 1: Write the failing test**

```python
# tests/test_retrieval.py
from youtube_brain.retrieval.reranker import diversity_select, weighted_score


def test_diversity_select_limits_per_video():
    chunks = [
        {"chunk_id": f"c{i}", "video_id": "v1", "score": 1.0 - i * 0.01}
        for i in range(10)
    ]
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=8, top_k=20)
    assert len(selected) == 3


def test_diversity_select_mixed_videos():
    chunks = []
    for v in range(5):
        for c in range(4):
            chunks.append({
                "chunk_id": f"v{v}_c{c}",
                "video_id": f"v{v}",
                "channel_name": "ch1",
                "score": 1.0 - v * 0.1 - c * 0.01,
            })
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=8, top_k=10)
    assert len(selected) == 10
    video_counts = {}
    for s in selected:
        vid = s["video_id"]
        video_counts[vid] = video_counts.get(vid, 0) + 1
    assert all(count <= 3 for count in video_counts.values())


def test_weighted_score():
    score = weighted_score(
        vector_sim=0.9, bm25=0.8, meta_match=0.5, recency=0.3,
        recency_weight=0.1,
    )
    assert 0.0 < score < 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL

**Step 3: Write reranker**

```python
# src/youtube_brain/retrieval/reranker.py
from collections import defaultdict


def weighted_score(
    vector_sim: float = 0.0,
    bm25: float = 0.0,
    meta_match: float = 0.0,
    recency: float = 0.0,
    recency_weight: float = 0.1,
) -> float:
    w_vec = 0.4
    w_bm25 = 0.3
    w_meta = 0.2
    w_rec = recency_weight
    total_other = w_vec + w_bm25 + w_meta
    scale = (1.0 - w_rec) / total_other if total_other > 0 else 1.0

    return (
        w_vec * scale * vector_sim
        + w_bm25 * scale * bm25
        + w_meta * scale * meta_match
        + w_rec * recency
    )


def diversity_select(
    chunks: list[dict],
    max_per_video: int = 3,
    max_per_channel: int = 8,
    top_k: int = 20,
) -> list[dict]:
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    video_counts: dict[str, int] = defaultdict(int)
    channel_counts: dict[str, int] = defaultdict(int)
    selected = []

    for chunk in sorted_chunks:
        if len(selected) >= top_k:
            break
        vid = chunk.get("video_id", "")
        ch = chunk.get("channel_name", "")
        if video_counts[vid] >= max_per_video:
            continue
        if ch and channel_counts[ch] >= max_per_channel:
            continue
        selected.append(chunk)
        video_counts[vid] += 1
        if ch:
            channel_counts[ch] += 1

    return selected
```

**Step 4: Write search engine**

```python
# src/youtube_brain/retrieval/search.py
import json
import logging
from dataclasses import dataclass, field

from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.storage.chunks import search_fts
from youtube_brain.storage.database import get_session, chunks, chunk_embeddings, videos
from youtube_brain.retrieval.reranker import diversity_select, weighted_score
from sqlalchemy import select, text

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    video_id: str
    video_title: str = ""
    channel_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    score: float = 0.0
    caption_kind: str = ""
    topics: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    results: list[SearchResult] = field(default_factory=list)
    chunks_searched: int = 0
    expanded_query: str = ""


EXPAND_SYSTEM = """You are a query expansion assistant for a YouTube transcript search engine.
Given a user question, produce:
1. search_query: a rewritten version optimized for semantic search
2. fts_keywords: key terms for full-text search
3. metadata_filters: matching controlled taxonomy values

Controlled values:
business_type: saas, ecommerce, agency, marketplace, content, physical_product, service, mobile_app, other
advice_category: marketing, distribution, pricing, hiring, fundraising, product, operations, customer_acquisition, retention, monetization, launch, growth, technical, legal, other
stage: idea, pre_launch, early_stage, growth, scaling, mature, exit, other

Return JSON: {"search_query": "...", "fts_keywords": [...], "metadata_filters": {"business_type": [...], "advice_category": [...], "stage": [...]}}"""


async def retrieve(
    query: str,
    brain_id: str,
    client: GeminiClient,
    recency_weight: float = 0.1,
    top_k: int = 20,
) -> RetrievalResult:
    # Step 1: query expansion (original + expanded)
    expanded = await _expand_query(client, query)

    # Step 2: four-lane retrieval
    fts_chunks = await _fts_search(query, expanded, brain_id, limit=50)
    vec_chunks = await _vector_search(query, client, brain_id, limit=50)
    fts_summaries = await _fts_summary_search(query, expanded, brain_id, limit=10)
    vec_summaries = await _vector_summary_search(query, client, brain_id, limit=10)

    # Step 3: merge + dedupe
    all_results: dict[str, dict] = {}
    for chunk in fts_chunks:
        cid = chunk["chunk_id"]
        if cid not in all_results:
            all_results[cid] = chunk
        all_results[cid]["bm25"] = chunk.get("bm25", 0.0)

    for chunk in vec_chunks:
        cid = chunk["chunk_id"]
        if cid not in all_results:
            all_results[cid] = chunk
        all_results[cid]["vector_sim"] = chunk.get("vector_sim", 0.0)

    # incorporate summary hits by boosting their source video chunks
    summary_video_ids = set()
    for s in fts_summaries + vec_summaries:
        summary_video_ids.add(s.get("video_id", ""))

    for cid, chunk in all_results.items():
        meta_match = _compute_meta_match(chunk, expanded.get("metadata_filters", {}))
        recency = 0.5  # placeholder until we add date-based scoring
        if chunk.get("video_id") in summary_video_ids:
            meta_match = min(1.0, meta_match + 0.2)

        chunk["score"] = weighted_score(
            vector_sim=chunk.get("vector_sim", 0.0),
            bm25=chunk.get("bm25", 0.0),
            meta_match=meta_match,
            recency=recency,
            recency_weight=recency_weight,
        )

    # Step 4: diversity select
    merged = list(all_results.values())
    selected = diversity_select(merged, max_per_video=3, max_per_channel=8, top_k=top_k)

    results = [
        SearchResult(
            chunk_id=s["chunk_id"],
            video_id=s.get("video_id", ""),
            video_title=s.get("video_title", ""),
            channel_name=s.get("channel_name", ""),
            start_time=s.get("start_time", 0.0),
            end_time=s.get("end_time", 0.0),
            text=s.get("text", ""),
            score=s.get("score", 0.0),
            caption_kind=s.get("caption_kind", ""),
            topics=s.get("topics", []),
        )
        for s in selected
    ]

    return RetrievalResult(
        results=results,
        chunks_searched=len(all_results),
        expanded_query=expanded.get("search_query", query),
    )


async def _expand_query(client: GeminiClient, query: str) -> dict:
    try:
        return await client.generate_json(query, system=EXPAND_SYSTEM)
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return {"search_query": query, "fts_keywords": query.split(), "metadata_filters": {}}


async def _fts_search(original: str, expanded: dict, brain_id: str, limit: int) -> list[dict]:
    keywords = expanded.get("fts_keywords", [])
    combined_query = " ".join(set(original.split() + keywords))
    fts_results = await search_fts(combined_query, brain_id, limit=limit)

    results = []
    for i, chunk in enumerate(fts_results):
        results.append({
            "chunk_id": str(chunk.id),
            "video_id": str(chunk.video_id),
            "brain_id": str(chunk.brain_id),
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "text": chunk.text,
            "topics": chunk.topics or [],
            "business_type": chunk.business_type or [],
            "advice_category": chunk.advice_category or [],
            "stage": chunk.stage or [],
            "bm25": 1.0 - (i / max(limit, 1)),
        })
    return results


async def _vector_search(query: str, client: GeminiClient, brain_id: str, limit: int) -> list[dict]:
    query_embedding = (await client.embed_texts([query]))[0]

    async with get_session() as session:
        # For MVP: load all embeddings for this brain and compute similarity in Python
        # Replace with sqlite-vec knn query once integrated
        stmt = (
            select(chunks, chunk_embeddings.c.embedding)
            .join(chunk_embeddings, chunks.c.id == chunk_embeddings.c.chunk_id)
            .where(chunks.c.brain_id == brain_id)
        )
        result = await session.execute(stmt)
        rows = result.fetchall()

    scored = []
    for row in rows:
        emb = json.loads(row.embedding)
        sim = _cosine_sim(query_embedding, emb)
        scored.append({
            "chunk_id": row.id,
            "video_id": row.video_id,
            "brain_id": row.brain_id,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "text": row.text,
            "topics": row.topics or [],
            "business_type": row.business_type or [],
            "advice_category": row.advice_category or [],
            "stage": row.stage or [],
            "vector_sim": sim,
        })

    scored.sort(key=lambda x: x["vector_sim"], reverse=True)
    return scored[:limit]


async def _fts_summary_search(original: str, expanded: dict, brain_id: str, limit: int) -> list[dict]:
    keywords = expanded.get("fts_keywords", original.split())
    combined = " OR ".join(f'"{k}"' for k in keywords if k.strip())
    if not combined:
        return []

    async with get_session() as session:
        sql = text("""
            SELECT id, video_id, title, video_summary
            FROM videos
            WHERE brain_id = :brain_id
            AND video_summary IS NOT NULL
            AND (video_summary LIKE :pattern OR title LIKE :pattern)
            LIMIT :limit
        """)
        pattern = f"%{keywords[0] if keywords else original}%"
        result = await session.execute(sql, {"brain_id": brain_id, "pattern": pattern, "limit": limit})
        return [{"video_id": row.video_id, "title": row.title} for row in result.fetchall()]


async def _vector_summary_search(query: str, client: GeminiClient, brain_id: str, limit: int) -> list[dict]:
    # Placeholder — will embed summaries separately later
    return []


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_meta_match(chunk: dict, filters: dict) -> float:
    if not filters:
        return 0.0
    matches = 0
    total = 0
    for key in ["business_type", "advice_category", "stage"]:
        filter_vals = set(filters.get(key, []))
        if not filter_vals:
            continue
        total += 1
        chunk_vals = set(chunk.get(key, []))
        if filter_vals & chunk_vals:
            matches += 1
    return matches / total if total > 0 else 0.0
```

**Step 5: Run tests**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/youtube_brain/retrieval/ tests/test_retrieval.py
git commit -m "feat: 4-lane hybrid retrieval with diversity selection and reranking"
```

---

## Task 11: Answer Generation

**Files:**
- Create: `src/youtube_brain/generation/__init__.py`
- Create: `src/youtube_brain/generation/answer.py`
- Create: `src/youtube_brain/generation/prompts.py`
- Create: `tests/test_answer.py`

**Step 1: Write prompts module**

```python
# src/youtube_brain/generation/prompts.py

QA_SYSTEM = """You are a research assistant for the "{brain_name}" knowledge base.
Answer the user's question using ONLY the provided evidence from YouTube transcripts.

Rules:
- Cite every claim with [Video Title | timestamp]
- Never invent information beyond what's in the chunks
- If evidence is insufficient, say so explicitly
- Use the raw transcript text for accuracy
- Prefer citing multiple independent sources over a single source"""

ARTICLE_SYSTEM = """You are a writer creating a researched article for the "{brain_name}" knowledge base.
Write a long-form article using ONLY the provided evidence from YouTube transcripts.

Rules:
- Structure with clear headings
- Cite every major claim with [Video Title | timestamp]
- Synthesize across multiple sources when possible
- Be factual and grounded — no speculation"""

PLAYBOOK_SYSTEM = """You are creating a step-by-step action plan based on the "{brain_name}" knowledge base.
Build a practical playbook using ONLY the provided evidence from YouTube transcripts.

Rules:
- Number each step clearly
- For each step, cite the source: [Video Title | timestamp]
- Prioritize frequently repeated advice across multiple videos
- Be specific and actionable"""

SUMMARY_SYSTEM = """You are summarizing what the "{brain_name}" knowledge base contains about a topic.
Create a thematic overview using ONLY the provided evidence.

Rules:
- Group findings by theme
- Cite key claims: [Video Title | timestamp]
- Note when multiple sources agree
- Be concise but comprehensive"""

FAQ_SYSTEM = """You are generating a FAQ based on the "{brain_name}" knowledge base.
Create questions this brain can answer well, based on the provided evidence.

Rules:
- Generate 5-10 questions with short answers
- Each answer must cite at least one source: [Video Title | timestamp]
- Focus on questions where the evidence is strong
- Order by likely usefulness"""

PROMPTS = {
    "qa": QA_SYSTEM,
    "article": ARTICLE_SYSTEM,
    "playbook": PLAYBOOK_SYSTEM,
    "summary": SUMMARY_SYSTEM,
    "faq": FAQ_SYSTEM,
}
```

**Step 2: Write answer generator**

```python
# src/youtube_brain/generation/answer.py
import re
import logging
from dataclasses import dataclass, field

from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.retrieval.search import SearchResult, RetrievalResult, retrieve
from youtube_brain.generation.prompts import PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    video_title: str
    video_url: str
    timestamp: float
    timestamp_display: str
    transcript_text: str
    caption_kind: str
    chunk_id: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    chunks_searched: int = 0
    chunks_used: int = 0
    mode: str = "qa"


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_context(results: list[SearchResult]) -> str:
    lines = []
    for r in results:
        ts_start = _format_timestamp(r.start_time)
        ts_end = _format_timestamp(r.end_time)
        lines.append(
            f"[Video: {r.video_title} | Channel: {r.channel_name}]\n"
            f"[Timestamp: {ts_start} – {ts_end} | Caption: {r.caption_kind}]\n"
            f"[Topics: {', '.join(r.topics)}]\n"
            f"---\n{r.text}\n---\n"
        )
    return "\n\n".join(lines)


def _build_citations(results: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            video_title=r.video_title,
            video_url=f"https://youtu.be/{r.video_id}?t={int(r.start_time)}",
            timestamp=r.start_time,
            timestamp_display=_format_timestamp(r.start_time),
            transcript_text=r.text[:200],
            caption_kind=r.caption_kind,
            chunk_id=r.chunk_id,
        )
        for r in results
    ]


def _compute_confidence(results: list[SearchResult]) -> dict:
    video_ids = set(r.video_id for r in results)
    caption_kinds = [r.caption_kind for r in results if r.caption_kind]
    manual_count = sum(1 for k in caption_kinds if k == "manual")

    if len(results) >= 10 and len(video_ids) >= 5:
        level = "high"
    elif len(results) >= 5 and len(video_ids) >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "supporting_chunks": len(results),
        "supporting_videos": len(video_ids),
        "caption_quality": "mostly_manual" if manual_count > len(caption_kinds) / 2 else "mostly_auto",
    }


async def generate_answer(
    query: str,
    brain_id: str,
    brain_name: str,
    client: GeminiClient,
    mode: str = "qa",
    recency_weight: float = 0.1,
) -> AnswerResult:
    retrieval = await retrieve(query, brain_id, client, recency_weight=recency_weight)

    if not retrieval.results:
        return AnswerResult(
            answer="I couldn't find enough evidence in this brain to answer your question.",
            confidence={"level": "none", "supporting_chunks": 0, "supporting_videos": 0},
            mode=mode,
        )

    context = _build_context(retrieval.results)
    system_prompt = PROMPTS.get(mode, PROMPTS["qa"]).format(brain_name=brain_name)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    answer_text = await client.generate(prompt, system=system_prompt, temperature=0.4)

    return AnswerResult(
        answer=answer_text,
        citations=_build_citations(retrieval.results),
        confidence=_compute_confidence(retrieval.results),
        chunks_searched=retrieval.chunks_searched,
        chunks_used=len(retrieval.results),
        mode=mode,
    )
```

**Step 3: Write test**

```python
# tests/test_answer.py
from youtube_brain.generation.answer import _format_timestamp, _compute_confidence
from youtube_brain.retrieval.search import SearchResult


def test_format_timestamp():
    assert _format_timestamp(0) == "0:00"
    assert _format_timestamp(65) == "1:05"
    assert _format_timestamp(3661) == "1:01:01"


def test_confidence_high():
    results = [
        SearchResult(chunk_id=f"c{i}", video_id=f"v{i % 6}", text="x")
        for i in range(12)
    ]
    conf = _compute_confidence(results)
    assert conf["level"] == "high"
    assert conf["supporting_chunks"] == 12


def test_confidence_low():
    results = [SearchResult(chunk_id="c1", video_id="v1", text="x")]
    conf = _compute_confidence(results)
    assert conf["level"] == "low"
```

**Step 4: Run tests**

Run: `pytest tests/test_answer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/generation/ tests/test_answer.py
git commit -m "feat: answer generation with 5 output modes, citations, and confidence scoring"
```

---

## Task 12: FastAPI Endpoints

**Files:**
- Create: `src/youtube_brain/api/__init__.py`
- Create: `src/youtube_brain/api/app.py`
- Create: `src/youtube_brain/api/routes.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL

**Step 3: Write FastAPI app**

```python
# src/youtube_brain/api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from youtube_brain.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="YouTube Brain", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app
```

**Step 4: Write routes**

```python
# src/youtube_brain/api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from youtube_brain.storage.brains import list_brains, get_brain
from youtube_brain.storage.videos import get_videos_by_brain
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.generation.answer import generate_answer

router = APIRouter()


class IngestRequest(BaseModel):
    url: str
    name: str | None = None


class AskRequest(BaseModel):
    query: str
    mode: str = "qa"


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/api/brains")
async def api_list_brains():
    brains_list = await list_brains()
    return [
        {
            "id": str(b.id),
            "name": b.name,
            "status": b.status.value,
            "video_count": b.video_count,
            "visibility": b.visibility,
            "created_at": b.created_at.isoformat(),
        }
        for b in brains_list
    ]


@router.get("/api/brains/{brain_id}")
async def api_get_brain(brain_id: str):
    brain = await get_brain(brain_id)
    if not brain:
        raise HTTPException(status_code=404, detail="Brain not found")
    vids = await get_videos_by_brain(brain_id)
    return {
        "id": str(brain.id),
        "name": brain.name,
        "status": brain.status.value,
        "video_count": brain.video_count,
        "visibility": brain.visibility,
        "recency_weight": brain.recency_weight,
        "created_at": brain.created_at.isoformat(),
        "videos": [
            {
                "id": str(v.id),
                "video_id": v.video_id,
                "title": v.title,
                "status": v.status.value,
                "caption_kind": v.caption_kind,
                "video_summary": v.video_summary,
            }
            for v in vids
        ],
    }


@router.post("/api/brains/ingest")
async def api_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    from youtube_brain.ingest.pipeline import ingest_url
    from youtube_brain.storage.database import init_database

    await init_database()

    async def _run():
        await ingest_url(req.url, brain_name=req.name)

    background_tasks.add_task(_run)
    return {"status": "ingesting", "url": req.url}


@router.post("/api/brains/{brain_id}/ask")
async def api_ask(brain_id: str, req: AskRequest):
    brain = await get_brain(brain_id)
    if not brain:
        raise HTTPException(status_code=404, detail="Brain not found")

    client = GeminiClient()
    try:
        result = await generate_answer(
            query=req.query,
            brain_id=brain_id,
            brain_name=brain.name,
            client=client,
            mode=req.mode,
            recency_weight=brain.recency_weight,
        )
        return {
            "answer": result.answer,
            "citations": [
                {
                    "video_title": c.video_title,
                    "video_url": c.video_url,
                    "timestamp": c.timestamp,
                    "timestamp_display": c.timestamp_display,
                    "transcript_text": c.transcript_text,
                    "caption_kind": c.caption_kind,
                }
                for c in result.citations
            ],
            "confidence": result.confidence,
            "chunks_searched": result.chunks_searched,
            "chunks_used": result.chunks_used,
            "mode": result.mode,
        }
    finally:
        await client.close()
```

**Step 5: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/youtube_brain/api/ tests/test_api.py
git commit -m "feat: FastAPI endpoints for brains, ingestion, and Q&A"
```

---

## Task 13: CLI Entry Point

**Files:**
- Create: `src/youtube_brain/cli.py`

**Step 1: Write CLI**

```python
# src/youtube_brain/cli.py
import asyncio
import logging
import click

from youtube_brain.config.settings import get_settings


def run_async(coro):
    return asyncio.run(coro)


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@main.command()
@click.argument("url")
@click.option("--name", "-n", help="Brain name")
def ingest(url: str, name: str | None):
    """Ingest a YouTube URL into a new brain."""
    from youtube_brain.storage.database import init_database
    from youtube_brain.ingest.pipeline import ingest_url

    async def _run():
        await init_database()
        result = await ingest_url(url, brain_name=name)
        click.echo(f"Brain: {result.brain_id}")
        click.echo(f"Videos found: {result.videos_found}")
        click.echo(f"Videos processed: {result.videos_processed}")
        click.echo(f"Chunks created: {result.chunks_created}")
        if result.errors:
            click.echo(f"Errors: {len(result.errors)}")
            for e in result.errors[:5]:
                click.echo(f"  - {e}")

    run_async(_run())


@main.command()
@click.argument("brain_id")
@click.argument("query")
@click.option("--mode", "-m", default="qa", type=click.Choice(["qa", "article", "playbook", "summary", "faq"]))
def ask(brain_id: str, query: str, mode: str):
    """Ask a brain a question."""
    from youtube_brain.storage.database import init_database
    from youtube_brain.storage.brains import get_brain
    from youtube_brain.llm.gemini import GeminiClient
    from youtube_brain.generation.answer import generate_answer

    async def _run():
        await init_database()
        brain = await get_brain(brain_id)
        if not brain:
            click.echo(f"Brain {brain_id} not found")
            return

        client = GeminiClient()
        result = await generate_answer(
            query=query, brain_id=brain_id, brain_name=brain.name,
            client=client, mode=mode, recency_weight=brain.recency_weight,
        )
        click.echo(f"\n{result.answer}\n")
        click.echo(f"Confidence: {result.confidence.get('level', 'unknown')}")
        click.echo(f"Chunks: {result.chunks_used}/{result.chunks_searched}")
        if result.citations:
            click.echo("\nSources:")
            for c in result.citations[:5]:
                click.echo(f"  [{c.timestamp_display}] {c.video_title}")
                click.echo(f"    {c.video_url}")
        await client.close()

    run_async(_run())


@main.command()
def serve():
    """Start the API server."""
    import uvicorn
    from youtube_brain.storage.database import init_database

    run_async(init_database())
    settings = get_settings()
    uvicorn.run(
        "youtube_brain.api.app:create_app",
        host=settings.api_host,
        port=settings.api_port,
        factory=True,
        reload=True,
    )


@main.command("list")
def list_brains():
    """List all brains."""
    from youtube_brain.storage.database import init_database
    from youtube_brain.storage.brains import list_brains as _list_brains

    async def _run():
        await init_database()
        brains_list = await _list_brains()
        if not brains_list:
            click.echo("No brains yet.")
            return
        for b in brains_list:
            click.echo(f"[{b.status.value:16s}] {b.name} ({b.video_count} videos) — {b.id}")

    run_async(_run())
```

**Step 2: Verify CLI works**

Run: `ytbrain --help`
Expected: Shows help with ingest, ask, serve, list commands.

**Step 3: Commit**

```bash
git add src/youtube_brain/cli.py
git commit -m "feat: CLI with ingest, ask, serve, and list commands"
```

---

## Task 14: React PWA Frontend

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/components/BrainCard.tsx`
- Create: `frontend/src/components/BrainDetail.tsx`
- Create: `frontend/src/components/AskPanel.tsx`
- Create: `frontend/src/components/CitationList.tsx`
- Create: `frontend/src/components/IngestDialog.tsx`

This is a larger task. The plan provides the component structure and API contract — the implementation engineer should scaffold with Vite and build each component.

**Step 1: Scaffold React app**

Run:
```bash
cd "C:\Python Projects\Youtube Brain"
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

**Step 2: Create API client**

```typescript
// frontend/src/api.ts
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Brain {
  id: string;
  name: string;
  status: string;
  video_count: number;
  visibility: string;
  created_at: string;
}

export interface Citation {
  video_title: string;
  video_url: string;
  timestamp: number;
  timestamp_display: string;
  transcript_text: string;
  caption_kind: string;
}

export interface AnswerResponse {
  answer: string;
  citations: Citation[];
  confidence: {
    level: string;
    supporting_chunks: number;
    supporting_videos: number;
    caption_quality: string;
  };
  chunks_searched: number;
  chunks_used: number;
  mode: string;
}

export async function listBrains(): Promise<Brain[]> {
  const res = await fetch(`${BASE}/api/brains`);
  return res.json();
}

export async function getBrain(id: string) {
  const res = await fetch(`${BASE}/api/brains/${id}`);
  return res.json();
}

export async function ingestUrl(url: string, name?: string) {
  const res = await fetch(`${BASE}/api/brains/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name }),
  });
  return res.json();
}

export async function askBrain(
  brainId: string,
  query: string,
  mode: string = 'qa'
): Promise<AnswerResponse> {
  const res = await fetch(`${BASE}/api/brains/${brainId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode }),
  });
  return res.json();
}
```

**Step 3: Build each component**

The engineer should create these components following standard React patterns:
- `BrainCard.tsx` — card in the grid showing brain name, status, video count
- `BrainDetail.tsx` — detail view with tabs (Ask, Articles, Videos, Settings)
- `AskPanel.tsx` — question input, mode selector, answer display with markdown
- `CitationList.tsx` — sidebar/list of citations with clickable YouTube timestamp links
- `IngestDialog.tsx` — modal to paste a YouTube URL and optional brain name
- `App.tsx` — router between brain list and brain detail views

**Step 4: Test the frontend**

Run:
```bash
cd frontend
npm run dev
```

Open browser, verify:
- Brain list loads (empty initially)
- Ingest dialog opens and submits
- Brain detail view shows after ingestion

**Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: React PWA frontend with brain cards, Q&A, and citations"
```

---

## Task 15: End-to-End Integration Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write integration test**

```python
# tests/test_e2e.py
import pytest


@pytest.mark.integration
async def test_ingest_single_video_and_ask(tmp_settings):
    """Full pipeline test with a real YouTube video. Run: pytest -m integration"""
    import os
    if not os.environ.get("YTBRAIN_GEMINI_API_KEY"):
        pytest.skip("No Gemini API key")

    from youtube_brain.storage.database import init_database
    from youtube_brain.ingest.pipeline import ingest_url
    from youtube_brain.storage.brains import get_brain
    from youtube_brain.llm.gemini import GeminiClient
    from youtube_brain.generation.answer import generate_answer

    await init_database(tmp_settings)

    # Ingest a short, well-known video
    result = await ingest_url(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        brain_name="Test Brain",
    )
    assert result.videos_processed >= 1
    assert result.chunks_created >= 1

    brain = await get_brain(result.brain_id)
    assert brain is not None
    assert brain.name == "Test Brain"

    # Ask a question
    client = GeminiClient()
    answer = await generate_answer(
        query="What is this video about?",
        brain_id=result.brain_id,
        brain_name=brain.name,
        client=client,
    )
    assert len(answer.answer) > 0
    assert answer.chunks_used > 0
    await client.close()
```

**Step 2: Run unit tests (no integration)**

Run: `pytest tests/ -v -k "not integration" --ignore=tests/test_e2e.py`
Expected: All PASS

**Step 3: Run integration test (requires API key)**

Run: `YTBRAIN_GEMINI_API_KEY=your-key pytest tests/test_e2e.py -v -m integration`
Expected: PASS — full pipeline runs end to end

**Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "feat: end-to-end integration test for full ingest + ask pipeline"
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 1 | Project scaffolding | `pyproject.toml`, `settings.py` |
| 2 | Database schema | `storage/database.py` (7 tables + FTS5) |
| 3 | Models & enums | `core/models.py`, `core/enums.py` |
| 4 | Storage CRUD | `storage/brains.py`, `videos.py`, `chunks.py` |
| 5 | Transcript fetcher | `ingest/transcripts.py` |
| 6 | URL resolver | `ingest/resolver.py` |
| 7 | Chunking engine | `ingest/chunker.py` |
| 8 | Gemini client | `llm/gemini.py` |
| 9 | Ingestion pipeline | `ingest/pipeline.py`, `labeler.py`, `summarizer.py` |
| 10 | Retrieval engine | `retrieval/search.py`, `reranker.py` |
| 11 | Answer generation | `generation/answer.py`, `prompts.py` |
| 12 | FastAPI endpoints | `api/app.py`, `routes.py` |
| 13 | CLI | `cli.py` |
| 14 | React PWA | `frontend/` |
| 15 | E2E test | `tests/test_e2e.py` |

**Estimated time:** Tasks 1-13 (backend) ~2-3 sessions. Task 14 (frontend) ~1-2 sessions. Task 15 (integration) ~30 minutes.

**First milestone:** After Task 9, you can `ytbrain ingest <url>` and see transcripts chunked + embedded in SQLite. After Task 13, you can `ytbrain ask <brain_id> "question"` from the command line. That's when you'll know if it feels magical.
