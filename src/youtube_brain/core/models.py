"""Pydantic models for the YouTube Brain domain."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from youtube_brain.core.enums import (
    ArticleType,
    BrainStatus,
    SourceStatus,
    SourceType,
    TranscriptSource,
    VideoStatus,
)


def _now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
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
    caption_kind: str | None = None
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
    video_id: UUID | str | None = None
    youtube_id: str | None = None
    creator: str | None = None
    obs_type: str
    claim: str
    value: str | None = None
    entities: list[str] | None = None
    evidence_quote: str | None = None
    chunk_id: str | None = None
    start_time: float | None = None
    confidence: float | None = None
    domain: str = "founders"
    cluster_id: int | None = None
    created_at: datetime = Field(default_factory=_now)


class Article(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    brain_id: UUID
    title: str
    body: str
    article_type: ArticleType
    source_chunk_ids: list[str] | None = None
    created_at: datetime = Field(default_factory=_now)
