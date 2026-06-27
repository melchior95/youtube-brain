"""Tests for Pydantic models and enums."""

from youtube_brain.core.enums import (
    BrainStatus,
    BusinessType,
    SourceType,
)
from youtube_brain.core.models import Brain, Chunk


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
