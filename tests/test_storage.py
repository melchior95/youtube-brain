"""Tests for storage CRUD operations."""

import pytest

from youtube_brain.core.enums import (
    BrainStatus,
    SourceType,
    VideoStatus,
)
from youtube_brain.core.models import Brain, Chunk, Source, Video
from youtube_brain.storage.brains import (
    get_brain,
    increment_video_count,
    insert_brain,
    list_brains,
    update_brain_status,
)
from youtube_brain.storage.chunks import (
    get_chunks_by_video,
    insert_chunks,
    search_fts,
    store_embedding,
)
from youtube_brain.storage.database import close_database, init_database
from youtube_brain.storage.videos import (
    get_videos_by_brain,
    insert_video,
    update_video,
    video_exists,
)


@pytest.fixture
async def db(tmp_settings):
    await init_database(tmp_settings)
    yield
    await close_database()


# ---------------------------------------------------------------------------
# Brains
# ---------------------------------------------------------------------------


async def test_insert_and_get_brain(db):
    brain = Brain(name="Test Brain")
    await insert_brain(brain)
    fetched = await get_brain(brain.id)
    assert fetched is not None
    assert fetched.name == "Test Brain"
    assert fetched.status == BrainStatus.PENDING


async def test_insert_brain_returns_true_on_first_insert(db):
    brain = Brain(name="Fresh Brain")
    result = await insert_brain(brain)
    assert result is True


async def test_insert_brain_duplicate_returns_false(db):
    brain = Brain(name="Dup Brain")
    await insert_brain(brain)
    result = await insert_brain(brain)
    assert result is False


async def test_get_brain_not_found(db):
    fetched = await get_brain("00000000-0000-0000-0000-000000000000")
    assert fetched is None


async def test_list_brains(db):
    await insert_brain(Brain(name="Alpha"))
    await insert_brain(Brain(name="Beta"))
    result = await list_brains()
    assert len(result) >= 2
    names = [b.name for b in result]
    assert "Alpha" in names
    assert "Beta" in names


async def test_update_brain_status(db):
    brain = Brain(name="Status Brain")
    await insert_brain(brain)
    await update_brain_status(brain.id, BrainStatus.INGESTING)
    fetched = await get_brain(brain.id)
    assert fetched.status == BrainStatus.INGESTING


async def test_increment_video_count(db):
    brain = Brain(name="Counter Brain")
    await insert_brain(brain)
    await increment_video_count(brain.id, 3)
    fetched = await get_brain(brain.id)
    assert fetched.video_count == 3
    await increment_video_count(brain.id)
    fetched = await get_brain(brain.id)
    assert fetched.video_count == 4


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def _make_video(brain: Brain, source: Source, video_id: str = "abc") -> Video:
    """Helper to create a Video linked to a brain and source."""
    return Video(
        brain_id=brain.id,
        source_id=source.id,
        video_id=video_id,
        url=f"https://youtube.com/watch?v={video_id}",
    )


async def test_insert_and_get_video(db):
    brain = Brain(name="Vid Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=abc",
        source_id="abc",
    )
    video = _make_video(brain, source)
    inserted = await insert_video(video)
    assert inserted is True

    vids = await get_videos_by_brain(brain.id)
    assert len(vids) == 1
    assert vids[0].video_id == "abc"


async def test_get_videos_by_brain_with_status_filter(db):
    brain = Brain(name="Filter Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=x",
        source_id="x",
    )
    v1 = _make_video(brain, source, "v1")
    v2 = _make_video(brain, source, "v2")
    await insert_video(v1)
    await insert_video(v2)
    await update_video(v2.id, status=VideoStatus.FETCHED)

    pending = await get_videos_by_brain(brain.id, status=VideoStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].video_id == "v1"

    fetched = await get_videos_by_brain(brain.id, status=VideoStatus.FETCHED)
    assert len(fetched) == 1
    assert fetched[0].video_id == "v2"


async def test_video_exists(db):
    brain = Brain(name="Exists Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=xyz",
        source_id="xyz",
    )
    video = _make_video(brain, source, "xyz")
    await insert_video(video)

    assert await video_exists(brain.id, "xyz") is True
    assert await video_exists(brain.id, "nope") is False


async def test_update_video(db):
    brain = Brain(name="Update Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=upd",
        source_id="upd",
    )
    video = _make_video(brain, source, "upd")
    await insert_video(video)

    await update_video(video.id, title="Updated Title", status=VideoStatus.FETCHED)
    vids = await get_videos_by_brain(brain.id)
    assert vids[0].title == "Updated Title"
    assert vids[0].status == VideoStatus.FETCHED


# ---------------------------------------------------------------------------
# Chunks & FTS
# ---------------------------------------------------------------------------


async def test_insert_and_get_chunks(db):
    brain = Brain(name="Chunk Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=ch1",
        source_id="ch1",
    )
    video = _make_video(brain, source, "ch1")
    await insert_video(video)

    c1 = Chunk(
        video_id=video.id, brain_id=brain.id,
        start_time=0.0, end_time=150.0, text="First chunk",
    )
    c2 = Chunk(
        video_id=video.id, brain_id=brain.id,
        start_time=150.0, end_time=300.0, text="Second chunk",
    )
    count = await insert_chunks([c2, c1])  # insert out of order
    assert count == 2

    fetched = await get_chunks_by_video(video.id)
    assert len(fetched) == 2
    # Should be ordered by start_time
    assert fetched[0].start_time == 0.0
    assert fetched[1].start_time == 150.0


async def test_insert_chunks_empty(db):
    count = await insert_chunks([])
    assert count == 0


async def test_insert_and_search_chunks(db):
    brain = Brain(name="Chunk Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=abc",
        source_id="abc",
    )
    video = Video(
        brain_id=brain.id,
        source_id=source.id,
        video_id="abc",
        url="https://youtube.com/watch?v=abc",
    )
    await insert_video(video)
    chunk = Chunk(
        video_id=video.id,
        brain_id=brain.id,
        start_time=0.0,
        end_time=150.0,
        text="Reddit was our best marketing channel for SaaS growth",
    )
    await insert_chunks([chunk])
    results = await search_fts("Reddit marketing", str(brain.id), limit=10)
    assert len(results) >= 1
    assert "Reddit" in results[0].text


async def test_search_fts_no_results(db):
    brain = Brain(name="Empty Search Brain")
    await insert_brain(brain)
    results = await search_fts("nonexistent", str(brain.id), limit=10)
    assert results == []


async def test_store_embedding(db):
    brain = Brain(name="Embed Brain")
    await insert_brain(brain)
    source = Source(
        brain_id=brain.id,
        source_type=SourceType.VIDEO,
        source_url="https://youtube.com/watch?v=emb",
        source_id="emb",
    )
    video = _make_video(brain, source, "emb")
    await insert_video(video)
    chunk = Chunk(
        video_id=video.id, brain_id=brain.id,
        start_time=0.0, end_time=60.0, text="embedding test",
    )
    await insert_chunks([chunk])

    embedding = [0.1, 0.2, 0.3]
    await store_embedding(chunk.id, "text-embedding-004", 3, embedding)

    # Upsert again to verify no error
    await store_embedding(chunk.id, "text-embedding-004", 3, [0.4, 0.5, 0.6])
