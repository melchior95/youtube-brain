"""Tests for the answer generation module."""

from youtube_brain.generation.answer import (
    _build_citations,
    _compute_confidence,
    _format_timestamp,
)
from youtube_brain.retrieval.search import SearchResult


def _make_result(
    chunk_id: str = "c1",
    video_id: str = "v1",
    youtube_id: str = "yt1",
    text: str = "x",
    video_title: str = "",
    channel_name: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
    score: float = 0.0,
    caption_kind: str = "",
    topics: list[str] | None = None,
) -> SearchResult:
    """Convenience factory for :class:`SearchResult` with sensible defaults."""
    return SearchResult(
        chunk_id=chunk_id,
        video_id=video_id,
        youtube_id=youtube_id,
        text=text,
        video_title=video_title,
        channel_name=channel_name,
        start_time=start_time,
        end_time=end_time,
        score=score,
        caption_kind=caption_kind,
        topics=topics if topics is not None else [],
    )


def test_format_timestamp():
    assert _format_timestamp(0) == "0:00"
    assert _format_timestamp(65) == "1:05"
    assert _format_timestamp(3661) == "1:01:01"


def test_confidence_high():
    results = [
        _make_result(chunk_id=f"c{i}", video_id=f"v{i % 6}")
        for i in range(12)
    ]
    conf = _compute_confidence(results)
    assert conf["level"] == "high"
    assert conf["supporting_chunks"] == 12


def test_confidence_low():
    results = [_make_result(chunk_id="c1", video_id="v1")]
    conf = _compute_confidence(results)
    assert conf["level"] == "low"


def test_confidence_medium():
    results = [
        _make_result(chunk_id=f"c{i}", video_id=f"v{i % 3}")
        for i in range(6)
    ]
    conf = _compute_confidence(results)
    assert conf["level"] == "medium"


def test_build_citations():
    results = [
        SearchResult(
            chunk_id="c1",
            video_id="vid123",
            youtube_id="ytABC",
            video_title="Test",
            channel_name="TestChannel",
            start_time=65.0,
            end_time=120.0,
            text="Some text here",
            score=0.9,
            caption_kind="manual",
        )
    ]
    citations = _build_citations(results)
    assert len(citations) == 1
    assert "t=65" in citations[0].video_url
    assert "ytABC" in citations[0].video_url
    assert citations[0].timestamp_display == "1:05"


def test_build_citations_truncates_text():
    long_text = "a" * 300
    results = [
        _make_result(chunk_id="c1", video_id="v1", text=long_text, start_time=10.0)
    ]
    citations = _build_citations(results)
    assert len(citations[0].transcript_text) == 200


def test_confidence_caption_quality_manual():
    results = [
        _make_result(chunk_id=f"c{i}", video_id=f"v{i}", caption_kind="manual")
        for i in range(4)
    ]
    conf = _compute_confidence(results)
    assert conf["caption_quality"] == "mostly_manual"


def test_confidence_caption_quality_auto():
    results = [
        _make_result(chunk_id=f"c{i}", video_id=f"v{i}", caption_kind="auto")
        for i in range(4)
    ]
    conf = _compute_confidence(results)
    assert conf["caption_quality"] == "mostly_auto"


def test_format_timestamp_edge_cases():
    assert _format_timestamp(59) == "0:59"
    assert _format_timestamp(60) == "1:00"
    assert _format_timestamp(3600) == "1:00:00"
    assert _format_timestamp(7261) == "2:01:01"
