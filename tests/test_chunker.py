"""Tests for transcript chunking engine."""

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
    segments = [{"start": 0.0, "duration": 5.0, "text": "Short video."}]
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


def test_empty_segments():
    chunks = chunk_transcript([], window=150.0, overlap=30.0)
    assert chunks == []
