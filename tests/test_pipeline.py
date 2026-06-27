"""Tests for the ingestion pipeline orchestrator (transcript -> chunks -> FTS)."""

from unittest.mock import AsyncMock, patch

import pytest

from youtube_brain.core.enums import BrainStatus
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


class TestPipelineOrchestration:
    """Test the full pipeline with all external dependencies mocked.

    The lean pipeline does no embedding, labeling, or summarizing, so there's no
    Gemini client to mock: just transcript fetch, chunk, and storage.
    """

    @pytest.fixture
    def mock_transcript_result(self):
        from youtube_brain.ingest.transcripts import TranscriptResult

        return TranscriptResult(
            text_with_timestamps=[
                {"start": 0.0, "duration": 10.0, "text": "Hello world this is a test."},
                {"start": 10.0, "duration": 10.0, "text": "Second segment here."},
                {"start": 20.0, "duration": 10.0, "text": "Third segment now."},
            ],
            full_text="Hello world this is a test. Second segment here. Third segment now.",
            language="en",
            is_auto_generated=False,
            source="api",
        )

    async def test_pipeline_single_video(self, mock_transcript_result):
        """Verify the orchestration flow for a single video URL."""
        from youtube_brain.ingest.pipeline import ingest_url

        with (
            patch("youtube_brain.ingest.pipeline.insert_brain", new_callable=AsyncMock) as mock_insert_brain,
            patch("youtube_brain.ingest.pipeline.update_brain_status", new_callable=AsyncMock) as mock_update_status,
            patch("youtube_brain.ingest.pipeline.increment_video_count", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.video_exists", new_callable=AsyncMock, return_value=False),
            patch(
                "youtube_brain.ingest.pipeline.fetch_transcript",
                new_callable=AsyncMock,
                return_value=mock_transcript_result,
            ) as mock_fetch,
            patch("youtube_brain.ingest.pipeline.insert_video", new_callable=AsyncMock, return_value=True),
            patch("youtube_brain.ingest.pipeline.insert_chunks", new_callable=AsyncMock, return_value=1),
            patch("youtube_brain.ingest.pipeline.update_video", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.embed_texts", return_value=[[0.1, 0.2, 0.3]]),
            patch("youtube_brain.ingest.pipeline.store_embedding", new_callable=AsyncMock),
        ):
            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            mock_insert_brain.assert_called_once()

            status_calls = [call.args[1] for call in mock_update_status.call_args_list]
            assert status_calls[0] == BrainStatus.INGESTING
            assert status_calls[-1] == BrainStatus.READY

            mock_fetch.assert_called_once_with("abc123")

            assert result.videos_found == 1
            assert result.videos_processed == 1
            assert result.chunks_created >= 1
            assert len(result.errors) == 0

    async def test_pipeline_skips_existing_video(self, mock_transcript_result):
        """Verify that existing videos are skipped."""
        from youtube_brain.ingest.pipeline import ingest_url

        with (
            patch("youtube_brain.ingest.pipeline.insert_brain", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_brain_status", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.video_exists", new_callable=AsyncMock, return_value=True),
            patch(
                "youtube_brain.ingest.pipeline.fetch_transcript",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
        ):
            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            mock_fetch.assert_not_called()
            assert result.videos_processed == 0

    async def test_pipeline_handles_transcript_failure(self):
        """Verify that a transcript failure is handled gracefully."""
        from youtube_brain.ingest.pipeline import ingest_url

        with (
            patch("youtube_brain.ingest.pipeline.insert_brain", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_brain_status", new_callable=AsyncMock) as mock_update_status,
            patch("youtube_brain.ingest.pipeline.video_exists", new_callable=AsyncMock, return_value=False),
            patch(
                "youtube_brain.ingest.pipeline.fetch_transcript",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
        ):
            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            assert result.videos_processed == 0
            assert len(result.errors) == 1
            assert "No transcript" in result.errors[0]

            status_calls = [call.args[1] for call in mock_update_status.call_args_list]
            assert status_calls[-1] == BrainStatus.ERROR

    async def test_pipeline_resolve_failure(self):
        """Verify that a resolve failure sets brain to ERROR."""
        from youtube_brain.ingest.pipeline import ingest_url

        with (
            patch("youtube_brain.ingest.pipeline.insert_brain", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_brain_status", new_callable=AsyncMock) as mock_update_status,
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
            patch(
                "youtube_brain.ingest.pipeline.resolve_video_ids",
                side_effect=RuntimeError("yt-dlp not found"),
            ),
        ):
            result = await ingest_url("https://www.youtube.com/@somechannel")

            assert result.videos_found == 0
            assert result.videos_processed == 0
            assert len(result.errors) == 1
            assert "resolve" in result.errors[0].lower() or "yt-dlp" in result.errors[0].lower()

            status_calls = [call.args[1] for call in mock_update_status.call_args_list]
            assert BrainStatus.ERROR in status_calls

    async def test_pipeline_continues_on_single_video_error(self, mock_transcript_result):
        """Verify that an error on one video doesn't stop processing of others."""
        from youtube_brain.ingest.pipeline import ingest_url

        async def fetch_side_effect(vid_id):
            if vid_id == "vid1":
                raise Exception("Network error")
            return mock_transcript_result

        with (
            patch("youtube_brain.ingest.pipeline.insert_brain", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_brain_status", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.increment_video_count", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.video_exists", new_callable=AsyncMock, return_value=False),
            patch(
                "youtube_brain.ingest.pipeline.fetch_transcript",
                new_callable=AsyncMock,
                side_effect=fetch_side_effect,
            ),
            patch("youtube_brain.ingest.pipeline.insert_video", new_callable=AsyncMock, return_value=True),
            patch("youtube_brain.ingest.pipeline.insert_chunks", new_callable=AsyncMock, return_value=1),
            patch("youtube_brain.ingest.pipeline.update_video", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.embed_texts", return_value=[[0.1, 0.2, 0.3]]),
            patch("youtube_brain.ingest.pipeline.store_embedding", new_callable=AsyncMock),
            patch(
                "youtube_brain.ingest.pipeline.resolve_video_ids",
                return_value=[
                    {"video_id": "vid1", "title": "Video 1", "channel_name": "Ch", "duration_seconds": 100},
                    {"video_id": "vid2", "title": "Video 2", "channel_name": "Ch", "duration_seconds": 200},
                ],
            ),
        ):
            result = await ingest_url("https://www.youtube.com/playlist?list=PLtest")

            assert result.videos_found == 2
            assert result.videos_processed == 1
            assert len(result.errors) >= 1
            assert any("vid1" in e for e in result.errors)


class TestPipelineHelpers:
    """Test helper functions in the pipeline module."""

    def test_format_transcript_raw(self):
        from youtube_brain.ingest.pipeline import _format_transcript_raw

        segments = [
            {"start": 0.0, "duration": 5.0, "text": "Hello"},
            {"start": 65.5, "duration": 3.0, "text": "World"},
        ]
        raw = _format_transcript_raw(segments)
        assert "[00:00.00] Hello" in raw
        assert "[01:05.50] World" in raw

    def test_derive_brain_name_handle(self):
        from youtube_brain.ingest.pipeline import _derive_brain_name
        from youtube_brain.ingest.resolver import parse_youtube_url

        parsed = parse_youtube_url("https://www.youtube.com/@testchannel")
        name = _derive_brain_name(parsed, "https://www.youtube.com/@testchannel")
        assert name == "@testchannel"

    def test_derive_brain_name_video(self):
        from youtube_brain.ingest.pipeline import _derive_brain_name
        from youtube_brain.ingest.resolver import parse_youtube_url

        parsed = parse_youtube_url("https://www.youtube.com/watch?v=abc123")
        name = _derive_brain_name(parsed, "https://www.youtube.com/watch?v=abc123")
        assert "abc123" in name
