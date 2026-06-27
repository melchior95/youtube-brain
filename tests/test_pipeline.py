"""Tests for the ingestion pipeline orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    """Test the full pipeline with all external dependencies mocked."""

    @pytest.fixture
    def mock_transcript_result(self):
        """Create a mock TranscriptResult."""
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
            patch("youtube_brain.ingest.pipeline.store_embedding", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_video", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._update_chunk_metadata", new_callable=AsyncMock),
            patch(
                "youtube_brain.ingest.pipeline.label_chunks",
                new_callable=AsyncMock,
                return_value=[{"topics": ["test"], "business_type": "saas"}],
            ),
            patch(
                "youtube_brain.ingest.pipeline.summarize_video",
                new_callable=AsyncMock,
                return_value={
                    "video_summary": "A test video summary.",
                    "key_points": ["point 1"],
                    "businesses_mentioned": [],
                    "people_mentioned": [],
                    "main_topics": ["testing"],
                },
            ),
            patch(
                "youtube_brain.ingest.pipeline.GeminiClient",
            ) as mock_client_cls,
        ):
            # Set up the mock GeminiClient instance
            mock_client = AsyncMock()
            mock_client.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            mock_client.embed_model = "text-embedding-004"
            mock_client.embed_dims = 768
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            # Verify brain was created
            mock_insert_brain.assert_called_once()

            # Verify brain status transitions: INGESTING -> READY
            status_calls = [call.args[1] for call in mock_update_status.call_args_list]
            assert status_calls[0] == BrainStatus.INGESTING
            assert status_calls[-1] == BrainStatus.READY

            # Verify transcript was fetched
            mock_fetch.assert_called_once_with("abc123")

            # Verify result
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
            patch("youtube_brain.ingest.pipeline.GeminiClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            # Transcript should NOT have been fetched since video exists
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
            patch("youtube_brain.ingest.pipeline.GeminiClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await ingest_url("https://www.youtube.com/watch?v=abc123")

            assert result.videos_processed == 0
            assert len(result.errors) == 1
            assert "No transcript" in result.errors[0]

            # Brain should be set to ERROR since 0 videos processed
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
            patch("youtube_brain.ingest.pipeline.GeminiClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

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

        call_count = 0

        async def fetch_side_effect(vid_id):
            nonlocal call_count
            call_count += 1
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
            patch("youtube_brain.ingest.pipeline.store_embedding", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline.update_video", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._insert_source", new_callable=AsyncMock),
            patch("youtube_brain.ingest.pipeline._update_chunk_metadata", new_callable=AsyncMock),
            patch(
                "youtube_brain.ingest.pipeline.resolve_video_ids",
                return_value=[
                    {"video_id": "vid1", "title": "Video 1", "channel_name": "Ch", "duration_seconds": 100},
                    {"video_id": "vid2", "title": "Video 2", "channel_name": "Ch", "duration_seconds": 200},
                ],
            ),
            patch(
                "youtube_brain.ingest.pipeline.label_chunks",
                new_callable=AsyncMock,
                return_value=[{}],
            ),
            patch(
                "youtube_brain.ingest.pipeline.summarize_video",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch("youtube_brain.ingest.pipeline.GeminiClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
            mock_client.embed_model = "text-embedding-004"
            mock_client.embed_dims = 768
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await ingest_url("https://www.youtube.com/playlist?list=PLtest")

            # vid1 should have errored, vid2 should have succeeded
            assert result.videos_found == 2
            assert result.videos_processed == 1
            assert len(result.errors) >= 1
            assert any("vid1" in e for e in result.errors)


# Import here so the test class can reference the enum
from youtube_brain.core.enums import BrainStatus


class TestLabeler:
    """Basic tests for labeler module."""

    async def test_label_chunks_returns_list(self):
        from youtube_brain.ingest.labeler import label_chunks

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(
            return_value=[
                {
                    "business_type": "saas",
                    "advice_category": "marketing",
                    "stage": "growth",
                    "asset_type": "interview",
                    "topics": ["scaling", "growth"],
                }
            ]
        )

        chunks = [{"text": "Some text about SaaS.", "start_time": 0, "end_time": 150}]
        labels = await label_chunks(mock_client, chunks, "Test Video", "Test Channel")

        assert len(labels) == 1
        assert labels[0]["business_type"] == "saas"

    async def test_label_chunks_handles_failure(self):
        from youtube_brain.ingest.labeler import label_chunks

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(side_effect=Exception("API error"))

        chunks = [{"text": "Some text.", "start_time": 0, "end_time": 150}]
        labels = await label_chunks(mock_client, chunks, "Test", "Channel")

        assert len(labels) == 1
        assert labels[0] == {}

    async def test_label_chunks_batches_correctly(self):
        from youtube_brain.ingest.labeler import label_chunks

        call_count = 0

        async def mock_generate(prompt, system=None):
            nonlocal call_count
            call_count += 1
            # Return 10 labels for first batch, 3 for second
            if call_count == 1:
                return [{"topics": [f"topic_{i}"]} for i in range(10)]
            return [{"topics": [f"topic_{i}"]} for i in range(3)]

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(side_effect=mock_generate)

        chunks = [
            {"text": f"Chunk {i}.", "start_time": i * 150, "end_time": (i + 1) * 150}
            for i in range(13)
        ]
        labels = await label_chunks(mock_client, chunks, "Test", "Channel", batch_size=10)

        assert len(labels) == 13
        assert call_count == 2  # 13 chunks / 10 per batch = 2 calls


class TestSummarizer:
    """Basic tests for summarizer module."""

    async def test_summarize_video_returns_dict(self):
        from youtube_brain.ingest.summarizer import summarize_video

        expected = {
            "video_summary": "A great video.",
            "key_points": ["point 1"],
            "businesses_mentioned": ["Acme"],
            "people_mentioned": ["John"],
            "main_topics": ["business"],
        }
        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(return_value=expected)

        result = await summarize_video(mock_client, "transcript text", "Title", "Channel")

        assert result == expected

    async def test_summarize_video_handles_failure(self):
        from youtube_brain.ingest.summarizer import summarize_video

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(side_effect=Exception("API error"))

        result = await summarize_video(mock_client, "transcript text", "Title", "Channel")

        assert result == {}

    async def test_summarize_video_truncates_long_transcript(self):
        from youtube_brain.ingest.summarizer import summarize_video

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(return_value={"video_summary": "Summary"})

        long_transcript = "x" * 20000
        await summarize_video(mock_client, long_transcript, "Title", "Channel")

        # Verify the prompt sent to generate_json has truncated transcript
        call_args = mock_client.generate_json.call_args
        prompt = call_args.args[0]
        # The transcript portion should be at most 8000 chars, not the full 20000
        assert len(prompt) < 10000


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
