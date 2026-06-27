"""Tests for the transcript fetching module."""

from unittest.mock import MagicMock, patch

import pytest

from youtube_brain.ingest.transcripts import (
    TranscriptResult,
    clean_transcript,
    fetch_transcript,
    _try_youtube_transcript_api,
    _try_yt_dlp,
    _parse_json3,
)


class TestTranscriptResultModel:
    def test_transcript_result_model(self):
        result = TranscriptResult(
            text_with_timestamps=[{"start": 0.0, "duration": 5.0, "text": "Hello world"}],
            full_text="Hello world",
            language="en",
            is_auto_generated=False,
            source="api",
        )
        assert result.full_text == "Hello world"
        assert result.source == "api"
        assert result.language == "en"
        assert result.is_auto_generated is False
        assert len(result.text_with_timestamps) == 1

    def test_transcript_result_defaults(self):
        result = TranscriptResult(
            text_with_timestamps=[],
            full_text="",
        )
        assert result.language is None
        assert result.is_auto_generated is False
        assert result.source == "api"

    def test_transcript_result_multiple_segments(self):
        segments = [
            {"start": 0.0, "duration": 2.0, "text": "First"},
            {"start": 2.0, "duration": 3.0, "text": "Second"},
            {"start": 5.0, "duration": 1.5, "text": "Third"},
        ]
        result = TranscriptResult(
            text_with_timestamps=segments,
            full_text="First Second Third",
            language="en",
            source="yt-dlp",
        )
        assert len(result.text_with_timestamps) == 3
        assert result.source == "yt-dlp"


class TestCleanTranscript:
    def test_clean_transcript_removes_music(self):
        raw = "Hello [Music] world [Applause]  how  are   you"
        cleaned = clean_transcript(raw)
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        assert "  " not in cleaned

    def test_clean_transcript_removes_laughter(self):
        cleaned = clean_transcript("Ha ha [Laughter] funny")
        assert "[Laughter]" not in cleaned
        assert cleaned == "Ha ha funny"

    def test_clean_transcript_case_insensitive(self):
        cleaned = clean_transcript("[music] hello [APPLAUSE] world [laughter]")
        assert cleaned == "hello world"

    def test_clean_transcript_normalizes_whitespace(self):
        cleaned = clean_transcript("  hello   world  ")
        assert cleaned == "hello world"

    def test_clean_transcript_empty_string(self):
        assert clean_transcript("") == ""

    def test_clean_transcript_no_tags(self):
        text = "Normal text without any tags"
        assert clean_transcript(text) == text

    def test_clean_transcript_only_tags(self):
        cleaned = clean_transcript("[Music] [Applause] [Laughter]")
        assert cleaned == ""


class TestParseJson3:
    def test_parse_json3_basic(self):
        data = {
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 2000,
                    "segs": [{"utf8": "Hello world"}],
                },
                {
                    "tStartMs": 3000,
                    "dDurationMs": 1500,
                    "segs": [{"utf8": "Second line"}],
                },
            ]
        }
        segments = _parse_json3(data)
        assert len(segments) == 2
        assert segments[0] == {"start": 1.0, "duration": 2.0, "text": "Hello world"}
        assert segments[1] == {"start": 3.0, "duration": 1.5, "text": "Second line"}

    def test_parse_json3_skips_empty_events(self):
        data = {
            "events": [
                {"tStartMs": 0},  # No segs
                {"tStartMs": 1000, "segs": [{"utf8": ""}]},  # Empty text
                {"tStartMs": 2000, "segs": [{"utf8": "\n"}]},  # Only newline
                {"tStartMs": 3000, "segs": [{"utf8": "Valid"}]},
            ]
        }
        segments = _parse_json3(data)
        assert len(segments) == 1
        assert segments[0]["text"] == "Valid"

    def test_parse_json3_concatenates_segs(self):
        data = {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 5000,
                    "segs": [{"utf8": "Hello "}, {"utf8": "world"}],
                }
            ]
        }
        segments = _parse_json3(data)
        assert len(segments) == 1
        assert segments[0]["text"] == "Hello world"

    def test_parse_json3_empty_events(self):
        assert _parse_json3({"events": []}) == []

    def test_parse_json3_missing_events(self):
        assert _parse_json3({}) == []


class TestTryYoutubeTranscriptApi:
    def test_returns_result_for_manual_transcript(self):
        mock_snippet = MagicMock()
        mock_snippet.text = "Hello world"
        mock_snippet.start = 0.0
        mock_snippet.duration = 5.0

        mock_fetched = MagicMock()
        mock_fetched.to_raw_data.return_value = [
            {"text": "Hello world", "start": 0.0, "duration": 5.0}
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False
        mock_transcript.fetch.return_value = mock_fetched

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript

        mock_api = MagicMock()
        mock_api.list.return_value = mock_transcript_list

        with patch(
            "youtube_brain.ingest.transcripts.YouTubeTranscriptApi",
            return_value=mock_api,
        ):
            result = _try_youtube_transcript_api("test_id")

        assert result is not None
        assert result.full_text == "Hello world"
        assert result.source == "api"
        assert result.is_auto_generated is False
        assert result.language == "en"

    def test_falls_back_to_auto_generated(self):
        mock_fetched = MagicMock()
        mock_fetched.to_raw_data.return_value = [
            {"text": "Auto text", "start": 0.0, "duration": 3.0}
        ]

        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = True
        mock_transcript.fetch.return_value = mock_fetched

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.side_effect = Exception("not found")
        mock_transcript_list.find_generated_transcript.return_value = mock_transcript

        mock_api = MagicMock()
        mock_api.list.return_value = mock_transcript_list

        with patch(
            "youtube_brain.ingest.transcripts.YouTubeTranscriptApi",
            return_value=mock_api,
        ):
            result = _try_youtube_transcript_api("test_id")

        assert result is not None
        assert result.is_auto_generated is True

    def test_returns_none_on_failure(self):
        mock_api = MagicMock()
        mock_api.list.side_effect = Exception("Video not found")

        with patch(
            "youtube_brain.ingest.transcripts.YouTubeTranscriptApi",
            return_value=mock_api,
        ):
            result = _try_youtube_transcript_api("bad_id")

        assert result is None


def _no_strategies():
    """Neutralize .env-configured cookie/proxy strategies for deterministic tests."""
    return (
        patch("youtube_brain.ingest.transcripts._cookies_browser", return_value=None),
        patch("youtube_brain.ingest.transcripts._scrapedo_proxy", return_value=None),
        patch("youtube_brain.ingest.transcripts._cookies_file", return_value=None),
    )


class TestFetchTranscript:
    async def test_fetch_uses_api_first(self):
        expected = TranscriptResult(
            text_with_timestamps=[{"start": 0.0, "duration": 1.0, "text": "Test"}],
            full_text="Test",
            language="en",
            source="api",
        )
        p1, p2, p3 = _no_strategies()
        with p1, p2, p3, patch(
            "youtube_brain.ingest.transcripts._try_youtube_transcript_api",
            return_value=expected,
        ):
            result = await fetch_transcript("test_id")

        assert result is not None
        assert result.source == "api"

    async def test_fetch_falls_back_to_ytdlp(self):
        expected = TranscriptResult(
            text_with_timestamps=[{"start": 0.0, "duration": 1.0, "text": "Fallback"}],
            full_text="Fallback",
            language="en",
            source="yt-dlp",
        )
        p1, p2, p3 = _no_strategies()
        with (
            p1,
            p2,
            p3,
            patch(
                "youtube_brain.ingest.transcripts._try_youtube_transcript_api",
                return_value=None,
            ),
            patch(
                "youtube_brain.ingest.transcripts._try_yt_dlp",
                return_value=expected,
            ),
        ):
            result = await fetch_transcript("test_id")

        assert result is not None
        assert result.source == "yt-dlp"

    async def test_fetch_returns_none_when_both_fail(self):
        p1, p2, p3 = _no_strategies()
        with (
            p1,
            p2,
            p3,
            patch(
                "youtube_brain.ingest.transcripts._try_youtube_transcript_api",
                return_value=None,
            ),
            patch(
                "youtube_brain.ingest.transcripts._try_yt_dlp",
                return_value=None,
            ),
        ):
            result = await fetch_transcript("bad_id")

        assert result is None


@pytest.mark.integration
async def test_fetch_real_transcript():
    result = await fetch_transcript("dQw4w9WgXcQ")
    assert result is not None
    assert len(result.full_text) > 100
