"""Tests for the YouTube URL resolver."""

import json
from unittest.mock import MagicMock, patch

import pytest

from youtube_brain.ingest.resolver import (
    UrlParseResult,
    parse_youtube_url,
    resolve_video_ids,
    _resolve_playlist,
    _resolve_channel,
)


# ---------------------------------------------------------------------------
# parse_youtube_url — URL parsing tests
# ---------------------------------------------------------------------------


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


def test_parse_custom_channel_url():
    result = parse_youtube_url("https://www.youtube.com/c/MyCustomChannel")
    assert result.source_type == "channel"
    assert result.channel_handle == "MyCustomChannel"


def test_parse_mobile_url():
    result = parse_youtube_url("https://m.youtube.com/watch?v=abc123")
    assert result.source_type == "video"
    assert result.video_id == "abc123"


def test_parse_watch_url_with_extra_params():
    result = parse_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120")
    assert result.source_type == "video"
    assert result.video_id == "dQw4w9WgXcQ"


def test_parse_original_url_preserved():
    url = "https://www.youtube.com/watch?v=abc123"
    result = parse_youtube_url(url)
    assert result.original_url == url


def test_parse_unrecognized_host():
    with pytest.raises(ValueError, match="Not a recognized YouTube URL"):
        parse_youtube_url("https://example.com/watch?v=abc123")


def test_parse_unrecognized_path():
    with pytest.raises(ValueError, match="Unrecognized YouTube URL pattern"):
        parse_youtube_url("https://www.youtube.com/feed/subscriptions")


def test_parse_missing_video_id():
    with pytest.raises(ValueError, match="Missing video ID"):
        parse_youtube_url("https://www.youtube.com/watch?t=120")


def test_parse_missing_playlist_id():
    with pytest.raises(ValueError, match="Missing playlist ID"):
        parse_youtube_url("https://www.youtube.com/playlist?foo=bar")


# ---------------------------------------------------------------------------
# resolve_video_ids — resolution tests
# ---------------------------------------------------------------------------


def test_resolve_single_video():
    pr = UrlParseResult(source_type="video", video_id="abc123", original_url="")
    result = resolve_video_ids(pr)
    assert len(result) == 1
    assert result[0]["video_id"] == "abc123"
    assert result[0]["title"] is None
    assert result[0]["channel_name"] is None
    assert result[0]["duration_seconds"] is None


@patch("youtube_brain.ingest.resolver.subprocess.run")
def test_resolve_playlist(mock_run):
    entries = [
        {"id": "vid1", "title": "Video 1", "channel": "Chan1", "duration": 120},
        {"id": "vid2", "title": "Video 2", "uploader": "Chan2", "duration": 300},
    ]
    mock_run.return_value = MagicMock(
        stdout="\n".join(json.dumps(e) for e in entries),
        returncode=0,
    )

    result = _resolve_playlist("PLtest123")
    assert len(result) == 2
    assert result[0]["video_id"] == "vid1"
    assert result[0]["title"] == "Video 1"
    assert result[0]["channel_name"] == "Chan1"
    assert result[0]["duration_seconds"] == 120
    assert result[1]["channel_name"] == "Chan2"


@patch("youtube_brain.ingest.resolver.subprocess.run")
def test_resolve_channel_handle(mock_run):
    entries = [
        {"id": "v1", "title": "T1", "channel": "C1", "duration": 60},
    ]
    mock_run.return_value = MagicMock(
        stdout=json.dumps(entries[0]),
        returncode=0,
    )

    pr = UrlParseResult(source_type="channel", channel_handle="testchan")
    result = _resolve_channel(pr)
    assert len(result) == 1
    assert result[0]["video_id"] == "v1"

    # Verify yt-dlp was called with the correct URL
    call_args = mock_run.call_args[0][0]
    assert "https://www.youtube.com/@testchan/videos" in " ".join(call_args)


@patch("youtube_brain.ingest.resolver.subprocess.run")
def test_resolve_channel_id(mock_run):
    entries = [
        {"id": "v2", "title": "T2", "channel": "C2", "duration": 90},
    ]
    mock_run.return_value = MagicMock(
        stdout=json.dumps(entries[0]),
        returncode=0,
    )

    pr = UrlParseResult(source_type="channel", channel_id="UC123abc")
    result = _resolve_channel(pr)
    assert len(result) == 1
    assert result[0]["video_id"] == "v2"

    call_args = mock_run.call_args[0][0]
    assert "https://www.youtube.com/channel/UC123abc/videos" in " ".join(call_args)


def test_resolve_video_missing_id():
    pr = UrlParseResult(source_type="video", original_url="")
    with pytest.raises(ValueError, match="no video_id"):
        resolve_video_ids(pr)


def test_resolve_channel_missing_both():
    pr = UrlParseResult(source_type="channel", original_url="")
    with pytest.raises(ValueError, match="neither handle nor channel_id"):
        _resolve_channel(pr)
