"""YouTube URL parser and video ID resolver."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _parse_published(entry: dict) -> datetime | None:
    """Extract an upload date from a yt-dlp entry (timestamp or upload_date)."""
    ts = entry.get("timestamp")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            pass
    ud = entry.get("upload_date")  # "YYYYMMDD"
    if isinstance(ud, str) and len(ud) == 8 and ud.isdigit():
        try:
            return datetime(int(ud[:4]), int(ud[4:6]), int(ud[6:8]), tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


class UrlParseResult(BaseModel):
    """Result of parsing a YouTube URL."""

    source_type: str  # "video", "playlist", or "channel"
    video_id: str | None = None
    playlist_id: str | None = None
    channel_id: str | None = None
    channel_handle: str | None = None
    original_url: str = ""


def parse_youtube_url(url: str) -> UrlParseResult:
    """Parse a YouTube URL and extract the source type and identifiers.

    Supported URL patterns:
        - youtu.be/ID              -> video
        - /watch?v=ID              -> video
        - /playlist?list=ID        -> playlist
        - /@handle                 -> channel
        - /channel/UCID            -> channel
        - /c/CustomName            -> channel

    Raises:
        ValueError: If the URL is not a recognized YouTube URL pattern.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path

    # Normalize hostname: strip www. prefix
    hostname = hostname.lower().removeprefix("www.")

    # youtu.be short URLs
    if hostname == "youtu.be":
        video_id = path.lstrip("/").split("/")[0].split("?")[0]
        if not video_id:
            raise ValueError(f"Could not extract video ID from short URL: {url}")
        return UrlParseResult(
            source_type="video",
            video_id=video_id,
            original_url=url,
        )

    # Full youtube.com URLs
    if hostname not in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        raise ValueError(f"Not a recognized YouTube URL: {url}")

    # /watch?v=ID
    if path == "/watch" or path.startswith("/watch"):
        qs = parse_qs(parsed.query)
        video_ids = qs.get("v")
        if not video_ids:
            raise ValueError(f"Missing video ID in watch URL: {url}")
        return UrlParseResult(
            source_type="video",
            video_id=video_ids[0],
            original_url=url,
        )

    # /playlist?list=ID
    if path == "/playlist" or path.startswith("/playlist"):
        qs = parse_qs(parsed.query)
        playlist_ids = qs.get("list")
        if not playlist_ids:
            raise ValueError(f"Missing playlist ID in playlist URL: {url}")
        return UrlParseResult(
            source_type="playlist",
            playlist_id=playlist_ids[0],
            original_url=url,
        )

    # /@handle
    if path.startswith("/@"):
        handle = path[2:].strip("/").split("/")[0]
        if not handle:
            raise ValueError(f"Missing channel handle in URL: {url}")
        return UrlParseResult(
            source_type="channel",
            channel_handle=handle,
            original_url=url,
        )

    # /channel/UCID
    if path.startswith("/channel/"):
        channel_id = path[len("/channel/") :].strip("/").split("/")[0]
        if not channel_id:
            raise ValueError(f"Missing channel ID in URL: {url}")
        return UrlParseResult(
            source_type="channel",
            channel_id=channel_id,
            original_url=url,
        )

    # /c/CustomName
    if path.startswith("/c/"):
        custom_name = path[len("/c/") :].strip("/").split("/")[0]
        if not custom_name:
            raise ValueError(f"Missing custom channel name in URL: {url}")
        return UrlParseResult(
            source_type="channel",
            channel_handle=custom_name,
            original_url=url,
        )

    raise ValueError(f"Unrecognized YouTube URL pattern: {url}")


def resolve_video_ids(parse_result: UrlParseResult) -> list[dict]:
    """Resolve a parsed YouTube URL to a list of video metadata dicts.

    For single videos, returns a one-element list immediately.
    For playlists and channels, invokes yt-dlp to enumerate all videos.

    Returns:
        List of dicts with keys: video_id, title, channel_name, duration_seconds.
    """
    if parse_result.source_type == "video":
        if not parse_result.video_id:
            raise ValueError("UrlParseResult has source_type='video' but no video_id")
        return [_resolve_single_video(parse_result.video_id)]

    if parse_result.source_type == "playlist":
        if not parse_result.playlist_id:
            raise ValueError("UrlParseResult has source_type='playlist' but no playlist_id")
        return _resolve_playlist(parse_result.playlist_id)

    if parse_result.source_type == "channel":
        return _resolve_channel(parse_result)

    raise ValueError(f"Unknown source_type: {parse_result.source_type}")


def _fetch_full_metadata(video_id: str) -> dict | None:
    """Run yt-dlp --dump-json for a single video; None if the subprocess fails.

    Unlike --flat-playlist enumeration (used for channels/playlists), this full
    extraction includes view_count, like_count, comment_count, and
    channel_follower_count.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-warnings",
        "--quiet",
        "--skip-download",
        url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=True
        )
        return json.loads(result.stdout.strip().splitlines()[0])
    except Exception as exc:
        logger.warning("Could not fetch metadata for video %s: %s", video_id, exc)
        return None


def _resolve_single_video(video_id: str) -> dict:
    """Fetch metadata for a single video via yt-dlp --dump-json.

    Falls back to a bare dict (id only) if yt-dlp fails, so ingestion can
    still proceed using the transcript alone.

    Returns:
        Dict with keys: video_id, title, channel_name, channel_id,
        duration_seconds, published_at, view_count, like_count,
        comment_count, channel_follower_count, stats_fetched_at.
    """
    entry = _fetch_full_metadata(video_id)
    fetched_at = datetime.now(timezone.utc) if entry else None
    entry = entry or {}
    return {
        "video_id": video_id,
        "title": entry.get("title"),
        "channel_name": entry.get("channel") or entry.get("uploader"),
        "channel_id": entry.get("channel_id"),
        "duration_seconds": entry.get("duration"),
        "published_at": _parse_published(entry),
        "view_count": entry.get("view_count"),
        "like_count": entry.get("like_count"),
        "comment_count": entry.get("comment_count"),
        "channel_follower_count": entry.get("channel_follower_count"),
        "stats_fetched_at": fetched_at,
    }


def fetch_video_stats(video_id: str) -> dict:
    """Best-effort fetch of view/like/comment/subscriber counts for a video.

    --flat-playlist enumeration (used for channel/playlist ingestion) doesn't
    include these fields, so the pipeline calls this per video to backfill them
    from a full yt-dlp metadata fetch. Never raises; failures come back as all
    None (including stats_fetched_at) so ingestion proceeds without stats
    rather than being blocked by them.

    Also used stand-alone to refresh already-stored stats on demand: counts
    are otherwise frozen at whatever moment a video was first captured, so
    they drift out of sync across videos ingested at different times.
    """
    entry = _fetch_full_metadata(video_id)
    fetched_at = datetime.now(timezone.utc) if entry else None
    entry = entry or {}
    return {
        "view_count": entry.get("view_count"),
        "like_count": entry.get("like_count"),
        "comment_count": entry.get("comment_count"),
        "channel_follower_count": entry.get("channel_follower_count"),
        "stats_fetched_at": fetched_at,
    }


def _resolve_playlist(playlist_id: str) -> list[dict]:
    """Use yt-dlp to enumerate all videos in a playlist.

    Args:
        playlist_id: The YouTube playlist ID (e.g. PLrAXtmErZgOe...).

    Returns:
        List of dicts with keys: video_id, title, channel_name, duration_seconds.
    """
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    return _run_ytdlp_flat(url)


def _resolve_channel(parse_result: UrlParseResult) -> list[dict]:
    """Use yt-dlp to enumerate all videos on a channel.

    Handles both @handle and /channel/UCID formats.

    Args:
        parse_result: A UrlParseResult with source_type='channel'.

    Returns:
        List of dicts with keys: video_id, title, channel_name, duration_seconds.
    """
    if parse_result.channel_handle:
        url = f"https://www.youtube.com/@{parse_result.channel_handle}/videos"
    elif parse_result.channel_id:
        url = f"https://www.youtube.com/channel/{parse_result.channel_id}/videos"
    else:
        raise ValueError("Channel parse result has neither handle nor channel_id")
    return _run_ytdlp_flat(url)


def _run_ytdlp_flat(url: str) -> list[dict]:
    """Run yt-dlp with --flat-playlist --dump-json and parse the output.

    Args:
        url: The YouTube URL to enumerate.

    Returns:
        List of dicts with keys: video_id, title, channel_name, duration_seconds.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--quiet",
        url,
    ]

    logger.info("Running yt-dlp: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"yt-dlp failed (exit {exc.returncode}): {exc.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out after 300 seconds")

    videos: list[dict] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping unparseable yt-dlp output line: %s", line[:200])
            continue

        video_id = entry.get("id") or entry.get("url")
        if not video_id:
            continue

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title"),
                "channel_name": entry.get("channel") or entry.get("uploader"),
                "channel_id": entry.get("channel_id"),
                "duration_seconds": entry.get("duration"),
                "published_at": _parse_published(entry),
            }
        )

    return videos
