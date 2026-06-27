"""Video summarization via Gemini."""

from __future__ import annotations

import logging

from youtube_brain.llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = """\
You are a video summarization assistant. Given a transcript from a YouTube video, \
produce a JSON object with the following fields:

- video_summary: A concise summary of the video content (100-200 words).
- key_points: A list of the main takeaways or key points discussed.
- businesses_mentioned: A list of any businesses, companies, or brands mentioned.
- people_mentioned: A list of any people mentioned by name.
- main_topics: A list of the primary topics covered.

Return only the JSON object. Do not add commentary outside the JSON.\
"""


async def summarize_video(
    client: GeminiClient,
    transcript_clean: str,
    video_title: str,
    channel_name: str,
) -> dict:
    """Summarize a video transcript using Gemini.

    Truncates the transcript to 8000 characters before sending to avoid
    exceeding token limits.

    Args:
        client: An initialized GeminiClient.
        transcript_clean: Cleaned transcript text.
        video_title: Title of the video for context.
        channel_name: Channel name for context.

    Returns:
        Parsed dict with summary fields, or empty dict on failure.
    """
    truncated = transcript_clean[:8000]

    prompt = (
        f"Video: {video_title}\n"
        f"Channel: {channel_name}\n\n"
        f"Transcript:\n{truncated}"
    )

    try:
        result = await client.generate_json(prompt, system=SUMMARY_SYSTEM)
        if isinstance(result, dict):
            return result
        logger.warning("Summarize returned non-dict result, returning empty dict")
        return {}
    except Exception:
        logger.warning("Video summarization failed", exc_info=True)
        return {}
