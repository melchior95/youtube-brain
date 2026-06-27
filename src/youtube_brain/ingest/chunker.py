"""Transcript chunking with timestamp-aware windows, overlap, and sentence boundary snapping."""

from __future__ import annotations

import re


def chunk_transcript(
    segments: list[dict],
    window: float = 150.0,
    overlap: float = 30.0,
) -> list[dict]:
    """Split transcript segments into overlapping, timestamp-aware chunks.

    Args:
        segments: List of ``{"start": float, "duration": float, "text": str}``.
        window: Duration of each chunk window in seconds (default 150 s / 2:30).
        overlap: Overlap between consecutive windows in seconds (default 30 s).

    Returns:
        List of ``{"start_time": float, "end_time": float, "text": str}``.
    """
    if not segments:
        return []

    step = window - overlap

    # Determine total duration from the last segment.
    last_seg = segments[-1]
    total_duration = last_seg["start"] + last_seg["duration"]

    chunks: list[dict] = []
    win_start = 0.0

    while win_start < total_duration:
        win_end = win_start + window

        # Collect segments that overlap the current window.
        collected_texts: list[str] = []
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg_start + seg["duration"]
            # A segment overlaps the window if it starts before the window
            # ends AND ends after the window starts.
            if seg_start < win_end and seg_end > win_start:
                text = seg.get("text", "").strip()
                if text:
                    collected_texts.append(text)

        if collected_texts:
            joined = " ".join(collected_texts)
            snapped = _snap_to_sentence_boundary(joined)
            chunks.append(
                {
                    "start_time": win_start,
                    "end_time": win_end,
                    "text": snapped,
                }
            )

        win_start += step

    return chunks


def _snap_to_sentence_boundary(text: str) -> str:
    """Truncate *text* at the last sentence-ending punctuation mark.

    A sentence boundary is defined as a ``"."``, ``"!"``, or ``"?"``
    followed by a space or occurring at the very end of the string.

    If no boundary is found the full text is returned unchanged.
    """
    # Find the last occurrence of sentence-ending punctuation followed by
    # a space or at end-of-string.
    match = None
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        match = m

    if match is None:
        return text

    # Include the punctuation character itself but not the trailing space.
    return text[: match.start() + 1]
