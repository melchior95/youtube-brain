"""Reranking and diversity selection for retrieval results."""

from __future__ import annotations


def weighted_score(
    vector_sim: float = 0.0,
    bm25: float = 0.0,
    meta_match: float = 0.0,
    recency: float = 0.0,
    recency_weight: float = 0.1,
) -> float:
    """Compute a weighted combination of retrieval signals.

    Base weights (w_vec=0.4, w_bm25=0.3, w_meta=0.2) are scaled so
    they fill ``1.0 - recency_weight``, leaving room for the recency
    signal.
    """
    w_vec = 0.4
    w_bm25 = 0.3
    w_meta = 0.2

    base_total = w_vec + w_bm25 + w_meta
    scale = (1.0 - recency_weight) / base_total

    return (
        w_vec * scale * vector_sim
        + w_bm25 * scale * bm25
        + w_meta * scale * meta_match
        + recency_weight * recency
    )


def diversity_select(
    chunks: list[dict],
    max_per_video: int = 3,
    max_per_channel: int = 8,
    top_k: int = 20,
) -> list[dict]:
    """Select up to *top_k* chunks while enforcing per-video and per-channel limits.

    Chunks are first sorted by ``score`` descending, then greedily
    picked while respecting the diversity caps.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)

    video_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    selected: list[dict] = []

    for chunk in sorted_chunks:
        if len(selected) >= top_k:
            break

        vid = chunk.get("video_id", "")
        channel = chunk.get("channel_name", "")

        if video_counts.get(vid, 0) >= max_per_video:
            continue
        if channel and channel_counts.get(channel, 0) >= max_per_channel:
            continue

        selected.append(chunk)
        video_counts[vid] = video_counts.get(vid, 0) + 1
        if channel:
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

    return selected
