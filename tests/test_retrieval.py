"""Tests for the retrieval engine — reranker and search helpers."""

from __future__ import annotations

import pytest

from youtube_brain.retrieval.reranker import diversity_select, weighted_score
from youtube_brain.retrieval.search import _cosine_sim


# ------------------------------------------------------------------
# weighted_score
# ------------------------------------------------------------------


def test_weighted_score():
    score = weighted_score(
        vector_sim=0.9, bm25=0.8, meta_match=0.5, recency=0.3, recency_weight=0.1
    )
    assert 0.0 < score < 1.0


def test_weighted_score_all_zeros():
    assert weighted_score() == 0.0


def test_weighted_score_vector_only():
    score = weighted_score(vector_sim=1.0, recency_weight=0.0)
    # With recency_weight=0, base weights fill 1.0.
    # w_vec=0.4 scaled by 1.0/0.9 * 0.4 = 0.4444...
    assert score > 0.0


def test_weighted_score_high_recency_weight():
    # Recency dominates when its weight is high
    score_high = weighted_score(recency=1.0, recency_weight=0.9)
    score_low = weighted_score(recency=1.0, recency_weight=0.1)
    assert score_high > score_low


# ------------------------------------------------------------------
# diversity_select
# ------------------------------------------------------------------


def test_diversity_select_limits_per_video():
    chunks = [
        {"chunk_id": f"c{i}", "video_id": "v1", "score": 1.0 - i * 0.01}
        for i in range(10)
    ]
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=8, top_k=20)
    assert len(selected) == 3


def test_diversity_select_mixed_videos():
    chunks = []
    for v in range(5):
        for c in range(4):
            chunks.append(
                {
                    "chunk_id": f"v{v}_c{c}",
                    "video_id": f"v{v}",
                    "channel_name": "ch1",
                    "score": 1.0 - v * 0.1 - c * 0.01,
                }
            )
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=8, top_k=10)
    assert len(selected) == 8  # channel cap of 8 hit before top_k of 10
    video_counts: dict[str, int] = {}
    for s in selected:
        vid = s["video_id"]
        video_counts[vid] = video_counts.get(vid, 0) + 1
    assert all(count <= 3 for count in video_counts.values())


def test_diversity_select_no_channel_name():
    """Chunks without channel_name should not be subject to channel limits."""
    chunks = [
        {"chunk_id": f"c{i}", "video_id": f"v{i}", "score": 1.0 - i * 0.01}
        for i in range(15)
    ]
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=2, top_k=10)
    assert len(selected) == 10  # channel limit skipped (no channel_name)


def test_diversity_select_respects_top_k():
    chunks = [
        {"chunk_id": f"c{i}", "video_id": f"v{i}", "score": 1.0 - i * 0.01}
        for i in range(50)
    ]
    selected = diversity_select(chunks, max_per_video=3, max_per_channel=8, top_k=5)
    assert len(selected) == 5


def test_diversity_select_empty():
    assert diversity_select([], top_k=10) == []


# ------------------------------------------------------------------
# _cosine_sim
# ------------------------------------------------------------------


def test_cosine_sim_identical():
    assert _cosine_sim([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    assert _cosine_sim([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_sim_zero_vector():
    assert _cosine_sim([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_cosine_sim_opposite():
    assert _cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_sim_arbitrary():
    # [3,4] dot [4,3] = 24, |[3,4]|=5, |[4,3]|=5 => 24/25 = 0.96
    assert _cosine_sim([3, 4], [4, 3]) == pytest.approx(0.96)
