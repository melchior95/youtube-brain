"""Tests for the observation clustering engine."""

from youtube_brain.observations.cluster import cosine, greedy_cluster


def test_cosine_basic():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 0], [0, 0]) == 0.0


def test_greedy_cluster_groups_similar():
    # Two tight groups plus an outlier.
    items = [
        ("a", [1.0, 0.0]),
        ("b", [0.99, 0.01]),
        ("c", [0.0, 1.0]),
        ("d", [0.02, 0.98]),
        ("e", [0.7, 0.7]),  # outlier between the two
    ]
    assignments = greedy_cluster(items, threshold=0.95)
    assert assignments["a"] == assignments["b"]
    assert assignments["c"] == assignments["d"]
    assert assignments["a"] != assignments["c"]


def test_greedy_cluster_singletons_when_threshold_high():
    items = [("a", [1.0, 0.0]), ("b", [0.0, 1.0])]
    assignments = greedy_cluster(items, threshold=0.99)
    assert assignments["a"] != assignments["b"]
    assert len(set(assignments.values())) == 2
