"""Embedding-based clustering of observations.

Greedy agglomerative clustering: each observation joins the nearest existing
cluster if cosine similarity to its centroid exceeds a threshold, else it
seeds a new cluster. Centroids are running means. Deterministic given input
order, so the same observations always cluster the same way.
"""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):  # mismatched dims (e.g. embedding model changed): no match
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def greedy_cluster(
    items: list[tuple[str, list[float]]], threshold: float = 0.82
) -> dict[str, int]:
    """Cluster (id, vector) pairs. Returns {id: cluster_id}.

    Args:
        items: list of (observation_id, embedding).
        threshold: minimum cosine similarity to join a cluster.
    """
    centroids: list[list[float]] = []
    counts: list[int] = []
    assignments: dict[str, int] = {}

    for oid, vec in items:
        best_i, best_sim = -1, -1.0
        for i, c in enumerate(centroids):
            s = cosine(vec, c)
            if s > best_sim:
                best_sim, best_i = s, i

        if best_i >= 0 and best_sim >= threshold:
            n = counts[best_i]
            centroids[best_i] = [(c * n + v) / (n + 1) for c, v in zip(centroids[best_i], vec)]
            counts[best_i] = n + 1
            assignments[oid] = best_i
        else:
            centroids.append(list(vec))
            counts.append(1)
            assignments[oid] = len(centroids) - 1

    return assignments
