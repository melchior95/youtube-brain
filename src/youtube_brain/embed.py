"""Local text embeddings via fastembed (ONNX, no external API).

Lazy-loads a small ONNX sentence-embedding model on first use (fastembed
downloads it once, then caches it on disk) and reuses it for the process
lifetime. Used for dense retrieval only; consensus and lint stay entity-based
and need no embeddings.

No API key, no network at inference. The only model in the *answering* loop is
still Claude; this just makes keyword-mismatched chunks findable.
"""

from __future__ import annotations

import math
from functools import lru_cache

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 for mismatched dims or a zero vector."""
    if len(a) != len(b):  # mismatched dims: don't silently truncate via zip
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; returns one vector (list[float]) per input."""
    if not texts:
        return []
    return [[float(x) for x in vec] for vec in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
