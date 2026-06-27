"""Four-lane hybrid search with query expansion, merging, and diversity selection."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

from sqlalchemy import text

from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.retrieval.reranker import diversity_select, weighted_score
from youtube_brain.storage.database import get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single ranked retrieval result."""

    chunk_id: str
    video_id: str
    youtube_id: str
    video_title: str
    channel_name: str
    start_time: float
    end_time: float
    text: str
    score: float
    caption_kind: str
    topics: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Container for a complete retrieval response."""

    results: list[SearchResult]
    chunks_searched: int
    expanded_query: str


# ---------------------------------------------------------------------------
# Query-expansion system prompt
# ---------------------------------------------------------------------------

EXPAND_SYSTEM = """\
You are a search query expander for a YouTube knowledge base.

Given a user question, produce a JSON object with three keys:

1. "search_query" — a rephrased, self-contained search query optimised for
   semantic vector search.
2. "fts_keywords" — a list of 3-8 keywords/phrases for full-text search
   (BM25).  Include synonyms, acronyms, and related terms.
3. "metadata_filters" — a dict whose keys come ONLY from the controlled
   taxonomy below.  Values are lists of matching enum values.

Controlled taxonomy keys and allowed values:
  business_type: saas, ecommerce, agency, marketplace, content,
                 physical_product, service, mobile_app, other
  advice_category: marketing, distribution, pricing, hiring, fundraising,
                   product, operations, customer_acquisition, retention,
                   monetization, launch, growth, technical, legal, other
  stage: idea, pre_launch, early_stage, growth, scaling, mature, exit, other
  asset_type: interview, tutorial, review, commentary, case_study,
              earnings_call, lecture, panel, other

Return ONLY valid JSON — no markdown fences, no commentary.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 for zero-vectors."""
    if len(a) != len(b):  # mismatched dims: don't silently truncate via zip
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_meta_match(chunk: dict, filters: dict) -> float:
    """Fraction of filter categories where the chunk has a matching value."""
    if not filters:
        return 0.0

    matched = 0
    total = 0

    for key, filter_values in filters.items():
        if not filter_values:
            continue
        total += 1
        chunk_values = chunk.get(key) or []
        if isinstance(chunk_values, str):
            chunk_values = [chunk_values]
        if any(v in filter_values for v in chunk_values):
            matched += 1

    return matched / total if total else 0.0


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


async def _expand_query(client: GeminiClient, query: str) -> dict:
    """Use Gemini to expand the user query.  Falls back to raw query on error."""
    try:
        result = await client.generate_json(query, system=EXPAND_SYSTEM, temperature=0.2)
        if isinstance(result, dict):
            return {
                "search_query": result.get("search_query", query),
                "fts_keywords": result.get("fts_keywords", query.split()),
                "metadata_filters": result.get("metadata_filters", {}),
            }
    except Exception:
        logger.warning("Query expansion failed, using raw query", exc_info=True)

    return {
        "search_query": query,
        "fts_keywords": query.split(),
        "metadata_filters": {},
    }


# ---------------------------------------------------------------------------
# Lane 1 — FTS5 chunk search
# ---------------------------------------------------------------------------


async def _fts_search(
    original: str, expanded: list[str], brain_id: str, limit: int = 50
) -> list[dict]:
    """Run FTS5 MATCH over chunks using original + expanded keywords."""
    all_terms = list({original} | set(expanded))
    safe_terms = [t.replace('"', '') for t in all_terms if t.strip()]
    fts_query = " OR ".join(f'"{t}"' for t in safe_terms)
    if not fts_query:
        return []

    sql = text(
        "SELECT c.*, v.video_id AS youtube_id, v.title AS video_title, "
        "v.channel_name, v.caption_kind, "
        "bm25(chunks_fts) AS bm25_score "
        "FROM chunks c "
        "JOIN chunks_fts ON chunks_fts.rowid = c.rowid "
        "JOIN videos v ON v.id = c.video_id "
        "WHERE chunks_fts MATCH :query AND c.brain_id = :brain_id "
        "ORDER BY bm25(chunks_fts) LIMIT :limit"
    )

    async with get_session() as session:
        result = await session.execute(
            sql, {"query": fts_query, "brain_id": brain_id, "limit": limit}
        )
        rows = result.fetchall()

    chunks = []
    for row in rows:
        row_dict = dict(row._mapping)
        # BM25 scores from SQLite are negative (lower = better); normalise
        raw_bm25 = row_dict.pop("bm25_score", 0.0)
        row_dict["bm25_score"] = -raw_bm25 if raw_bm25 else 0.0
        chunks.append(row_dict)
    return chunks


# ---------------------------------------------------------------------------
# Lane 2 — Vector chunk search
# ---------------------------------------------------------------------------


async def _vector_search(
    query: str, client: GeminiClient, brain_id: str, limit: int = 50
) -> list[dict]:
    """Embed the query and compute cosine similarity against all brain chunk embeddings.

    MVP approach: loads all embeddings into Python.  Will be replaced by
    sqlite-vec knn query later.
    """
    query_vec = (await client.embed_texts([query]))[0]

    sql = text(
        "SELECT ce.chunk_id, ce.embedding, c.*, "
        "v.video_id AS youtube_id, v.title AS video_title, "
        "v.channel_name, v.caption_kind "
        "FROM chunk_embeddings ce "
        "JOIN chunks c ON c.id = ce.chunk_id "
        "JOIN videos v ON v.id = c.video_id "
        "WHERE c.brain_id = :brain_id"
    )

    async with get_session() as session:
        result = await session.execute(sql, {"brain_id": brain_id})
        rows = result.fetchall()

    scored: list[dict] = []
    for row in rows:
        row_dict = dict(row._mapping)
        emb_raw = row_dict.pop("embedding")
        emb = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
        sim = _cosine_sim(query_vec, emb)
        row_dict["vector_score"] = sim
        scored.append(row_dict)

    scored.sort(key=lambda d: d["vector_score"], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Lane 3 — FTS5 video-summary search
# ---------------------------------------------------------------------------


async def _fts_summary_search(
    original: str, expanded: list[str], brain_id: str, limit: int = 10
) -> list[dict]:
    """LIKE search on video_summary and title to find relevant videos."""
    all_terms = list({original} | set(expanded))

    # Build OR-joined LIKE clauses
    conditions = []
    params: dict = {"brain_id": brain_id, "limit": limit}
    for i, term in enumerate(all_terms):
        if not term.strip():
            continue
        pkey = f"t{i}"
        conditions.append(
            f"(v.video_summary LIKE :like_{pkey} OR v.title LIKE :like_{pkey})"
        )
        params[f"like_{pkey}"] = f"%{term}%"

    if not conditions:
        return []

    where = " OR ".join(conditions)
    sql = text(
        f"SELECT v.id, v.video_id, v.title, v.channel_name, v.video_summary "
        f"FROM videos v "
        f"WHERE v.brain_id = :brain_id AND ({where}) "
        f"LIMIT :limit"
    )

    async with get_session() as session:
        result = await session.execute(sql, params)
        rows = result.fetchall()

    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------------------------
# Lane 4 — Vector video-summary search (placeholder)
# ---------------------------------------------------------------------------


async def _vector_summary_search(
    query: str, client: GeminiClient, brain_id: str, limit: int = 10
) -> list[dict]:
    """Placeholder — will embed video summaries and search later."""
    return []


# ---------------------------------------------------------------------------
# Main retrieval pipeline
# ---------------------------------------------------------------------------


async def retrieve(
    query: str,
    brain_id: str,
    client: GeminiClient,
    recency_weight: float = 0.1,
    top_k: int = 20,
) -> RetrievalResult:
    """Four-lane hybrid search with diversity selection and reranking.

    Steps:
        1. Expand the query via Gemini (original + expanded).
        2. Run four retrieval lanes in parallel.
        3. Merge and deduplicate by chunk_id, combining scores.
        4. Boost chunks from videos that matched summary search.
        5. Compute final weighted_score per chunk.
        6. Apply diversity_select.
        7. Build and return SearchResult list.
    """
    # 1. Query expansion
    expansion = await _expand_query(client, query)
    expanded_query = expansion["search_query"]
    fts_keywords = expansion["fts_keywords"]
    meta_filters = expansion["metadata_filters"]

    # 2. Four-lane retrieval
    fts_chunks = await _fts_search(query, fts_keywords, brain_id, limit=50)
    vec_chunks = await _vector_search(expanded_query, client, brain_id, limit=50)
    summary_hits = await _fts_summary_search(query, fts_keywords, brain_id, limit=10)
    _vec_summary = await _vector_summary_search(query, client, brain_id, limit=10)

    # Collect summary video IDs for boosting
    summary_video_ids: set[str] = set()
    for sv in summary_hits:
        vid = sv.get("id") or sv.get("video_id")
        if vid:
            summary_video_ids.add(str(vid))

    # 3. Merge + dedupe by chunk_id
    merged: dict[str, dict] = {}

    for chunk in fts_chunks:
        cid = str(chunk.get("id", chunk.get("chunk_id", "")))
        if cid not in merged:
            merged[cid] = {**chunk, "chunk_id": cid}
        merged[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)

    for chunk in vec_chunks:
        cid = str(chunk.get("id", chunk.get("chunk_id", "")))
        if cid not in merged:
            merged[cid] = {**chunk, "chunk_id": cid}
        merged[cid]["vector_score"] = chunk.get("vector_score", 0.0)

    chunks_searched = len(merged)

    # 4. Normalise scores within their lanes
    bm25_vals = [c.get("bm25_score", 0.0) for c in merged.values()]
    vec_vals = [c.get("vector_score", 0.0) for c in merged.values()]
    bm25_max = max(bm25_vals) if bm25_vals else 1.0
    vec_max = max(vec_vals) if vec_vals else 1.0
    # Guard against a non-positive max (e.g. all-negative cosine sims), which
    # would flip the ranking when used as a divisor.
    if bm25_max <= 0:
        bm25_max = 1.0
    if vec_max <= 0:
        vec_max = 1.0

    for chunk in merged.values():
        bm25_norm = chunk.get("bm25_score", 0.0) / bm25_max if bm25_max else 0.0
        vec_norm = chunk.get("vector_score", 0.0) / vec_max if vec_max else 0.0

        meta_match = _compute_meta_match(chunk, meta_filters)

        # 4b. Summary boost — if this chunk's video appeared in summary hits
        video_id = str(chunk.get("video_id", ""))
        summary_boost = 0.1 if video_id in summary_video_ids else 0.0

        # 5. Weighted score (recency placeholder = 0)
        score = weighted_score(
            vector_sim=vec_norm,
            bm25=bm25_norm,
            meta_match=meta_match,
            recency=0.0,
            recency_weight=recency_weight,
        ) + summary_boost

        chunk["score"] = score

    # 6. Diversity select
    # Resolve channel_name for chunks that don't have it
    all_chunks = list(merged.values())
    selected = diversity_select(all_chunks, max_per_video=3, max_per_channel=8, top_k=top_k)

    # 7. Build SearchResult list
    results: list[SearchResult] = []
    for chunk in selected:
        topics_raw = chunk.get("topics")
        if isinstance(topics_raw, str):
            try:
                topics_raw = json.loads(topics_raw)
            except (json.JSONDecodeError, TypeError):
                topics_raw = []
        if topics_raw is None:
            topics_raw = []

        results.append(
            SearchResult(
                chunk_id=str(chunk.get("chunk_id", chunk.get("id", ""))),
                video_id=str(chunk.get("video_id", "")),
                youtube_id=str(chunk.get("youtube_id", "")),
                video_title=chunk.get("video_title", chunk.get("title", "")) or "",
                channel_name=chunk.get("channel_name", "") or "",
                start_time=float(chunk.get("start_time", 0)),
                end_time=float(chunk.get("end_time", 0)),
                text=chunk.get("text", ""),
                score=chunk.get("score", 0.0),
                caption_kind=chunk.get("caption_kind", "") or "",
                topics=topics_raw,
            )
        )

    return RetrievalResult(
        results=results,
        chunks_searched=chunks_searched,
        expanded_query=expanded_query,
    )
