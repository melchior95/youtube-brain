"""Lexical retrieval over chunks: FTS5/BM25 with diversity selection.

Keyword retrieval only, no embeddings and no external API. Claude reads the
returned chunks (or a full transcript via the bridge's `pull`) and does the
semantic work; it can also re-issue searches with better terms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from sqlalchemy import text

from youtube_brain.retrieval.reranker import diversity_select, weighted_score
from youtube_brain.storage.database import get_session

logger = logging.getLogger(__name__)


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
# Lane 1: FTS5 chunk search
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
        try:
            result = await session.execute(
                sql, {"query": fts_query, "brain_id": brain_id, "limit": limit}
            )
            rows = result.fetchall()
        except Exception as exc:  # malformed FTS query
            logger.warning("FTS chunk search failed: %s", exc)
            return []

    chunks = []
    for row in rows:
        row_dict = dict(row._mapping)
        # BM25 scores from SQLite are negative (lower = better); normalise
        raw_bm25 = row_dict.pop("bm25_score", 0.0)
        row_dict["bm25_score"] = -raw_bm25 if raw_bm25 else 0.0
        chunks.append(row_dict)
    return chunks


# ---------------------------------------------------------------------------
# Lane 2: FTS5 video-summary search
# ---------------------------------------------------------------------------


async def _fts_summary_search(
    original: str, expanded: list[str], brain_id: str, limit: int = 10
) -> list[dict]:
    """LIKE search on video_summary and title to find relevant videos."""
    all_terms = list({original} | set(expanded))

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
# Main retrieval pipeline
# ---------------------------------------------------------------------------


async def retrieve(
    query: str,
    brain_id: str,
    recency_weight: float = 0.1,
    top_k: int = 20,
) -> RetrievalResult:
    """Lexical retrieval with a summary boost and diversity selection.

    Steps: FTS5 chunk search + FTS5 summary search (for a per-video boost), merge,
    normalise BM25, score, then diversity_select. No embeddings, no LLM.
    """
    terms = [t for t in query.split() if t.strip()]

    fts_chunks = await _fts_search(query, terms, brain_id, limit=50)
    summary_hits = await _fts_summary_search(query, terms, brain_id, limit=10)

    summary_video_ids: set[str] = set()
    for sv in summary_hits:
        vid = sv.get("id") or sv.get("video_id")
        if vid:
            summary_video_ids.add(str(vid))

    merged: dict[str, dict] = {}
    for chunk in fts_chunks:
        cid = str(chunk.get("id", chunk.get("chunk_id", "")))
        if cid not in merged:
            merged[cid] = {**chunk, "chunk_id": cid}
        merged[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)

    chunks_searched = len(merged)

    bm25_vals = [c.get("bm25_score", 0.0) for c in merged.values()]
    bm25_max = max(bm25_vals) if bm25_vals else 1.0
    if bm25_max <= 0:
        bm25_max = 1.0

    for chunk in merged.values():
        bm25_norm = chunk.get("bm25_score", 0.0) / bm25_max if bm25_max else 0.0
        video_id = str(chunk.get("video_id", ""))
        summary_boost = 0.1 if video_id in summary_video_ids else 0.0
        chunk["score"] = weighted_score(
            vector_sim=0.0,
            bm25=bm25_norm,
            meta_match=0.0,
            recency=0.0,
            recency_weight=recency_weight,
        ) + summary_boost

    selected = diversity_select(
        list(merged.values()), max_per_video=3, max_per_channel=8, top_k=top_k
    )

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
        expanded_query=query,
    )
