"""CRUD operations for the chunks and chunk_embeddings tables."""

import json
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.models import Chunk
from youtube_brain.storage.database import chunk_embeddings, chunks, get_session


def _uuid(val: UUID | str) -> str:
    """Convert a UUID or string to a plain string for storage."""
    return str(val)


def _json_or_empty(val) -> list:
    """Safely convert a JSON column value to a list.

    Raw SQL results may return the literal string 'null' instead of
    Python None for JSON columns, so we must handle both cases.
    """
    if val is None or val == "null":
        return []
    if isinstance(val, str):
        parsed = json.loads(val)
        return parsed if parsed is not None else []
    return val


def _row_to_chunk(row) -> Chunk:
    """Convert a database row to a Chunk model."""
    return Chunk(
        id=row.id,
        video_id=row.video_id,
        brain_id=row.brain_id,
        start_time=row.start_time,
        end_time=row.end_time,
        text=row.text,
        topics=_json_or_empty(row.topics),
        business_type=_json_or_empty(row.business_type),
        advice_category=_json_or_empty(row.advice_category),
        stage=_json_or_empty(row.stage),
        asset_type=_json_or_empty(row.asset_type),
        created_at=row.created_at,
    )


async def insert_chunks(chunk_list: list[Chunk]) -> int:
    """Batch insert chunks. Returns the number of rows inserted."""
    if not chunk_list:
        return 0
    async with get_session() as session:
        rows = [
            {
                "id": _uuid(c.id),
                "video_id": _uuid(c.video_id),
                "brain_id": _uuid(c.brain_id),
                "start_time": c.start_time,
                "end_time": c.end_time,
                "text": c.text,
                "topics": c.topics,
                "business_type": c.business_type,
                "advice_category": c.advice_category,
                "stage": c.stage,
                "asset_type": c.asset_type,
                "created_at": c.created_at,
            }
            for c in chunk_list
        ]
        result = await session.execute(chunks.insert(), rows)
        return len(chunk_list)


async def get_chunks_by_video(video_id: UUID | str) -> list[Chunk]:
    """Get all chunks for a video, ordered by start_time."""
    async with get_session() as session:
        stmt = (
            select(chunks)
            .where(chunks.c.video_id == _uuid(video_id))
            .order_by(chunks.c.start_time)
        )
        result = await session.execute(stmt)
        return [_row_to_chunk(row) for row in result.fetchall()]


async def search_fts(query: str, brain_id: str, limit: int = 50) -> list[Chunk]:
    """Full-text search using FTS5 MATCH with BM25 ranking.

    Splits the query into terms and OR-joins them as quoted terms
    for broadest matching.
    """
    terms = query.strip().split()
    if not terms:
        return []
    safe_terms = [t.replace('"', '') for t in terms if t.strip()]
    if not safe_terms:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in safe_terms)

    sql = text(
        "SELECT c.* FROM chunks c "
        "JOIN chunks_fts ON chunks_fts.rowid = c.rowid "
        "WHERE chunks_fts MATCH :query AND c.brain_id = :brain_id "
        "ORDER BY bm25(chunks_fts) LIMIT :limit"
    )

    async with get_session() as session:
        result = await session.execute(
            sql, {"query": fts_query, "brain_id": brain_id, "limit": limit}
        )
        return [_row_to_chunk(row) for row in result.fetchall()]


async def store_embedding(
    chunk_id: UUID | str,
    model: str,
    dims: int,
    embedding: list[float],
) -> None:
    """Upsert an embedding for a chunk, JSON-serializing the vector."""
    async with get_session() as session:
        stmt = (
            insert(chunk_embeddings)
            .values(
                chunk_id=_uuid(chunk_id),
                model=model,
                dimensions=dims,
                embedding=json.dumps(embedding),
                created_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "model": model,
                    "dimensions": dims,
                    "embedding": json.dumps(embedding),
                    "created_at": datetime.now(timezone.utc),
                },
            )
        )
        await session.execute(stmt)
