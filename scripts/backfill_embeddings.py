"""Backfill chunk embeddings with the current local model (fastembed).

Embeds the stored chunk text (no re-ingest; the only network is fastembed's
one-time model download) so `context` can use dense retrieval. Idempotent: a
chunk is (re)embedded only if it has no embedding, or its embedding was made by a
different model/dimension (e.g. legacy Gemini 768-dim vectors from before the
project went API-free). store_embedding upserts, so re-embedding overwrites.

Usage:
    python scripts/backfill_embeddings.py             # all brains
    python scripts/backfill_embeddings.py <brain_id>  # a single brain
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from youtube_brain.embed import EMBED_DIMS, EMBED_MODEL, embed_texts
from youtube_brain.storage.chunks import store_embedding
from youtube_brain.storage.database import get_session, init_database

BATCH = 256


async def _to_embed(brain_id: str | None) -> list:
    sql = (
        "SELECT c.id, c.text FROM chunks c "
        "LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id "
        "WHERE e.chunk_id IS NULL OR e.model != :model OR e.dimensions != :dims"
    )
    binds: dict = {"model": EMBED_MODEL, "dims": EMBED_DIMS}
    if brain_id:
        sql += " AND c.brain_id = :b"
        binds["b"] = brain_id
    async with get_session() as session:
        return (await session.execute(text(sql), binds)).fetchall()


async def main(brain_id: str | None) -> None:
    await init_database()
    rows = await _to_embed(brain_id)
    if not rows:
        print("Nothing to backfill: all chunks already embedded.")
        return

    print(f"Embedding {len(rows)} chunk(s) with {EMBED_MODEL} ...")
    done = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        vecs = await asyncio.to_thread(embed_texts, [r.text for r in batch])
        for r, vec in zip(batch, vecs):
            await store_embedding(r.id, EMBED_MODEL, EMBED_DIMS, vec)
            done += 1
        print(f"  {done}/{len(rows)}")
    print(f"Done. Embedded {done} chunk(s).")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
