"""CRUD for the observations and observation_embeddings tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.dialects.sqlite import insert

from youtube_brain.core.models import Observation
from youtube_brain.storage.database import (
    get_session,
    observation_embeddings,
    observations,
)


def _uuid(val: UUID | str) -> str:
    return str(val)


def _row_to_observation(row) -> Observation:
    entities = row.entities
    if isinstance(entities, str):
        entities = json.loads(entities) if entities and entities != "null" else []
    return Observation(
        id=row.id,
        brain_id=row.brain_id,
        video_id=row.video_id,
        youtube_id=row.youtube_id,
        creator=row.creator,
        obs_type=row.obs_type,
        claim=row.claim,
        value=row.value,
        entities=entities or [],
        evidence_quote=row.evidence_quote,
        chunk_id=row.chunk_id,
        start_time=row.start_time,
        confidence=row.confidence,
        domain=row.domain,
        cluster_id=row.cluster_id,
        created_at=row.created_at,
    )


async def insert_observations(obs_list: list[Observation]) -> int:
    """Batch insert observations. Returns number inserted."""
    if not obs_list:
        return 0
    async with get_session() as session:
        rows = [
            {
                "id": _uuid(o.id),
                "brain_id": _uuid(o.brain_id),
                "video_id": _uuid(o.video_id) if o.video_id else None,
                "youtube_id": o.youtube_id,
                "creator": o.creator,
                "obs_type": o.obs_type,
                "claim": o.claim,
                "value": o.value,
                "entities": o.entities,
                "evidence_quote": o.evidence_quote,
                "chunk_id": o.chunk_id,
                "start_time": o.start_time,
                "confidence": o.confidence,
                "domain": o.domain,
                "cluster_id": o.cluster_id,
                "created_at": o.created_at,
            }
            for o in obs_list
        ]
        await session.execute(observations.insert(), rows)
        return len(obs_list)


async def get_observations_by_brain(brain_id: UUID | str) -> list[Observation]:
    async with get_session() as session:
        stmt = select(observations).where(observations.c.brain_id == _uuid(brain_id))
        result = await session.execute(stmt)
        return [_row_to_observation(r) for r in result.fetchall()]


async def delete_observations_by_brain(brain_id: UUID | str) -> None:
    """Clear a brain's observations (and their embeddings) for a clean re-run."""
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM observation_embeddings WHERE observation_id IN "
                 "(SELECT id FROM observations WHERE brain_id=:b)"),
            {"b": _uuid(brain_id)},
        )
        await session.execute(
            text("DELETE FROM observations WHERE brain_id=:b"), {"b": _uuid(brain_id)}
        )


async def store_observation_embedding(
    observation_id: UUID | str, model: str, dims: int, embedding: list[float]
) -> None:
    async with get_session() as session:
        stmt = insert(observation_embeddings).values(
            observation_id=_uuid(observation_id),
            model=model,
            dimensions=dims,
            embedding=json.dumps(embedding),
            created_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(
            index_elements=["observation_id"],
            set_={"model": model, "dimensions": dims,
                  "embedding": json.dumps(embedding),
                  "created_at": datetime.now(timezone.utc)},
        )
        await session.execute(stmt)


async def get_observation_embeddings(brain_id: UUID | str) -> list[tuple[str, list[float]]]:
    """Return (observation_id, embedding) for all of a brain's observations."""
    async with get_session() as session:
        stmt = text(
            "SELECT oe.observation_id oid, oe.embedding emb FROM observation_embeddings oe "
            "JOIN observations o ON o.id = oe.observation_id WHERE o.brain_id = :b"
        )
        result = await session.execute(stmt, {"b": _uuid(brain_id)})
        return [(r.oid, json.loads(r.emb)) for r in result.fetchall()]


async def set_cluster_ids(assignments: dict[str, int]) -> None:
    """Persist cluster_id for each observation id."""
    async with get_session() as session:
        for obs_id, cid in assignments.items():
            await session.execute(
                update(observations).where(observations.c.id == obs_id).values(cluster_id=cid)
            )
