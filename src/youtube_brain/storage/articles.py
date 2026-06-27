"""CRUD for the articles table (generated long-form artifacts)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select

from youtube_brain.storage.database import articles, get_session


async def upsert_latest_article(
    brain_id: str, title: str, body: str, article_type: str
) -> None:
    """Replace the brain's article of this type with a fresh one."""
    async with get_session() as session:
        await session.execute(
            articles.delete().where(
                (articles.c.brain_id == brain_id)
                & (articles.c.article_type == article_type)
            )
        )
        await session.execute(articles.insert().values(
            id=str(uuid4()),
            brain_id=brain_id,
            title=title,
            body=body,
            article_type=article_type,
            source_chunk_ids=None,
            created_at=datetime.now(timezone.utc),
        ))


async def get_latest_article(brain_id: UUID | str, article_type: str) -> dict | None:
    async with get_session() as session:
        stmt = (
            select(articles)
            .where(
                (articles.c.brain_id == str(brain_id))
                & (articles.c.article_type == article_type)
            )
            .order_by(articles.c.created_at.desc())
            .limit(1)
        )
        row = (await session.execute(stmt)).fetchone()
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "body": row.body,
            "article_type": row.article_type,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
