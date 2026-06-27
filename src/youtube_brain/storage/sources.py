"""Read access for the sources table."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from youtube_brain.core.enums import SourceStatus, SourceType
from youtube_brain.core.models import Source
from youtube_brain.storage.database import get_session, sources as sources_table


def _row_to_source(row) -> Source:
    return Source(
        id=row.id,
        brain_id=row.brain_id,
        source_type=SourceType(row.source_type),
        source_url=row.source_url,
        source_title=row.source_title,
        source_id=row.source_id,
        status=SourceStatus(row.status),
        created_at=row.created_at,
    )


async def get_sources_by_brain(brain_id: UUID | str) -> list[Source]:
    async with get_session() as session:
        stmt = select(sources_table).where(sources_table.c.brain_id == str(brain_id))
        result = await session.execute(stmt)
        return [_row_to_source(r) for r in result.fetchall()]
