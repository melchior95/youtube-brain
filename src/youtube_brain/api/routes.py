"""API route definitions for YouTube Brain."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from youtube_brain.categories import brains_by_channel_id, get_category, load_categories
from youtube_brain.generation.answer import generate_answer
from youtube_brain.ingest.resolver import parse_youtube_url
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.crosscreator import cross_creator_intelligence
from youtube_brain.observations.report import build_intelligence
from youtube_brain.storage.brains import get_brain, list_brains
from youtube_brain.storage.database import init_database
from youtube_brain.storage.observations import get_observations_by_brain
from youtube_brain.storage.videos import get_published_dates_by_brain, get_videos_by_brain

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for the ingest endpoint."""

    url: str
    name: str | None = None


from typing import Literal


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""

    query: str
    mode: Literal["qa", "article", "playbook", "summary", "faq"] = "qa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brain_to_dict(brain) -> dict:
    """Convert a Brain model to a JSON-serialisable dict."""
    return {
        "id": str(brain.id),
        "name": brain.name,
        "status": brain.status.value if hasattr(brain.status, "value") else brain.status,
        "video_count": brain.video_count,
        "visibility": brain.visibility,
        "created_at": brain.created_at.isoformat() if brain.created_at else None,
    }


def _video_to_dict(video) -> dict:
    """Convert a Video model to a JSON-serialisable dict."""
    return {
        "id": str(video.id),
        "video_id": video.video_id,
        "title": video.title,
        "channel_name": video.channel_name,
        "url": video.url,
        "status": video.status.value if hasattr(video.status, "value") else video.status,
        "duration_seconds": video.duration_seconds,
        "video_summary": video.video_summary,
        "created_at": video.created_at.isoformat() if video.created_at else None,
    }


# ---------------------------------------------------------------------------
# Background task wrapper
# ---------------------------------------------------------------------------


async def _run_ingest(url: str, name: str | None) -> None:
    """Background task: channel-aware, zero-generate pull (shared with the skill bridge).

    ``name`` is intentionally ignored — the brain is named from the resolved
    channel so a web pull merges with bridge pulls of the same creator by
    channel_id (closing the category-page "pending forever" loop).
    """
    try:
        await init_database()
        from youtube_brain.ingest.pull import pull_creator
        await pull_creator(url)
    except Exception:
        logger.exception("Background ingestion failed for %s", url)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/api/brains")
async def api_list_brains():
    """List all brains."""
    brains = await list_brains()
    return [_brain_to_dict(b) for b in brains]


@router.get("/api/brains/{brain_id}")
async def api_get_brain(brain_id: str):
    """Get a single brain with its videos."""
    brain = await get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")
    videos = await get_videos_by_brain(brain_id)
    # A brain is "multi-creator" when its observations span >1 distinct creator
    # (e.g. Starter Story = many interviewed founders). For a single-person
    # channel the founders/consensus intelligence view is meaningless, so the
    # frontend hides it.
    observations = await get_observations_by_brain(brain_id)
    creators = {o.creator for o in observations if o.creator}
    result = _brain_to_dict(brain)
    result["videos"] = [_video_to_dict(v) for v in videos]
    result["multi_creator"] = len(creators) > 1
    return result


@router.post("/api/brains/ingest")
async def api_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Start ingesting a YouTube URL as a background task."""
    try:
        parse_youtube_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await init_database()
    background_tasks.add_task(_run_ingest, req.url, req.name)
    return {"status": "ingesting", "url": req.url}


@router.get("/api/brains/{brain_id}/intelligence")
async def api_intelligence(brain_id: str):
    """Structured intelligence: consensus themes + entity rollups + coverage."""
    brain = await get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")
    observations = await get_observations_by_brain(brain_id)
    return build_intelligence(brain.name, observations)


@router.get("/api/categories")
async def api_categories():
    """List categories with pulled/total creator counts."""
    cats = load_categories()
    cids = [cr.channel_id for c in cats for cr in c.creators if cr.channel_id]
    resolved = await brains_by_channel_id(cids)
    return [
        {
            "slug": c.slug, "name": c.name, "description": c.description,
            "creator_count": len(c.creators),
            "pulled_count": sum(1 for cr in c.creators
                                if cr.channel_id and cr.channel_id in resolved),
        }
        for c in cats
    ]


@router.get("/api/categories/{slug}")
async def api_category(slug: str):
    """Category detail: creators annotated with their pulled brain (if any)."""
    cat = get_category(slug)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    resolved = await brains_by_channel_id([cr.channel_id for cr in cat.creators if cr.channel_id])
    creators = []
    for cr in cat.creators:
        b = resolved.get(cr.channel_id) if cr.channel_id else None
        if b:
            creators.append({"handle": cr.handle, "pulled": True, **b})
        else:
            creators.append({"handle": cr.handle, "url": cr.url,
                             "channel_id": cr.channel_id, "pulled": False})
    return {"slug": cat.slug, "name": cat.name, "description": cat.description,
            "creators": creators}


@router.get("/api/categories/{slug}/consensus")
async def api_category_consensus(slug: str):
    """Cross-creator intelligence across the category's pulled brains."""
    cat = get_category(slug)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    resolved = await brains_by_channel_id([cr.channel_id for cr in cat.creators if cr.channel_id])
    brain_ids = [b["brain_id"] for b in resolved.values()]
    if not brain_ids:
        return build_intelligence(cat.name, [])
    client = GeminiClient()
    try:
        return await cross_creator_intelligence(cat.name, brain_ids, client)
    finally:
        await client.close()


@router.get("/api/brains/{brain_id}/editorial")
async def api_editorial(brain_id: str):
    """The latest editorial (magazine-style) report, if one has been generated."""
    from youtube_brain.storage.articles import get_latest_article

    brain = await get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")
    article = await get_latest_article(brain_id, "editorial")
    return article or {"title": None, "body": None}


@router.get("/api/brains/{brain_id}/timeline")
async def api_timeline(brain_id: str, granularity: str = "month"):
    """Theme timeline: cumulative distinct-founder counts per entity over time."""
    from youtube_brain.observations.timeline import build_timeline

    brain = await get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")
    observations = await get_observations_by_brain(brain_id)
    published = await get_published_dates_by_brain(brain_id)
    if granularity not in ("week", "month", "quarter"):
        granularity = "month"
    return build_timeline(observations, published, granularity)


@router.post("/api/brains/{brain_id}/ask")
async def api_ask(brain_id: str, req: AskRequest):
    """Ask a question against a brain's knowledge base."""
    brain = await get_brain(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    client = GeminiClient()
    try:
        result = await generate_answer(
            query=req.query,
            brain_id=str(brain.id),
            brain_name=brain.name,
            client=client,
            mode=req.mode,
            recency_weight=brain.recency_weight,
        )
        return {
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "confidence": result.confidence,
            "chunks_searched": result.chunks_searched,
            "chunks_used": result.chunks_used,
            "mode": result.mode,
        }
    finally:
        await client.close()
