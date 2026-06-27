"""The watchlist loop: new video -> extract -> re-cluster -> changelog.

Turns a brain into a compounding asset. Each refresh detects new videos on the
brain's source, ingests them, extracts observations, re-clusters the whole
store, and reports what changed since last time.
"""

from __future__ import annotations

import logging
import sqlite3

from youtube_brain.config.settings import get_settings
from youtube_brain.ingest.resolver import parse_youtube_url, resolve_video_ids
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.cluster import greedy_cluster
from youtube_brain.observations.extractor import extract_observations
from youtube_brain.storage.brains import get_brain
from youtube_brain.storage.observations import (
    get_observation_embeddings,
    get_observations_by_brain,
    insert_observations,
    set_cluster_ids,
    store_observation_embedding,
)
from youtube_brain.storage.sources import get_sources_by_brain
from youtube_brain.storage.videos import get_videos_by_brain

logger = logging.getLogger(__name__)

CLUSTER_THRESHOLD = 0.74


# ---------------------------------------------------------------------------
# Changelog — pure, testable
# ---------------------------------------------------------------------------


def diff_intelligence(before: dict, after: dict) -> dict:
    """Compare two intelligence payloads into a human-readable changelog."""
    changes: list[dict] = []
    for cat, rows_after in after.get("rollups", {}).items():
        before_map = {
            r["value"]: {e["creator"] for e in r["evidence"]}
            for r in before.get("rollups", {}).get(cat, [])
        }
        for r in rows_after:
            after_creators = {e["creator"] for e in r["evidence"]}
            before_creators = before_map.get(r["value"], set())
            added = after_creators - before_creators
            if added:
                changes.append({
                    "category": cat,
                    "value": r["value"],
                    "before": len(before_creators),
                    "after": len(after_creators),
                    "new_creators": sorted(added),
                    "is_new": r["value"] not in before_map,
                })
    changes.sort(key=lambda c: (-len(c["new_creators"]), not c["is_new"]))
    return {
        "new_observations": after.get("total_observations", 0)
        - before.get("total_observations", 0),
        "new_founders": after.get("founders", 0) - before.get("founders", 0),
        "rollup_changes": changes,
    }


# ---------------------------------------------------------------------------
# Detection & helpers
# ---------------------------------------------------------------------------


async def find_new_videos(brain_id: str):
    """Resolve the brain's source and return (new_video_metas, source)."""
    sources = await get_sources_by_brain(brain_id)
    if not sources:
        return [], None
    source = sources[0]
    parsed = parse_youtube_url(source.source_url)
    metas = resolve_video_ids(parsed)
    existing = {v.video_id for v in await get_videos_by_brain(brain_id)}
    new = [m for m in metas if m.get("video_id") and m["video_id"] not in existing]
    return new, source


def _videos_needing_observations(db_path: str, brain_id: str) -> list[dict]:
    """Videos that have a clean transcript but no observations yet."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT v.id, v.video_id, v.title, v.transcript_clean "
        "FROM videos v WHERE v.brain_id=? AND v.transcript_clean IS NOT NULL "
        "AND v.id NOT IN (SELECT DISTINCT video_id FROM observations WHERE video_id IS NOT NULL)",
        (brain_id,),
    ).fetchall()
    out = []
    for v in rows:
        chunks = con.execute(
            "SELECT id, start_time, text FROM chunks WHERE video_id=? ORDER BY start_time",
            (v["id"],),
        ).fetchall()
        out.append({
            "id": v["id"], "video_id": v["video_id"], "title": v["title"],
            "transcript": v["transcript_clean"],
            "chunks": [{"id": c["id"], "start_time": c["start_time"], "text": c["text"]}
                       for c in chunks],
        })
    con.close()
    return out


async def recluster(brain_id: str, client: GeminiClient) -> int:
    """Embed any observations missing an embedding, then re-cluster all. Returns cluster count."""
    embedded = {oid for oid, _ in await get_observation_embeddings(brain_id)}
    all_obs = await get_observations_by_brain(brain_id)
    missing = [o for o in all_obs if str(o.id) not in embedded]
    if missing:
        vecs = await client.embed_texts([o.claim for o in missing])
        for o, vec in zip(missing, vecs):
            await store_observation_embedding(o.id, client.embed_model, client.embed_dims, vec)

    pairs = await get_observation_embeddings(brain_id)
    assignments = greedy_cluster(pairs, threshold=CLUSTER_THRESHOLD)
    await set_cluster_ids(assignments)
    return len(set(assignments.values()))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def refresh_brain(
    brain_id: str, max_videos: int | None = None, ingest_new: bool = True
) -> dict:
    """Run one watchlist refresh. Returns {new_videos, new_observations, changelog}.

    Steps: snapshot -> ingest new videos -> extract missing observations ->
    re-cluster -> snapshot -> diff.
    """
    from youtube_brain.observations.report import build_intelligence

    settings = get_settings()
    db_path = str(settings.database_path)

    brain = await get_brain(brain_id)
    if brain is None:
        raise ValueError(f"Brain {brain_id} not found")

    before = build_intelligence(brain.name, await get_observations_by_brain(brain_id))

    # 1. Ingest newly-uploaded videos into the existing brain.
    ingested = 0
    if ingest_new:
        new_videos, source = await find_new_videos(brain_id)
        if max_videos is not None:
            new_videos = new_videos[:max_videos]
        if new_videos and source is not None:
            from youtube_brain.ingest.pipeline import ingest_url
            res = await ingest_url(
                source.source_url, existing_brain=brain, existing_source=source,
                video_metas=new_videos,
            )
            ingested = res.videos_processed

    # 2. Extract observations for any videos that lack them.
    needing = _videos_needing_observations(db_path, brain_id)
    new_obs_count = 0
    if needing:
        client = GeminiClient()
        try:
            for v in needing:
                obs = await extract_observations(
                    client, brain_id=brain_id, video_id=v["id"],
                    youtube_id=v["video_id"], creator=v["title"] or v["video_id"],
                    transcript=v["transcript"], chunks=v["chunks"],
                )
                if obs:
                    await insert_observations(obs)
                    new_obs_count += len(obs)
            # 3. Re-cluster the whole store.
            await recluster(brain_id, client)
        finally:
            await client.close()

    after = build_intelligence(brain.name, await get_observations_by_brain(brain_id))
    return {
        "new_videos": ingested,
        "new_observations": new_obs_count,
        "changelog": diff_intelligence(before, after),
    }
