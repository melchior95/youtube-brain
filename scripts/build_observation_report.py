"""Observation V1 pipeline: persist -> embed -> cluster -> report.

Loads the extracted observations (data/observations_all.json), stores them in
the observations table, embeds each claim, clusters by embedding, persists the
cluster ids, and writes an Intelligence Report to data/intelligence_report.md.

Run: python scripts/build_observation_report.py
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

from youtube_brain.config.settings import get_settings
from youtube_brain.core.models import Observation
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.cluster import greedy_cluster
from youtube_brain.observations.report import build_report
from youtube_brain.storage.database import init_database
from youtube_brain.storage.observations import (
    delete_observations_by_brain,
    get_observation_embeddings,
    get_observations_by_brain,
    insert_observations,
    set_cluster_ids,
    store_observation_embedding,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BRAIN = "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
BRAIN_NAME = "Starter Story"
IN = "data/observations_all.json"
OUT = "data/intelligence_report.md"

CREATORS = {
    "P4QodeA_lQ0": "Benji (Snag)",
    "Nnpz1wsTjBI": "David (Shipyard)",
    "vbEKEWtnndU": "Cedric (Peptide AI)",
    "BHhg-l9AZpM": "Bo (Savvy Nomad)",
    "32vqaJa90kw": "Nicole (Glamour)",
    "r4R_Hlw7sbo": "Brian (Once)",
    "D4fkiQfzw_I": "Mark (solopreneur)",
    "iVy5J7iE-3Q": "Jeremy (Task Magic)",
    "bq3-qH-CpYQ": "Jordan (Parakeet Chat)",
    "PIXXEAfo6MY": "Ethan (Cut Coach)",
}


def migrate(db_path: str) -> None:
    """Drop the legacy observations tables so create_all rebuilds the V1 schema."""
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS observation_embeddings")
    con.execute("DROP TABLE IF EXISTS observations")
    con.commit()
    con.close()


def youtube_to_video_id(db_path: str) -> dict[str, str]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT video_id yt, id vid FROM videos WHERE brain_id=?", (BRAIN,)
    ).fetchall()
    con.close()
    return {r["yt"]: r["vid"] for r in rows}


async def main() -> None:
    settings = get_settings()
    db_path = str(settings.database_path)

    migrate(db_path)
    await init_database(settings)

    raw = json.loads(Path(IN).read_text(encoding="utf-8"))
    vid_map = youtube_to_video_id(db_path)

    obs_models = []
    for o in raw:
        yt = o.get("youtube_id")
        obs_models.append(Observation(
            brain_id=BRAIN,
            video_id=vid_map.get(yt),
            youtube_id=yt,
            creator=CREATORS.get(yt, o.get("video_title", yt)),
            obs_type=o.get("type", "other"),
            claim=o.get("claim", ""),
            value=o.get("value") or None,
            entities=[o["entity"]] if o.get("entity") else [],
            evidence_quote=o.get("evidence_quote"),
            chunk_id=o.get("chunk_id"),
            start_time=o.get("start_time"),
            confidence=o.get("confidence"),
            domain="founders",
        ))

    await delete_observations_by_brain(BRAIN)
    n = await insert_observations(obs_models)
    print(f"Persisted {n} observations")

    # Embed claims.
    client = GeminiClient()
    try:
        claims = [o.claim for o in obs_models]
        print(f"Embedding {len(claims)} claims via {client.embed_model}...")
        vectors = await client.embed_texts(claims)
        for o, vec in zip(obs_models, vectors):
            await store_observation_embedding(
                o.id, client.embed_model, client.embed_dims, vec
            )
    finally:
        await client.close()

    # Cluster by embedding.
    pairs = await get_observation_embeddings(BRAIN)
    assignments = greedy_cluster(pairs, threshold=0.74)
    await set_cluster_ids(assignments)
    n_clusters = len(set(assignments.values()))
    print(f"Clustered into {n_clusters} clusters")

    # Report.
    observations = await get_observations_by_brain(BRAIN)
    report = build_report(BRAIN_NAME, observations)
    Path(OUT).write_text(report, encoding="utf-8")
    print(f"Wrote report to {OUT}  ({len(report)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
