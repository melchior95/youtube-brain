"""Load Claude-extracted observations + summaries for the back-catalog videos.

Reads data/observations_new.json ([{youtube_id, summary, observations}]),
updates each video's summary, attributes evidence quotes to chunks, inserts
observations, embeds new claims (Gemini embeddings only), re-clusters the
whole store, and writes the refreshed raw report.
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
from youtube_brain.observations.extractor import attribute
from youtube_brain.observations.refresh import recluster
from youtube_brain.observations.report import build_intelligence, build_report
from youtube_brain.storage.database import init_database
from youtube_brain.storage.observations import get_observations_by_brain, insert_observations

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BRAIN = "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
IN = "data/observations_new.json"


def main_sync_lookup(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    out = {}
    for r in con.execute(
        "SELECT id, video_id, title FROM videos WHERE brain_id=?", (BRAIN,)
    ).fetchall():
        chunks = con.execute(
            "SELECT id, start_time, text FROM chunks WHERE video_id=? ORDER BY start_time",
            (r["id"],),
        ).fetchall()
        out[r["video_id"]] = {
            "uuid": r["id"], "title": r["title"],
            "chunks": [{"id": c["id"], "start_time": c["start_time"], "text": c["text"]}
                       for c in chunks],
        }
    con.close()
    return out


async def main() -> None:
    settings = get_settings()
    db_path = str(settings.database_path)
    await init_database(settings)

    videos = main_sync_lookup(db_path)
    data = json.loads(Path(IN).read_text(encoding="utf-8"))

    con = sqlite3.connect(db_path)
    total_obs, attributed = 0, 0
    all_models: list[Observation] = []
    for entry in data:
        yt = entry["youtube_id"]
        v = videos.get(yt)
        if not v:
            print(f"  ! no ingested video for {yt}, skipping")
            continue
        # Summary -> video record.
        if entry.get("summary"):
            con.execute("UPDATE videos SET video_summary=?, status='summarized' WHERE id=?",
                        (entry["summary"], v["uuid"]))
        for o in entry.get("observations", []):
            attr = attribute(o.get("evidence_quote", ""), v["chunks"])
            if attr:
                attributed += 1
            all_models.append(Observation(
                brain_id=BRAIN, video_id=v["uuid"], youtube_id=yt,
                creator=v["title"] or yt,
                obs_type=o.get("type", "other"), claim=o["claim"],
                value=o.get("value") or None,
                entities=[o["entity"]] if o.get("entity") else [],
                evidence_quote=o.get("evidence_quote"),
                chunk_id=attr["chunk_id"] if attr else None,
                start_time=attr["start_time"] if attr else None,
                confidence=o.get("confidence"), domain="founders",
            ))
            total_obs += 1
    con.commit()
    con.close()

    n = await insert_observations(all_models)
    print(f"Inserted {n} observations ({attributed}/{total_obs} attributed to chunks)")

    client = GeminiClient()
    try:
        n_clusters = await recluster(BRAIN, client)
    finally:
        await client.close()
    print(f"Re-clustered all observations into {n_clusters} clusters")

    obs = await get_observations_by_brain(BRAIN)
    intel = build_intelligence("Starter Story", obs)
    Path("data/intelligence_report.md").write_text(
        build_report("Starter Story", obs), encoding="utf-8")
    print(f"\nCorpus: {intel['total_observations']} observations, "
          f"{intel['founders']} founders, {len(intel['consensus'])} consensus themes")
    print("Top consensus:")
    for t in intel["consensus"][:6]:
        print(f"  {t['founders']}/{intel['founders']}  {t['label'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
