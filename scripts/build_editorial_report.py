"""Generate the editorial (magazine-style) report via an LLM.

The raw Intelligence Report is the audit layer; this is the product layer.
Costs ~1 Gemini generate call. The article is written from a structured brief
whose every number comes from the clustered observation data.

Run: python scripts/build_editorial_report.py [BRAIN_ID]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.editorial import generate_editorial
from youtube_brain.storage.database import init_database
from youtube_brain.storage.observations import get_observations_by_brain

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BRAIN = sys.argv[1] if len(sys.argv) > 1 else "ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7"
BRAIN_NAME = "Starter Story"
OUT = "data/editorial_report.md"


async def main() -> None:
    await init_database()
    obs = await get_observations_by_brain(BRAIN)
    if not obs:
        print("No observations for this brain.")
        return
    client = GeminiClient()
    try:
        article = await generate_editorial(client, BRAIN_NAME, obs)
    finally:
        await client.close()
    Path(OUT).write_text(article, encoding="utf-8")
    print(f"Wrote editorial to {OUT} ({len(article)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
