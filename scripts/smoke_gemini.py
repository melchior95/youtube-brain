"""Connectivity smoke test for the Gemini client.

Confirms the API key + model in .env actually work with one generate call
and one embed call. Run: python scripts/smoke_gemini.py
"""

import asyncio

from youtube_brain.config.settings import get_settings
from youtube_brain.llm.gemini import GeminiClient


async def main() -> None:
    settings = get_settings()
    print(f"Model:           {settings.gemini_model}")
    print(f"Embedding model: {settings.gemini_embedding_model}")
    print(f"Key present:     {bool(settings.gemini_api_key)} (len={len(settings.gemini_api_key)})")
    print("-" * 50)

    client = GeminiClient()
    try:
        text = await client.generate(
            "Reply with exactly: smoke test ok",
            temperature=0.0,
        )
        print(f"generate() -> {text!r}")

        vecs = await client.embed_texts(["hello world"])
        vec = vecs[0]
        print(f"embed_texts() -> {len(vec)} dims, first 3: {vec[:3]}")

        print("-" * 50)
        print("PASS: Gemini connectivity confirmed.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
