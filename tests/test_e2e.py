import os

import pytest


@pytest.mark.integration
async def test_ingest_single_video_and_ask(tmp_path, monkeypatch):
    """Full pipeline test with a real YouTube video. Run: pytest -m integration"""
    api_key = os.environ.get("YTBRAIN_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("No Gemini API key")

    from youtube_brain.config.settings import Settings, get_settings
    from youtube_brain.storage.database import init_database, close_database
    from youtube_brain.ingest.pipeline import ingest_url
    from youtube_brain.storage.brains import get_brain
    from youtube_brain.llm.gemini import GeminiClient
    from youtube_brain.generation.answer import generate_answer

    settings = Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "test.db",
        gemini_api_key=api_key,
    )
    get_settings.cache_clear()
    monkeypatch.setattr("youtube_brain.config.settings.get_settings", lambda: settings)

    await init_database(settings)

    try:
        result = await ingest_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            brain_name="Test Brain",
        )
        assert result.videos_processed >= 1
        assert result.chunks_created >= 1

        brain = await get_brain(result.brain_id)
        assert brain is not None
        assert brain.name == "Test Brain"

        client = GeminiClient()
        try:
            answer = await generate_answer(
                query="What is this video about?",
                brain_id=result.brain_id,
                brain_name=brain.name,
                client=client,
            )
            assert len(answer.answer) > 0
            assert answer.chunks_used > 0
            assert len(answer.citations) > 0
        finally:
            await client.close()
    finally:
        await close_database()
        get_settings.cache_clear()
