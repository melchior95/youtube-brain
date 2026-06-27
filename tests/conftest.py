import shutil
import tempfile
from pathlib import Path

import pytest

from youtube_brain.config import settings as settings_mod
from youtube_brain.config.settings import Settings
from youtube_brain.storage import database as db_mod


@pytest.fixture(autouse=True)
async def _isolate_db(monkeypatch):
    """Force EVERY test onto a throwaway SQLite DB.

    Without this, any test that lazily triggers ``init_database()`` (with no
    explicit settings) falls through to ``get_settings()`` and reads/writes the
    real ``data/youtube_brain.db``, silently polluting it with fixture rows
    (e.g. "My Brain", "Video <id>"). We point the path at a temp dir via env,
    clear the cached settings, and reset the engine globals so the override
    actually takes effect for both lazy and explicit inits.

    Uses ``tempfile`` rather than pytest's ``tmp_path`` so isolation never
    depends on the pytest basetemp being writable.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="ytbrain-test-"))
    monkeypatch.setenv("YTBRAIN_DATA_DIR", str(tmpdir / "data"))
    monkeypatch.setenv("YTBRAIN_DATABASE_PATH", str(tmpdir / "data" / "test.db"))
    settings_mod.get_settings.cache_clear()
    db_mod._engine = None
    db_mod._async_session_factory = None
    try:
        yield
    finally:
        await db_mod.close_database()
        settings_mod.get_settings.cache_clear()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def tmp_settings():
    """Explicit throwaway Settings for tests that init the DB directly.

    Init kwargs outrank env, so this stays self-consistent regardless of the
    autouse override above. Tempfile-based for the same basetemp-independence.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="ytbrain-cfg-"))
    try:
        yield Settings(
            data_dir=tmpdir / "data",
            database_path=tmpdir / "data" / "test.db",
            gemini_api_key="test-key",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
