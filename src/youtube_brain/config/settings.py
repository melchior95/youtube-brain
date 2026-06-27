from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YTBRAIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    database_path: Path = Field(default=Path("data/youtube_brain.db"))

    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_dimensions: int = Field(default=768)

    # How many chunks to label per Gemini call. Each batch is ONE request
    # (a longer prompt), so larger values mean fewer requests against the
    # rate limit. Lower it if you prefer smaller individual requests.
    label_batch_size: int = Field(default=10)

    # Minimum seconds between Gemini requests (staggering to avoid bursts).
    # 0 disables. A small random jitter is added on top to avoid perfectly
    # periodic traffic. Set via YTBRAIN_GEMINI_REQUEST_COOLDOWN for real runs.
    gemini_request_cooldown: float = Field(default=0.0)
    gemini_request_jitter: float = Field(default=1.5)

    # scrape.do proxy token — routes transcript fetches through rotating
    # residential IPs to bypass YouTube rate-limiting. Empty = direct fetch.
    scrapedo_token: str = Field(default="")

    # Browser to read YouTube cookies from for authenticated yt-dlp fetches
    # (e.g. "edge", "chrome", "firefox", or "chrome:Profile 1"). Authenticated
    # requests usually sail past rate limits. On Windows the browser often
    # needs to be CLOSED so yt-dlp can read its cookie DB. NOTE: Chrome >= 127
    # uses app-bound encryption and CANNOT be read (DPAPI error) — use Firefox,
    # or export a cookies.txt instead. Never combined with the scrape.do proxy
    # (the proxy must not see account cookies).
    ytdlp_cookies_browser: str = Field(default="")

    # Path to a Netscape-format cookies.txt (export via a "Get cookies.txt
    # LOCALLY" browser extension). Takes precedence over the browser option.
    ytdlp_cookies_file: str = Field(default="")

    @property
    def gemini_api_keys(self) -> list[str]:
        """All configured Gemini API keys, de-duplicated, order-preserving.

        Collects every ``YTBRAIN_GEMINI_API_KEY*`` entry from the environment
        and the .env file (e.g. ``YTBRAIN_GEMINI_API_KEY_01`` ..._07). Each key
        is typically a separate Google Cloud project, so rotating across them
        multiplies the per-project free-tier quota.
        """
        import os

        found: dict[str, str] = {}
        for name, val in os.environ.items():
            if name.startswith("YTBRAIN_GEMINI_API_KEY") and val.strip():
                found[name] = val.strip()

        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if (
                    line.startswith("YTBRAIN_GEMINI_API_KEY")
                    and "=" in line
                    and not line.startswith("#")
                ):
                    name, _, val = line.partition("=")
                    name, val = name.strip(), val.strip()
                    if val:
                        found.setdefault(name, val)

        keys: list[str] = []
        seen: set[str] = set()
        # Bare KEY first, then numbered suffixes in order.
        for name in sorted(found, key=lambda n: (n != "YTBRAIN_GEMINI_API_KEY", n)):
            val = found[name]
            if val not in seen:
                seen.add(val)
                keys.append(val)

        if self.gemini_api_key and self.gemini_api_key not in seen:
            keys.insert(0, self.gemini_api_key)
        return keys

    chunk_window_seconds: float = Field(default=150.0)
    chunk_overlap_seconds: float = Field(default=30.0)

    max_concurrent_fetches: int = Field(default=5)
    partially_ready_threshold: int = Field(default=5)

    http_timeout: int = Field(default=30)
    http_max_retries: int = Field(default=3)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
