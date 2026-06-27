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

    # scrape.do proxy token, routes transcript fetches through rotating
    # residential IPs to bypass YouTube rate-limiting. Empty = direct fetch.
    scrapedo_token: str = Field(default="")

    # Browser to read YouTube cookies from for authenticated yt-dlp fetches
    # (e.g. "edge", "chrome", "firefox", or "chrome:Profile 1"). Authenticated
    # requests usually sail past rate limits. On Windows the browser often
    # needs to be CLOSED so yt-dlp can read its cookie DB. NOTE: Chrome >= 127
    # uses app-bound encryption and CANNOT be read (DPAPI error), use Firefox,
    # or export a cookies.txt instead. Never combined with the scrape.do proxy
    # (the proxy must not see account cookies).
    ytdlp_cookies_browser: str = Field(default="")

    # Path to a Netscape-format cookies.txt (export via a "Get cookies.txt
    # LOCALLY" browser extension). Takes precedence over the browser option.
    ytdlp_cookies_file: str = Field(default="")

    chunk_window_seconds: float = Field(default=150.0)
    chunk_overlap_seconds: float = Field(default=30.0)

    max_concurrent_fetches: int = Field(default=5)
    partially_ready_threshold: int = Field(default=5)

    http_timeout: int = Field(default=30)
    http_max_retries: int = Field(default=3)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
