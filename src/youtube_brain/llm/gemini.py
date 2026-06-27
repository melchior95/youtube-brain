"""Gemini API client using httpx directly, with key rotation and 429 backoff."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time

import httpx

from youtube_brain.config.settings import get_settings

logger = logging.getLogger(__name__)

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_SLEEP_SECONDS = 65.0


def _parse_retry_delay(error: dict) -> float | None:
    """Extract RetryInfo.retryDelay (e.g. '22s') from a Gemini error body."""
    for detail in error.get("details", []):
        if "RetryInfo" in detail.get("@type", ""):
            raw = detail.get("retryDelay", "")
            m = re.match(r"(\d+(?:\.\d+)?)s", str(raw))
            if m:
                return float(m.group(1))
    return None


def _quota_summary(error: dict) -> tuple[str, bool]:
    """Return (human summary, is_per_day) for a QuotaFailure error body."""
    is_per_day = False
    parts: list[str] = []
    for detail in error.get("details", []):
        if "QuotaFailure" in detail.get("@type", ""):
            for viol in detail.get("violations", []):
                qid = viol.get("quotaId", "")
                val = viol.get("quotaValue", "?")
                model = (viol.get("quotaDimensions") or {}).get("model", "")
                if "PerDay" in qid:
                    is_per_day = True
                parts.append(f"{qid}={val} (model={model})")
    return ("; ".join(parts) or "unknown quota", is_per_day)


class GeminiClient:
    """Lightweight async wrapper around the Gemini REST API.

    Rotates across all configured API keys and backs off on HTTP 429.
    Per-minute limits are waited out and retried; per-day limits fail fast
    (after trying every key) so callers can skip rather than hang for hours.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cooldown: float | None = None,
    ) -> None:
        settings = get_settings()
        if api_key:
            self.api_keys: list[str] = [api_key]
        else:
            self.api_keys = settings.gemini_api_keys or [settings.gemini_api_key]
        # Primary key, kept for backwards compatibility and introspection.
        self.api_key: str = self.api_keys[0] if self.api_keys else ""
        self.model: str = model or settings.gemini_model
        self.embed_model: str = settings.gemini_embedding_model
        self.embed_dims: int = settings.gemini_embedding_dimensions
        self.max_retries: int = settings.http_max_retries
        # Staggering: minimum seconds between requests, plus jitter.
        self._cooldown: float = (
            cooldown if cooldown is not None else settings.gemini_request_cooldown
        )
        self._jitter: float = settings.gemini_request_jitter
        self._last_request: float | None = None
        # Keys that hit their per-day quota this session — skipped thereafter.
        self._exhausted: set[str] = set()
        self._client = httpx.AsyncClient(timeout=60.0)

    async def _throttle(self) -> None:
        """Sleep so consecutive requests are spaced by at least the cooldown."""
        if self._cooldown <= 0:
            return
        if self._last_request is not None:
            wait = self._cooldown - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait + random.uniform(0, self._jitter))
        self._last_request = time.monotonic()

    # ------------------------------------------------------------------
    # Core request with rotation + backoff
    # ------------------------------------------------------------------

    async def _post(self, url: str, body: dict) -> httpx.Response:
        """POST with key rotation and 429 backoff. Returns a 2xx response.

        Raises the last error via ``raise_for_status`` if all attempts fail.
        Non-429 errors are raised immediately.
        """
        attempts = max(1, self.max_retries)
        last_error: httpx.Response | None = None

        for attempt in range(attempts):
            retry_delays: list[float] = []
            transient = False

            # Skip keys already known to be out of daily quota this session.
            live_keys = [k for k in self.api_keys if k not in self._exhausted]
            if not live_keys:
                logger.error("All keys exhausted their per-day quota this session.")
                break

            for key in live_keys:
                await self._throttle()
                resp = await self._client.post(url, params={"key": key}, json=body)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:
                    last_error = resp
                    try:
                        error = resp.json().get("error", {})
                    except Exception:
                        error = {}
                    summary, is_per_day = _quota_summary(error)
                    delay = _parse_retry_delay(error)
                    if is_per_day:
                        # Dead for the rest of the session; stop trying this key.
                        self._exhausted.add(key)
                        logger.warning("429 (per-day) on key …%s: %s — retiring key", key[-6:], summary)
                    else:
                        if delay is not None:
                            retry_delays.append(delay)
                        logger.warning(
                            "429 on key …%s: %s%s",
                            key[-6:],
                            summary,
                            f" (retry {delay}s)" if delay else "",
                        )
                    continue
                if resp.status_code >= 500:
                    # Transient server error (e.g. 503 overloaded): rotate + retry.
                    last_error = resp
                    transient = True
                    logger.warning(
                        "%d on key …%s (%s); will retry",
                        resp.status_code,
                        key[-6:],
                        url.rsplit("/", 1)[-1],
                    )
                    continue
                # Any other (4xx) error: surface immediately.
                resp.raise_for_status()

            # If every key just got retired this round, fail fast next iteration.
            if all(k in self._exhausted for k in self.api_keys):
                logger.error("Per-day quota exhausted on all keys; not retrying.")
                break
            if attempt < attempts - 1:
                if retry_delays:
                    sleep_for = min(max(retry_delays), _MAX_SLEEP_SECONDS)
                    logger.info("Keys rate-limited; sleeping %.0fs before retry.", sleep_for)
                    await asyncio.sleep(sleep_for)
                elif transient:
                    # Exponential backoff for transient 5xx.
                    sleep_for = min(2.0 ** attempt, _MAX_SLEEP_SECONDS)
                    logger.info("Transient errors; backing off %.0fs before retry.", sleep_for)
                    await asyncio.sleep(sleep_for)

        if last_error is not None:
            last_error.raise_for_status()
        raise RuntimeError("No API keys configured for Gemini client")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        response_json: bool = False,
    ) -> str:
        """Generate text from *prompt* using the Gemini generateContent endpoint."""
        url = f"{_BASE}/{self.model}:generateContent"

        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        if response_json:
            body["generationConfig"]["responseMimeType"] = "application/json"

        resp = await self._post(url, body)
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
    ) -> dict | list:
        """Generate and parse a JSON response."""
        raw = await self.generate(
            prompt, system=system, temperature=temperature, response_json=True
        )
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more texts, routing to single or batch endpoint."""
        if len(texts) == 1:
            vec = await self._embed_single(texts[0])
            return [vec]
        return await self._embed_batch(texts)

    async def _embed_single(self, text: str) -> list[float]:
        """Embed a single text via the embedContent endpoint."""
        url = f"{_BASE}/{self.embed_model}:embedContent"
        body = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.embed_dims,
        }
        resp = await self._post(url, body)
        return resp.json()["embedding"]["values"]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via batchEmbedContents, batching in groups of 100."""
        all_vectors: list[list[float]] = []
        model_path = f"models/{self.embed_model}"

        for start in range(0, len(texts), 100):
            chunk = texts[start : start + 100]
            url = f"{_BASE}/{self.embed_model}:batchEmbedContents"
            body = {
                "requests": [
                    {
                        "model": model_path,
                        "content": {"parts": [{"text": t}]},
                        "outputDimensionality": self.embed_dims,
                    }
                    for t in chunk
                ]
            }
            resp = await self._post(url, body)
            embeddings = resp.json()["embeddings"]
            all_vectors.extend(e["values"] for e in embeddings)

        return all_vectors

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
