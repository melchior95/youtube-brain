"""Probe Gemini quota by reading rate-limit metadata straight from the API.

A 429 response body carries structured QuotaFailure + RetryInfo details that
state the exact limit value and which metric was exhausted. This is the
authoritative way to discover the real per-day quota.

Reads every YTBRAIN_GEMINI_API_KEY* entry from .env and probes each with one
tiny generate call and one tiny embed call. Run: python scripts/probe_quota.py
"""

import json
from pathlib import Path

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEN_MODEL = "gemini-3.5-flash"
EMB_MODEL = "gemini-embedding-001"


def load_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    env = Path(".env")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("YTBRAIN_GEMINI_API_KEY") and "=" in line:
            name, _, val = line.partition("=")
            if val.strip():
                keys[name.strip()] = val.strip()
    return keys


def show(label: str, resp: httpx.Response) -> None:
    print(f"  {label}: HTTP {resp.status_code}")
    # Surface any rate/quota headers
    for h, v in resp.headers.items():
        if any(k in h.lower() for k in ("quota", "ratelimit", "retry")):
            print(f"    header {h}: {v}")
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {})
        except Exception:
            print(f"    body: {resp.text[:300]}")
            return
        print(f"    status: {err.get('status')}  message: {err.get('message','')[:160]}")
        for detail in err.get("details", []):
            t = detail.get("@type", "")
            if "QuotaFailure" in t:
                for viol in detail.get("violations", []):
                    print(f"    QUOTA metric:  {viol.get('quotaMetric')}")
                    print(f"          id:      {viol.get('quotaId')}")
                    print(f"          value:   {viol.get('quotaValue')}  <-- the limit")
                    dims = viol.get("quotaDimensions")
                    if dims:
                        print(f"          dims:    {dims}")
            elif "RetryInfo" in t:
                print(f"    retry after: {detail.get('retryDelay')}")
            elif "Help" in t:
                pass
            else:
                print(f"    detail: {json.dumps(detail)[:200]}")


def probe(name: str, key: str) -> None:
    print(f"\n=== {name} (…{key[-6:]}) ===")
    with httpx.Client(timeout=30.0) as c:
        r = c.post(
            f"{BASE}/{GEN_MODEL}:generateContent",
            params={"key": key},
            json={"contents": [{"parts": [{"text": "hi"}]}],
                  "generationConfig": {"maxOutputTokens": 1}},
        )
        show(f"{GEN_MODEL} generate", r)

        r = c.post(
            f"{BASE}/{EMB_MODEL}:embedContent",
            params={"key": key},
            json={"content": {"parts": [{"text": "hi"}]},
                  "outputDimensionality": 768},
        )
        show(f"{EMB_MODEL} embed", r)


if __name__ == "__main__":
    keys = load_keys()
    if not keys:
        print("No YTBRAIN_GEMINI_API_KEY* entries found in .env")
    for name, key in keys.items():
        probe(name, key)
