"""Force a 429 to read the exact quota metric + limit from the error body.

Fires rapid tiny generate calls on a single key until one is rejected, then
prints the structured QuotaFailure / RetryInfo. Tells us whether the binding
limit is per-minute (RPM, recovers in ~60s) or per-day (RPD, resets midnight PT)
and its numeric value. Run: python scripts/probe_burst.py
"""

import sys
from pathlib import Path

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEN_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash"
MAX_CALLS = 16


def first_key() -> tuple[str, str]:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("YTBRAIN_GEMINI_API_KEY") and "=" in line:
            name, _, val = line.partition("=")
            if val.strip():
                return name.strip(), val.strip()
    raise SystemExit("No key found")


def dump_429(resp: httpx.Response) -> None:
    err = resp.json().get("error", {})
    print(f"  status: {err.get('status')}")
    print(f"  message: {err.get('message','')[:200]}")
    for detail in err.get("details", []):
        t = detail.get("@type", "")
        if "QuotaFailure" in t:
            for viol in detail.get("violations", []):
                print("  --- quota violation ---")
                print(f"    metric: {viol.get('quotaMetric')}")
                print(f"    id:     {viol.get('quotaId')}")
                print(f"    value:  {viol.get('quotaValue')}  <-- LIMIT")
                if viol.get("quotaDimensions"):
                    print(f"    dims:   {viol.get('quotaDimensions')}")
        elif "RetryInfo" in t:
            print(f"  retry after: {detail.get('retryDelay')}  "
                  f"(short => per-minute limit; long => per-day)")


def main() -> None:
    name, key = first_key()
    print(f"Bursting {name} (…{key[-6:]}) up to {MAX_CALLS} calls...\n")
    with httpx.Client(timeout=30.0) as c:
        for i in range(1, MAX_CALLS + 1):
            r = c.post(
                f"{BASE}/{GEN_MODEL}:generateContent",
                params={"key": key},
                json={"contents": [{"parts": [{"text": "hi"}]}],
                      "generationConfig": {"maxOutputTokens": 1}},
            )
            print(f"  call {i:2d}: HTTP {r.status_code}")
            if r.status_code == 429:
                print()
                dump_429(r)
                return
        print("\nNo 429 in burst — RPM ceiling is above "
              f"{MAX_CALLS} or requests were spaced too slowly.")


if __name__ == "__main__":
    main()
