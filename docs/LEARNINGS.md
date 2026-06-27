# YouTube Brain — Learnings

Hard-won, non-obvious lessons. Update this when reality contradicts an assumption.

---

## Gemini API (verified live, June 2026)

**Models**
- `gemini-3.5-flash` — GA since 2026-05-19. The newest premium Flash model.
- `gemini-2.5-flash` — current default for ingest (labeling/summary/answers).
- `gemini-embedding-001` — text embeddings. Supports Matryoshka dimension
  scaling, so our 768-dim config works via `outputDimensionality`.
- `text-embedding-004` — **DEPRECATED 2026-01-14, now 404s.** Do not use.

**Free-tier limits (confirmed from 429 quota metadata, not docs)**
- **Both 2.5-flash and 3.5-flash: 20 generate requests/DAY, per *project*.**
  Switching models does NOT escape the daily wall — both are 20/day.
- Per-minute: 2.5-flash ≈ 5 RPM. Hit first in a burst, but 20/day is the real cap.
- Embeddings (`gemini-embedding-001`) have a **separate, much larger** quota —
  never the bottleneck. Embeds keep returning 200 while generate is exhausted.
- Limits are **per project, not per key.** Multiple keys multiply quota only if
  they belong to **separate Google Cloud projects** (ours do — 7 keys = ~140/day).
- The official docs no longer publish per-model RPD; the **429 error body is the
  authoritative source**. It carries `error.details[]`:
  - `QuotaFailure.violations[]` → `quotaId`, `quotaValue` (the limit),
    `quotaDimensions.model`. `quotaId` contains `PerDay` or `PerMinute`.
  - `RetryInfo.retryDelay` → e.g. `"22s"`.
- Inspect limits anytime with `scripts/probe_quota.py` and `scripts/probe_burst.py`.

**Cost math**
- Each video ingest ≈ **2 generate calls** (1 label batch of 10 chunks + 1 summary)
  plus 1 embedding batch.
- 20/day/project ÷ 2 ≈ **10 videos/day/project**. 7 projects ≈ 70 videos/day.
- A 158-video channel → ~2–3 days on the free tier, or more keys.

## Rate-limit handling

The reference implementation is a prior internal Gemini indexer
(the user's Magazine Brain indexer — battle-tested across the whole brain fleet).

**Ported into `llm/gemini.py`:**
- Rotate across all `YTBRAIN_GEMINI_API_KEY*` keys.
- **Retire a key for the session** once it returns a per-day 429 (stop re-pinging
  dead keys — the first naive run wasted a 429 round-trip on the dead key *every*
  call).
- Differentiate 429s: **per-minute** → sleep `retryDelay` and retry; **per-day** →
  retire key, fail fast (never hang for hours).
- **Transient 5xx** (503 overloaded) → exponential backoff + rotate, not drop.
- **Stagger** requests: `YTBRAIN_GEMINI_REQUEST_COOLDOWN` (default 0; `.env` sets 4s)
  + random jitter. Avoids per-minute bursts and fingerprintable traffic.

**Not yet ported (next steps for robust multi-day ingestion):**
- Persistent `.gemini_key_state` JSON (per-key `first_used_utc` + count, 24h-from-
  first-use window) so quota tracking survives across runs/days.
- `random.shuffle` of key order so the drain sequence varies (anti-fingerprinting).
- `thinking_budget=0` on generate calls for cheaper/faster extraction.
- Deterministic-failure routing (safety filter / JSON truncation → skip, never
  retry-burn quota).

## Attribution bugs (found only by ingesting real data)

Unit tests passed but real ingestion exposed these — **always smoke-test on real data.**
- **Citation URLs**: use `videos.video_id` (the real YouTube ID, e.g. `vbEKEWtnndU`),
  NOT `chunks.video_id` (an internal UUID FK to `videos.id`). Search JOINs expose it
  as `youtube_id`. Symptom: dead `youtu.be/<uuid>` links.
- **Video title/channel**: the `chunks` table has neither — search lanes must
  `JOIN videos` for `title`, `channel_name`, `caption_kind`. Without the JOIN, the
  diversity selector's per-channel cap silently no-ops and citations show blank titles.
- **Single-video ingest** must fetch metadata via `yt-dlp --dump-json`; the resolver
  used to hardcode `title=None`.

## Testing & ops

- 132 unit tests. Run: `pytest -k "not integration" --ignore=tests/test_e2e.py`.
- Integration/e2e tests skip without `YTBRAIN_GEMINI_API_KEY`.
- Tests read the real `.env` (a known fragility) — multi-post batch tests pass
  `cooldown=0` to stay fast, and model-default assertions track `.env`.
- CLI: `ytbrain ingest <url> [--limit N] [--name ...]`, `ask`, `serve`, `list`.
- `scripts/smoke_gemini.py` verifies key + model connectivity before an ingest.

## Scheduling the watchlist (step 4)

The watchlist loop is driven by per-brain schedules (`watchlist_schedules` table)
and a one-shot **`ytbrain watch run`** that refreshes every brain whose interval
has elapsed. `run` is idempotent and quota-bounded (`max_videos` per brain), so
the right pattern is to let the OS scheduler call it — not a long-lived daemon.

```
ytbrain watch enable <brain_id> --interval-hours 24 --max-videos 2
ytbrain watch list      # shows DUE / waiting / off
ytbrain watch run       # refresh all due brains (what the scheduler calls)
ytbrain watch loop      # convenience foreground daemon (polls instead of cron)
```

**Windows Task Scheduler** (daily) — run `watch run` quietly:
```
schtasks /create /tn "YouTubeBrain Watch" /sc daily /st 08:00 ^
  /tr "cmd /c cd /d \"C:\Python Projects\Youtube Brain\" && python -m youtube_brain.cli watch run"
```
**cron** (mac/Linux, daily 8am): `0 8 * * * cd /path/to/Youtube\ Brain && python -m youtube_brain.cli watch run`

Quota math: each refresh ≈ `max_videos` × (~2 generate + 1 embed) + 1 re-cluster
embed batch. At `--max-videos 2` that's ~4 generate calls/brain/day — safely
inside the 20 RPD/key budget. `is_due` is a pure function (unit-tested); the
24h window is measured from `last_refreshed_at`, so a missed day just runs late,
never double-runs.

## Current corpus

- Brain `ddebd2dc-f8f0-4d4f-96a1-964f0c602cf7` — "Starter Story", 10 latest videos,
  81 chunks, 81 embeddings, 9/10 summaries, 73/81 labeled. Ready for the
  observation-extraction spike (see `docs/plans/`).
