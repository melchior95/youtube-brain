# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**YouTube Brain** — turns YouTube creator archives into searchable, timestamp-cited
advisors via hybrid RAG. Users share channel/playlist/video URLs; transcripts are
ingested, chunked, embedded, labeled, and summarized; users then ask questions and
get answers with clickable `youtu.be/<id>?t=<sec>` citations.

Roadmap north star: the atomic unit is shifting from *chunk* to **observation** —
a shared layer that powers cited intelligence reports (consensus, disagreement,
timelines) across this and sibling brain projects.

## Read these first

- **`docs/LEARNINGS.md`** — hard-won, non-obvious lessons. **Read before touching
  the Gemini client, rate limiting, ingestion, or citations.** Especially: the free
  tier is **20 generate requests/day per project** for *both* 2.5- and 3.5-flash.
- `docs/plans/2026-06-02-youtube-brain-design.md` — approved architecture.
- `docs/plans/2026-06-02-youtube-brain-implementation.md` — the 15-task build plan.

## Stack & layout

Python 3.12 · FastAPI · SQLite (FTS5 + JSON) · Gemini (2.5-flash + gemini-embedding-001)
· React PWA. Source under `src/youtube_brain/`:
`config/` settings · `core/` models+enums · `storage/` schema+CRUD · `ingest/`
resolver, transcripts, chunker, labeler, summarizer, pipeline · `llm/` Gemini client
· `retrieval/` 4-lane hybrid search + reranker · `generation/` answers+prompts ·
`api/` FastAPI · `cli.py`. Frontend in `frontend/`.

## Commands

```bash
pytest -k "not integration" --ignore=tests/test_e2e.py   # 132 unit tests
python scripts/smoke_gemini.py                           # verify key+model connectivity
python -m youtube_brain.cli ingest "<url>" --limit N     # bounded ingest
python -m youtube_brain.cli ask <brain_id> "<question>"
python -m youtube_brain.cli list
```

## Conventions & guardrails

- **Secrets**: keys live in `.env` (gitignored) as `YTBRAIN_GEMINI_API_KEY*` — any
  number, loaded + rotated automatically. Never commit `.env`; never echo full keys.
- **Cost/limits**: ingestion spends real quota (~2 generate calls/video). Always
  bound large channels with `--limit`. The client rotates keys, retires per-day-
  exhausted keys, staggers requests, and backs off on 429/5xx — don't remove this.
- **Smoke-test on real data**: unit tests passed while citations were silently
  broken. After changes to ingest/retrieval/citations, run a 1-video ingest + ask.
- **Match existing patterns**; keep changes DRY/YAGNI. Run the test suite before
  committing. Update `docs/LEARNINGS.md` when reality contradicts an assumption.
- The reference implementation for the Gemini key-rotation/quota pattern is
  `src/youtube_brain/llm/gemini.py`; see `docs/LEARNINGS.md` for the rationale.
