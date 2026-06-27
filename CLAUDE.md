# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**YouTube Brain** turns YouTube creator archives into searchable, timestamp-cited
advisors. Transcripts are fetched (yt-dlp), chunked, indexed in SQLite FTS5, and
embedded with a local model (fastembed). Users ask questions and get answers with
clickable `youtu.be/<id>?t=<sec>` citations. There is **no external API**: Claude,
driving the tool in-loop via the skill bridge, does all the *semantic* work
(summaries, cited answers, observation extraction, synthesis); the only model
that runs automatically is the local retrieval embedder.

Roadmap north star: the atomic unit is the **observation**, a typed, cited claim.
Observations are grouped on their canonical entities to compute consensus
(`report`), divergence and staleness (`lint`), and timelines, all deterministic,
with counts computed rather than guessed.

## Read these first

- `docs/plans/2026-06-02-youtube-brain-design.md` and `-implementation.md`:
  original architecture and build plan. **Historical**: the project has since
  dropped its Gemini/embedding pipeline and gone API-free (FTS5 keyword retrieval
  + entity-based consensus + Claude in-loop), so treat embedding/Gemini details
  in older docs as superseded.
- `docs/LEARNINGS.md`: hard-won lessons. The Gemini quota lessons are now
  historical, but the citation and ingestion pitfalls still apply.

## Stack & layout

Python 3.12 · SQLite (FTS5 + JSON) · yt-dlp · fastembed (local ONNX embeddings).
CLI-focused, no external API. Source under `src/youtube_brain/`:
`config/` settings · `core/` models+enums · `storage/` schema+CRUD · `embed.py`
local embedder + cosine · `ingest/` resolver, transcripts, chunker, pipeline
(transcript -> chunk -> FTS -> embed) · `retrieval/` FTS5/BM25 + reranker ·
`observations/` report (consensus), lint (divergence), rollups, timeline ·
`cli.py` (the `ytbrain list` command). The `scripts/skill_bridge.py` bridge is the
primary surface; `scripts/backfill_embeddings.py` embeds pre-existing brains.
(An older Gemini-based web UI lives on the `web` branch.)

## Commands

```bash
pytest -k "not integration"                              # unit suite, no network
python scripts/skill_bridge.py pull "<url>" [--limit N]  # ingest (transcript + chunk + FTS)
python scripts/skill_bridge.py context <brain_id> "<q>"  # keyword retrieval for a cited answer
python scripts/skill_bridge.py report --all              # entity-based consensus
python scripts/skill_bridge.py lint --all                # contradictions / staleness
python -m youtube_brain.cli list                         # list brains
```

## Conventions & guardrails

- **No external API.** Retrieval is hybrid (local fastembed embeddings + FTS5/BM25
  with a lexical fallback); generation/summarization/extraction are Claude's job
  in-loop. Don't reintroduce a *networked* model client; a local embedder is fine.
- **Counts are computed, never guessed.** Consensus/divergence group observations
  by canonical entity and count distinct sources. An LLM may write the prose but
  only ever wraps real numbers.
- **Smoke-test on real data.** Unit tests passed while citations were silently
  broken once. After changes to ingest/retrieval/citations, run a 1-video `pull`
  + `context`.
- **Match existing patterns**; keep changes DRY/YAGNI. Run the test suite before
  committing. Update `docs/LEARNINGS.md` when reality contradicts an assumption.
- `chunk_embeddings` holds local fastembed vectors (used by `context`).
  `observation_embeddings` remains in the schema but is unused (consensus/lint are
  entity-based). `backfill_embeddings.py` re-embeds chunks whose vectors were made
  by a different model/dimension (e.g. legacy Gemini 768-dim).
