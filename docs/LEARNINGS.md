# YouTube Brain, Learnings

Hard-won, non-obvious lessons. Update this when reality contradicts an assumption.

> **History note.** This project started on Gemini (generation + embeddings) and
> later went **API-free**: FTS5/BM25 keyword retrieval plus a local fastembed
> model for dense retrieval, with Claude in-loop doing all generation. The old
> Gemini quota/rate-limit lessons are gone; what remains below still applies.

---

## Smoke-test on real data (the meta-lesson)

Unit tests passed while real ingestion was broken more than once. After any change
to ingest, retrieval, or citations, run a one-video `pull` + a `context` query and
read the output. The bugs below were all invisible to the unit suite.

## Attribution / citation bugs

- **Citation URLs** use `videos.video_id` (the real YouTube id, e.g. `vbEKEWtnndU`),
  NOT `chunks.video_id` (an internal UUID FK to `videos.id`). Retrieval JOINs expose
  it as `youtube_id`. Symptom of getting it wrong: dead `youtu.be/<uuid>` links.
- **Video title / channel** live on `videos`, not `chunks`; retrieval must
  `JOIN videos` for `title`, `channel_name`, `caption_kind`. Without the JOIN the
  diversity selector's per-channel cap silently no-ops and citations show blank titles.
- **Single-video ingest** must fetch metadata via `yt-dlp --dump-json`; an early
  resolver hardcoded `title=None`.
- **Evidence quotes** in observations must be copied VERBATIM from the transcript so
  `attribute()` can map them back to a chunk and recover a `youtu.be?t=` timestamp.
  Paraphrased quotes lose their citation.

## Video metadata / stats

- **`--flat-playlist` (channel/playlist enumeration) omits `view_count`,
  `like_count`, `comment_count`, `channel_follower_count` — and even
  `channel`/`channel_id`.** Only a full per-video `yt-dlp --dump-json` fetch
  (no `--flat-playlist`) returns them, confirmed empirically, not assumed. This
  means capturing view/like/comment/subscriber counts for channel or playlist
  ingestion is NOT free: the pipeline does one extra best-effort `yt-dlp`
  subprocess call per video (`resolver.fetch_video_stats`) to backfill them,
  skipped when the video_meta already carries `view_count` (the single-video
  resolve path already did a full fetch). That's one more request per video on
  top of the transcript fetch, worth knowing given how much effort the IP/rate
  limit resilience stack elsewhere in this project already spends.
- Because of the above, `channel_name` is still `None` for channel/playlist-
  ingested videos (a pre-existing gap, not introduced by the stats work) — flat
  enumeration doesn't return `channel`/`uploader` either, and the stats
  backfill deliberately doesn't fill it in (scoped tightly to the 4 numeric
  fields). Fix would be the same `fetch_video_stats`-style backfill extended to
  `channel_name`/`channel_id`.
- **`view_count` is a snapshot, not live** — captured once at ingest and never
  updated automatically. Comparing it across videos ingested weeks apart isn't
  apples-to-apples (a just-pulled video's count reflects "now"; an old one
  reflects whenever it happened to be pulled). `pipeline.refresh_video_stats`
  is an on-demand, per-brain re-fetch (`skill_bridge.py refresh-stats`) for
  when current numbers matter; it's deliberately NOT wired into the
  `watch run` cron loop, since that would silently add one yt-dlp call per
  already-known video on every scheduled tick.
- **Outlier detection groups by `brain_id`, not `channel_name`** —
  `observations/outliers.py`'s `compute_outliers` sidesteps the
  `channel_name`-is-null gap above by using the brain itself as the grouping
  unit (one brain == one creator in this project's ingestion model), comparing
  each video's `view_count` against its brain's own median.

## Retrieval (FTS5 + local embeddings)

- **`chunks_fts` is kept in sync by triggers**, not by `insert_chunks`. So keyword
  search works the moment chunks are inserted, independent of embeddings. This is
  why a brain with no embeddings still answers (the `context` lexical fallback).
- **Embedding dimension mismatch is silent.** Cosine over a 384-dim query vector and
  a stored 768-dim vector is meaningless; the guard returns 0.0, so dense retrieval
  quietly degrades to the FTS boost only. When the model changes, you MUST re-embed,
  not just fill in missing rows. `backfill_embeddings.py` re-embeds any chunk whose
  stored `model`/`dimensions` differ from the current model (it caught legacy Gemini
  768-dim vectors left in the table after the migration).
- **fastembed is CPU-bound**; call it via `asyncio.to_thread` so embedding a video's
  chunks at ingest doesn't block the event loop. First use downloads the model once
  (`bge-small-en-v1.5`, ~130MB), then it's fully offline.
- Dense recall is the point: a question worded differently than the transcript (e.g.
  "earliest customers" vs "first users / launch threads on reddit") still matches on
  meaning, which BM25 alone misses.

## Consensus and the trust rule

- "Counts are computed, never guessed" is stronger with **entity grouping** than with
  embedding clusters. A cluster boundary is model- and threshold-dependent: it
  silently over-merges similar claims and under-merges paraphrases. Grouping
  observations on their canonical `entities` and counting distinct sources gives a
  visible recall boundary instead of an invisible trust failure.
- So embeddings are used for **finding** chunks (retrieval), never for **counting**
  consensus. `report` and `lint` are pure entity grouping, no model involved.

## Testing & ops

- ~112 unit tests: `pytest -k "not integration"`. No API key needed; nothing hits the
  network in the default run.
- The local embedder needs a model download, so unit tests never call it directly:
  `embed.cosine` is unit-tested as a pure function, and the pipeline tests patch
  `embed_texts` / `store_embedding`.
- CLI surface is just `ytbrain list`; everything else runs through
  `scripts/skill_bridge.py`. `scripts/backfill_embeddings.py` embeds pre-existing
  brains' chunks without re-ingesting.
