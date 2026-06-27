---
name: youtube-brain
description: Use when the user shares a YouTube video / channel / playlist URL and wants its transcript pulled and summarized, OR asks a question to draw on creators they've already pulled, OR asks an open research question that YouTube should answer (e.g. "what's the best way to do X", "what do creators say about Y") — search YouTube, ingest the best results, and answer with citations. Triggers on "summarize this video", "pull this channel", "what does <creator> say about X", "ask youtube", "research <topic> on youtube", or any youtu.be / youtube.com URL handed over. Claude does the summary/answer in-loop — zero Gemini generate quota.
---

# YouTube Brain — Claude as the summarizer & answerer

> **Install:** copy this directory to `~/.claude/skills/youtube-brain/` (Claude
> Code) and set the repo root in the "Mandatory setup" block below to wherever
> you cloned YouTube Brain. The skill shells out to `scripts/skill_bridge.py`.

## Overview
`scripts/skill_bridge.py` is a **zero-generate** bridge over the YouTube Brain
pipeline. It fetches transcripts, chunks + embeds them, and persists them into a
brain — then hands the raw material to **you (Claude)** to summarize and to
answer questions with citations. It never spends the precious Gemini *generate*
quota (free tier ≈ **20 generate calls/day**); only the large *embed* quota.

You fill two roles the bridge deliberately leaves open: the **summarizer** (read
the returned transcript, write the summary) and the **answerer** (read the
returned chunks, write a cited answer).

## Mandatory setup
ALWAYS run from the project root with the python that has the package installed
(the DB path in settings is relative):

```bash
cd /path/to/youtube-brain        # <- the repo root you cloned
python scripts/skill_bridge.py <subcommand> ...
```

The bridge writes one JSON value to **stdout**; logs go to stderr. Parse stdout.

## Workflow A — pull a URL and summarize

```bash
python scripts/skill_bridge.py pull "<youtube-url>" [--limit N]
```

- A single video is grouped under its **channel's** brain (by channel name), so
  pulling more videos from the same creator accumulates into one brain — that's
  what makes "ask the channel later" work.
- Channel / playlist URLs default to `--limit 6` (bounds cost). Raise it only if
  the user asks for more.

Read the JSON, then **write the summary yourself** from `videos[].transcript`:
report the brain name + `brain_id`, and per video give a tight summary, key
points, and notable claims. If `transcript_truncated` is true, say so and offer
to go deeper via Workflow B. Surface any `errors` (e.g. no captions available).

## Workflow F — Ask YouTube (research a question from scratch)

When the user asks an open question that YouTube should answer (e.g. *"what's the
best way to use TikTok AI videos to market my app?"*) — not tied to a URL or an
already-pulled creator — discover, ingest, and answer:

1. **Search** for candidates (zero generate, no API). For fast-moving topics
   (AI, marketing, finance, anything "right now / 2026"), add **`--recent`** so
   YouTube returns only recent uploads — old how-to advice is usually stale:
   ```bash
   python scripts/skill_bridge.py search "<question or tight keyword form>" --recent year --limit 15
   ```
   (`--recent today|week|month|year`; omit for evergreen topics.)
2. **Curate.** Read `results[]` and pick the ~5–8 most relevant, substantive
   videos — favor on-topic titles, real duration (skip 1-min clips), decent
   `view_count`; drop clickbait and off-topic results.
3. **Ingest the chosen ones into a TOPIC brain** (zero generate; `--brain`
   collects them into one research brain instead of scattering per-channel — run
   once per chosen video):
   ```bash
   python scripts/skill_bridge.py pull "<url>" --brain "<topic slug>"
   ```
4. **Retrieve across the topic brain and answer** (the `pull` output gives the
   topic `brain_id`):
   ```bash
   python scripts/skill_bridge.py context "<topic brain_id>" "<the question>" --k 16
   ```
   Then **write the answer yourself** from `results[]` — synthesize across the
   videos, attribute each point to its `creator`, cite with the `youtu.be/<id>?t=`
   link, and note where creators disagree. The `pull` output gives each video's
   `published` date — surface it ("as of <month>") and weight recent sources for
   time-sensitive topics. The topic brain persists, so follow-ups reuse it (B).

## Workflow B — answer a question, using already-pulled data

1. List what's been pulled and pick the brain that matches the creator/topic:
   ```bash
   python scripts/skill_bridge.py brains
   ```
2. Retrieve the most relevant moments (one embed call, zero generate):
   ```bash
   python scripts/skill_bridge.py context <brain_id> "<question>" [--k 12]
   ```
3. **Write the answer yourself** from `results[]`. Ground every claim in the
   chunks and cite with the provided `citation` link, e.g.
   `youtu.be/<id>?t=<sec>`. If the chunks don't cover it, say so — don't invent.

If no brain matches the question, tell the user it hasn't been pulled yet and
offer to run Workflow A.

## Workflow C — cross-creator synthesis (consensus / disagreement)

This is the differentiated move: answer across MULTIPLE creators at once, with
attribution. Each result carries `creator` + `brain`, so you can say who said what.

```bash
# across specific creators:
python scripts/skill_bridge.py context --brains <id1>,<id2>,<id3> "<question>" --k 18
# or across everything pulled so far:
python scripts/skill_bridge.py context --all "<question>" --k 18
```

Then synthesize: lead with **consensus** (what most creators agree on), then
**disagreement / unique takes** (who diverges and how), each line cited to a
`youtu.be/<id>?t=<sec>` link and attributed to its `creator`. This is the answer
NotebookLM / a raw paste can't give — lean into it.

## Workflow D — persist intelligence (write-back)

Summaries and cross-creator answers are re-derived from chunks every time unless
you persist them. To make a brain COMPOUND, extract typed **observations** and
save them — zero generate (Claude is the extractor; the bridge only stores).

1. From a video's transcript/chunks, extract observations as JSON. Each evidence
   quote MUST be copied **verbatim** from the transcript so it attributes to a
   chunk (recovering a `youtu.be?t=` citation). Shape:
   ```json
   {"brain_id": "...", "videos": [{"youtube_id": "...", "creator": "...",
     "summary": "...",
     "observations": [{"type": "tactic", "entity": "...", "claim": "...",
       "value": "", "evidence_quote": "VERBATIM snippet", "confidence": 0.9}]}]}
   ```
   Types: acquisition_channel, business_model, monetization, metric, mistake,
   tactic, tool, principle, market (or "other").
2. Persist (updates video summaries, inserts + embeds observations):
   ```bash
   python scripts/skill_bridge.py save "<path-to.json>"
   ```
3. Re-read the durable layer any time (across creators with `--brains`/`--all`):
   ```bash
   python scripts/skill_bridge.py observations --brains <id1>,<id2> [--limit N]
   ```
   Prefer answering from these persisted, pre-cited observations over re-deriving.

## Workflow E — intelligence report (consensus / disagreement)

Once observations are persisted (Workflow D), build a deterministic report —
counts are computed from clustered observations, never asked of an LLM:

```bash
# one creator (consensus across that brain's videos/founders):
python scripts/skill_bridge.py report <brain_id> --out data/report.md
# cross-creator (a theme only counts when >=2 distinct CREATORS land in it):
python scripts/skill_bridge.py report --brains <id1>,<id2> [--out ...]
python scripts/skill_bridge.py report --all
```

Single-brain reuses the persisted per-brain clusters; cross-creator clusters in
memory (so per-brain clusters aren't disturbed). The JSON gives `top_consensus`;
the full Markdown (with cited evidence per creator) is written to `--out`. Zero
generate (embeddings only). Present the consensus themes and their citations;
note when a brain has no observations yet (run Workflow D first).

## Workflow G — lint (find contradictions, flip-flops & stale advice)

The DUAL of the report: where `report` shows what creators AGREE on, `lint`
surfaces where they CLASH — or where one creator's advice CHANGED over time.
Needs persisted observations (Workflow D). Zero generate — the bridge groups
observations by shared entity and orders them by date; YOU adjudicate.

```bash
python scripts/skill_bridge.py lint <brain_id>             # one creator: flip-flops & stale advice
python scripts/skill_bridge.py lint --brains <id1>,<id2>   # cross-creator: who disagrees
python scripts/skill_bridge.py lint --all
```

The bridge emits `candidates[]` — entity groups with real tension (≥2 distinct
sources, or spanning ≥2 dates). Each observation carries `creator`, its
`published` date, `value`, `evidence_quote`, and a `citation`. For each
candidate, **classify it and write it up yourself**:

- **Contradiction** — 2+ creators assert incompatible things about the same
  entity. Cite BOTH sides.
- **Evolution** — one creator's stance changed across dates. Show the timeline
  ("was X as of `<date>`, now Y as of `<date>`"), each step cited.
- **Stale** — an older claim superseded by a newer one; flag the old as stale.
- **Consistent** — no real conflict; drop it (don't manufacture disagreement).

Lead with the sharpest contradictions/flips; every line cites a
`youtu.be/<id>?t=` link. This is the temporal/divergence intelligence the
per-video AI tools can't do. If `candidates_total` exceeds `candidate_count`,
say the list was capped and offer `--max-groups N`.

## Quota guardrail
- `pull` and `context` cost **zero generate** calls — safe to use freely.
- Do NOT fall back to `python -m youtube_brain.cli ingest` / `ask` for this skill:
  those spend the 20/day generate budget. The whole point of the bridge is to
  keep Claude in the loop instead.

## Common mistakes
- Running from the wrong directory → "no such table" / empty results. `cd` first.
- Summarizing from `context` chunks (partial) when the user wanted a whole-video
  summary → use `pull` for summaries, `context` for Q&A. `pull` returns the FULL
  transcript by default (`--max-chars 0`); pass a positive `--max-chars` only if
  you deliberately want a shorter preview.
- Quoting an answer without a `citation` link → always cite from `results[]`.
- Treating a per-video "Video <id>" brain as the channel → if a pull couldn't
  resolve the channel name (yt-dlp failed), the video lands in its own brain;
  mention it and re-pull if grouping matters.
