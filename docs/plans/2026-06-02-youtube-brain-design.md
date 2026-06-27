# YouTube Brain - Design Document

**Date**: 2026-06-02
**Status**: Approved
**Approach**: B+ (Hybrid RAG now, C-shaped schema for growth)

## Product

Turn any YouTube creator's archive into a searchable advisor. Paste a channel/playlist/video URL, get a "Brain" card. Tap to ask questions, generate articles, playbooks, summaries — all with timestamp-cited answers grounded in the creator's actual words.

"Ask any YouTube channel."

## Stack

- **Backend**: Python, FastAPI, SQLAlchemy (async)
- **Database**: SQLite + FTS5 + sqlite-vec (migrate to PostgreSQL + pgvector later)
- **LLM**: Gemini (generation + embeddings)
- **Frontend**: React PWA (responsive web app)
- **Mobile**: Android WebView wrapper + share intent (Phase 2), iOS WKWebView + share extension (Phase 3)
- **Transcript fetching**: youtube-transcript-api (primary), yt-dlp (fallback)

## Schema

### MVP Tables

```
brains
──────
id                  UUID PK
name                text            -- "Starter Story Brain"
owner_user_id       text nullable
visibility          text            -- private / shared / public
canonical_brain_id  UUID nullable   -- links to reusable public brain
recency_weight      float default 0.1
video_count         int
status              text            -- pending / ingesting / partially_ready / ready / error
created_at          datetime
updated_at          datetime

sources
───────
id                  UUID PK
brain_id            FK -> brains
source_type         text            -- channel / playlist / video
source_url          text
source_title        text
source_id           text            -- channel_id / playlist_id / video_id
status              text
created_at          datetime

videos
──────
id                  UUID PK
brain_id            FK -> brains
source_id           FK -> sources
video_id            text            -- YouTube video ID
title               text
channel_name        text
published_at        datetime
duration_seconds    int
url                 text
transcript_raw      text
transcript_clean    text
transcript_source   text            -- manual / official_caption / auto_caption / yt_dlp / api
transcript_language text
caption_kind        text            -- manual / auto
transcript_quality_score float nullable
failure_reason      text nullable
video_summary       text nullable
key_points          JSON nullable
businesses_mentioned JSON nullable
people_mentioned    JSON nullable
main_topics         JSON nullable
status              text            -- pending / fetched / chunked / summarized / error
created_at          datetime

chunks
──────
id                  UUID PK
video_id            FK -> videos
brain_id            FK -> brains
start_time          float           -- seconds
end_time            float
text                text
topics              JSON
business_type       JSON            -- controlled taxonomy
advice_category     JSON            -- controlled taxonomy
stage               JSON            -- controlled taxonomy
asset_type          JSON            -- controlled taxonomy
created_at          datetime

chunks_fts          FTS5 virtual table on chunks.text

chunk_embeddings
────────────────
chunk_id            FK -> chunks
model               text            -- e.g. "gemini-text-embedding-004"
dimensions          int
embedding           blob
created_at          datetime

observations
────────────
id                  UUID PK
brain_id            FK -> brains
observation         text
source_chunk_ids    JSON
topic               text
confidence          float
created_at          datetime

articles
────────
id                  UUID PK
brain_id            FK -> brains
title               text
body                text            -- markdown
article_type        text            -- summary / playbook / faq / comparison
source_chunk_ids    JSON
created_at          datetime
```

### Future Tables (C-shaped growth)

```
entities            -- people, companies, concepts, tools
claims              -- "SEO takes 6 months to work"
predictions         -- "GME will hit $100" + date + outcome
creator_tracks      -- accuracy scores per creator per topic
brain_links         -- cross-brain relationships
questions           -- user queries (for caching/evals)
answers             -- generated answers
answer_sources      -- which chunks/observations backed each answer
```

### Metadata Taxonomy (Controlled Labels)

```
business_type:      saas / ecommerce / agency / marketplace / content /
                    physical_product / service / mobile_app / other

advice_category:    marketing / distribution / pricing / hiring / fundraising /
                    product / operations / customer_acquisition / retention /
                    monetization / launch / growth / technical / legal / other

stage:              idea / pre_launch / early_stage / growth / scaling /
                    mature / exit / other

asset_type:         interview / tutorial / review / commentary / case_study /
                    earnings_call / lecture / panel / other
```

Topics remain free-form. The rest are controlled to prevent metadata entropy.

## Ingestion Pipeline

```
resolve source (URL → brain + source + video records)
→ fetch transcripts (youtube-transcript-api, yt-dlp fallback)
→ chunk (150s windows, 30s overlap, snap to sentence boundaries)
→ embed (Gemini text-embedding, store in chunk_embeddings)
→ FTS5 index
→ metadata label (Gemini, batched 5-10 chunks, controlled taxonomy)
→ video summary (one cheap Gemini call per video: summary, key_points, mentions)
→ partially_ready (after 5-10 videos) / ready (all videos done)
```

### Pipeline Properties

- **Resumable**: each video/chunk has status, restart picks up where it left off
- **Progressive**: brain queryable at partially_ready (5-10 videos indexed)
- **Idempotent**: re-importing skips already-fetched videos (match on video_id)
- **Rate-limit aware**: Gemini key rotation, back off on 429s
- **Light cleaning**: keep filler removal minimal, citations point to raw transcript

## Retrieval & Answer Stack

### Step 1: Query Processing

Run BOTH original query and expanded query:

```
Original: "What did they say about App Store wrappers?"
Expanded: Gemini rewrites → "mobile app distribution strategy SaaS"
          + metadata_filter extraction from controlled taxonomy
```

Both queries feed into retrieval. Original preserves exact phrases. Expanded catches semantic matches.

### Step 2: Four-Lane Retrieval

```
Query (original + expanded)
├─ Lane 1: FTS5 chunks           → top 50
├─ Lane 2: Vector chunks         → top 50
├─ Lane 3: FTS5 video summaries  → top 10
└─ Lane 4: Vector video summaries→ top 10
(Future Lane 5: observation search)
```

All lanes filtered by brain_id. Metadata filters applied where specified.

### Step 3: Merge + Deduplicate + Diversity Select

```
Union all results → dedupe by chunk_id
→ Diversity constraints:
    max 3 chunks per video
    max 8 chunks per channel (for multi-source brains)
→ Weighted score:
    0.4 × vector_similarity
    0.3 × bm25_score
    0.2 × metadata_match_score
    0.1 × recency_boost (configurable per brain via brain.recency_weight)
→ Take top 20
```

### Step 4: Context Assembly

Top 20 chunks + relevant video summaries assembled with full provenance:
- Video title, channel, timestamp range
- Caption kind (manual/auto) for confidence signaling
- Metadata labels

### Step 5: Generate Answer

Gemini with strict grounding prompt:
- Answer ONLY from provided evidence
- Cite every claim: [Video Title | timestamp]
- Never invent beyond chunks
- Include citation density score

### Step 6: Response Shape

```json
{
  "answer": "markdown with inline citations...",
  "citations": [...],
  "confidence": {
    "level": "high",
    "supporting_chunks": 17,
    "supporting_videos": 9,
    "caption_quality": "mostly_manual"
  },
  "chunks_searched": 80,
  "chunks_used": 20
}
```

### Output Modes (shared retrieval, different prompts)

| Mode       | Output                              |
|------------|-------------------------------------|
| Q&A        | Cited answer                        |
| Article    | Long-form markdown with citations   |
| Playbook   | Numbered action steps + evidence    |
| Summary    | Thematic overview                   |
| FAQ        | Top questions + short answers       |

## Frontend (React PWA)

### Brain Cards View (home)

Grid of brain cards showing:
- Brain name, source channel/playlist
- Video count, status badge
- Last queried timestamp

### Brain Detail View

- **Articles tab**: suggested/generated articles
- **Ask tab**: question input + answer display with citations
- **Videos tab**: list of ingested videos with status
- **Settings**: recency weight, visibility

### Answer Display

- Markdown answer with inline citation links
- Citation sidebar: video title, timestamp (clickable to YouTube at timestamp), raw transcript excerpt
- Confidence badge: "High confidence - 17 chunks, 9 videos" vs "Low confidence - 2 chunks, 1 video"

## Phases

### Phase 1: Core (build now)
- FastAPI backend + SQLite/sqlite-vec/FTS5
- Ingest video/playlist/channel transcripts
- Chunk + embed + metadata label + video summary
- 4-lane hybrid retrieval with diversity selection
- Q&A with timestamp-cited answers
- React PWA with brain cards + ask interface
- Article/playbook/summary generation

### Phase 2: Distribution
- Android WebView wrapper + share intent
- iOS WKWebView wrapper + share extension

### Phase 3: Intelligence
- Observations table populated (compressed wisdom)
- Observation retrieval lane
- Multi-brain queries
- Cross-brain comparison

### Phase 4: Knowledge Graph
- Entities, claims, predictions
- Creator track records
- Prediction accuracy scoring

### Phase 5: Audio (only if users ask)
- TTS output for articles/summaries
- Google Cloud TTS or ElevenLabs
