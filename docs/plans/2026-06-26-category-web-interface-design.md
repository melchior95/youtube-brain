# Category Web Interface, Design

*Date: 2026-06-26 · Status: approved (brainstorm)*

## Purpose

Add a **category tier** on top of the existing per-brain frontend so YouTube
Brain is browsable as curated **topics** (e.g. "Making Money with AI"), each
seeded with ~5 top creators. A **creator is a brain**; a **category is a curated
set of creators**. The page leads with a **creator browser** (chosen centerpiece)
and surfaces the cross-creator **consensus** built earlier as a secondary tab.

## Scope / non-goals

- **Personal / localhost.** No auth, no multi-user, minimal polish.
- **Seed/config driven**, no in-app category CRUD (upgrade path noted below).
- **No new DB tables**, categories live in a JSON config; creators are matched
  to brains by the stable `channel_id` we keyed brains on.
- Reuse existing components and endpoints heavily; add the minimum.

## Views (spine)

| View | Component | Status |
|------|-----------|--------|
| **Home** | `CategoryGrid` (new), category cards (name, description, "N/M creators pulled") | new |
| **Category** | `CategoryPage` (new), grid of creator cards + **"Consensus" tab** | new |
| **Creator** | `BrainDetail` (existing), videos + per-creator `IntelligencePanel` + ask | reused |

Navigation today is a manual `useState` toggle in `App.tsx`. Replace with a
minimal route layer (Home → Category → Creator), dependency-light.

## Data model, no new tables

`data/categories.json`:

```json
[
  {
    "slug": "making-money-with-ai",
    "name": "Making Money with AI",
    "description": "Operators on turning AI skills into income.",
    "creators": [
      { "handle": "@nateherk", "url": "https://youtube.com/@nateherk",
        "channel_id": "UC2ojq-nuP8ceeHqiroeKhBA" }
    ]
  }
]
```

- **Creator → brain match:** the brain whose `sources.source_id == creator.channel_id`
  (channel_id was backfilled onto sources). `channel_id` is filled once by the
  seed script (one yt-dlp resolve per creator), so request-time matching is a
  fast DB lookup with **no network**.
- **Pending creator** = a config creator with no matching brain → rendered as a
  **Pull card**.

## Backend (FastAPI), new endpoints

- `GET /api/categories` → `[{slug, name, description, creator_count, pulled_count}]`
- `GET /api/categories/{slug}` → `{slug, name, description, creators: [...]}` where
  each creator is either **resolved** `{handle, brain_id, name, video_count,
  status, channel_id, pulled: true}` or **pending** `{handle, url, channel_id,
  pulled: false}`.
- `GET /api/categories/{slug}/consensus` → cross-creator intelligence over the
  category's pulled brains. **Same `Intelligence` shape** as `/intelligence`, so
  the existing `IntelligencePanel` renders it unchanged.

**Reused as-is:** `/api/brains/{id}`, `/api/brains/ingest` (Pull),
`/intelligence`, `/timeline`, `/editorial`, `/ask`.

A small config loader reads `categories.json`; a matching helper resolves
`channel_id → brain`.

## Pull card (confirmed in)

For a pending creator, render a card with the handle + a **Pull** button →
`POST /api/brains/ingest` (existing, bounded, background task). Show an
"ingesting…" state; on completion the creator resolves to a real creator card on
refresh. Reuses the existing `IngestDialog`/`ingestUrl` logic. This makes a
half-filled category self-completing.

## Cross-creator consensus

Computed **live** per visit, reusing the `report` logic: gather observations
across the category's brains → embed any unembedded claims (zero generate) →
`greedy_cluster` in-memory at `CLUSTER_THRESHOLD` → `build_intelligence` →
filter consensus to themes spanning **≥2 distinct creators** (creator-level
identity). Embeddings cache after first run; clustering is in-memory and fast at
this scale. Precompute only if it ever feels slow.

**DRY note:** extract the cross-brain clustering currently inline in
`scripts/skill_bridge.py` into a shared module (e.g.
`youtube_brain/observations/crosscreator.py`) so the bridge and the API call one
implementation, avoid a third copy of the retrieval/cluster path.

## Frontend additions

- `api.ts`: `listCategories()`, `getCategory(slug)`, `getCategoryConsensus(slug)`,
  reuse `ingestUrl()` for Pull.
- Components: `CategoryGrid`, `CategoryPage`, `PullCard`. Reuse `BrainCard`
  (creator cards), `BrainDetail`, `IntelligencePanel` (consensus tab).
- Minimal router (state/hash based).

## Seeding

- `data/categories.json` checked in with the proposed taxonomy (slugs, names,
  descriptions, creator handles; `channel_id` blank initially).
- `scripts/seed_categories.py`: resolve + fill `channel_id` for each creator
  (yt-dlp), validate handles, report which creators are already pulled vs
  pending. Optional `--pull` to bound-ingest pending creators.

## Testing

- **Backend unit:** config loader; `channel_id → brain` matching; the three
  `/api/categories*` endpoints (isolated test DB, seed a couple brains);
  consensus endpoint returns `Intelligence` shape with the ≥2-creator filter.
- **Frontend:** light, `vite` dev smoke (personal tooling); optional render test.
- **Smoke on real data:** Home → open "Making Money with AI" (has @nateherk,
  @shanehummus, Starter Story) → see creator cards + a Pull card for pending
  creators + the consensus tab.

## Open / deferred

- In-app category editing → promote `categories.json` to a `categories` table.
- Precomputed/cached consensus → only if live is slow.
- Explicit **disagreement** view (beyond consensus) → consensus tab v2.

## Appendix, proposed taxonomy (start with the ✅ ones)

| Category | Creators |
|---|---|
| **Making Money with AI** ✅ | Nate Herk ✅, Shane Hummus ✅, Starter Story ✅, Liam Ottley, Nick Saraev |
| **AI Thought / Strategy** | Nate B Jones, AI Explained, Dwarkesh Patel, Wes Roth, Matthew Berman |
| **Content Creation with AI** | Riley Brown, Matt Wolfe, The AI Advantage, Curious Refuge, AI Andy |
| **AI Coding / Agentic Dev** | AI Jason, Cole Medin, IndyDevDan, GosuCoder, Theo (t3) |
| **Finance / Investing** | Graham Stephan, Andrei Jikh, Patrick Boyle, Ben Felix, Joseph Carlson |
| **Solopreneur / Indie SaaS** | Greg Isenberg, Starter Story ✅, Marc Lou, Ben Tossell, Pieter Levels |
| **Marketing / Growth** | Alex Hormozi, GaryVee, Neil Patel, Exploding Topics, MarketingAgainstTheGrain |
| **Productivity / Knowledge Work** | Ali Abdaal, Thomas Frank, Tiago Forte, August Bradley |

Spine for the AI side: **think** (AI Thought) · **earn** (Making Money) · **build**
(AI Coding). Finance & Productivity prove the system generalizes beyond AI.
