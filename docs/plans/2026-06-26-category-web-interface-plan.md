# Category Web Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a category tier over the per-brain frontend — browsable topics (e.g. "Making Money with AI"), each a curated set of creator-brains, leading with a creator browser and a secondary cross-creator consensus tab.

**Architecture:** Categories live in a JSON config in the package (no new DB tables); creators are matched to brains by the stable `channel_id` on `sources.source_id`. Three new FastAPI endpoints (`/api/categories`, `/api/categories/{slug}`, `/api/categories/{slug}/consensus`) reuse existing brain endpoints. The consensus endpoint and the bridge's `report --brains` share one cross-creator module. Frontend adds Home → Category → Creator views, reusing `BrainCard`, `BrainDetail`, `IntelligencePanel`.

**Tech Stack:** Python 3.12 · FastAPI · SQLAlchemy/SQLite · pytest (asyncio auto, isolated DB via `tests/conftest.py`) · React + Vite + TypeScript.

**Conventions:** Run `pytest` from the project root. Run the dev frontend from `frontend/`. Commit after each task. The autouse fixture in `tests/conftest.py` already isolates the DB, so tests never touch `data/youtube_brain.db`.

---

## Task 1: Shared cross-creator consensus module

DRY: today the in-memory cross-brain clustering lives inline in `scripts/skill_bridge.py` (`cmd_report` multi-brain branch + `_render_cross_report`). Extract it so the API and the bridge share one implementation, with consensus counted at **creator** granularity.

**Files:**
- Create: `src/youtube_brain/observations/crosscreator.py`
- Test: `tests/test_crosscreator.py`

**Step 1: Write the failing test**

```python
# tests/test_crosscreator.py
import pytest
from youtube_brain.core.models import Brain, Source, Video, Chunk, Observation
from youtube_brain.core.enums import SourceType, SourceStatus
from youtube_brain.storage.brains import insert_brain
from youtube_brain.storage.videos import insert_video
from youtube_brain.storage.observations import insert_observations
from youtube_brain.observations.crosscreator import creator_consensus


def _obs(brain_id, creator, claim, yt, conf=0.9):
    return Observation(brain_id=brain_id, youtube_id=yt, creator=creator,
                       obs_type="principle", claim=claim, confidence=conf,
                       cluster_id=None)


def test_creator_consensus_counts_distinct_creators_not_videos():
    bid = "11111111-1111-1111-1111-111111111111"
    # Two creators in cluster 0 (real consensus); one creator twice in cluster 1.
    obs = [
        _obs(bid, "Nate", "AI skills pay a lot", "v1"),
        _obs(bid, "Shane", "AI skills pay a lot", "v2"),
        _obs(bid, "Shane", "pick a niche", "v3"),
        _obs(bid, "Shane", "pick a niche", "v4"),
    ]
    obs[0].cluster_id = 0
    obs[1].cluster_id = 0
    obs[2].cluster_id = 1
    obs[3].cluster_id = 1
    themes = creator_consensus(obs)
    labels = {t["label"]: t for t in themes}
    # Only the 2-distinct-creator cluster qualifies.
    assert len(themes) == 1
    assert themes[0]["founders"] == 2
    assert sorted(e["creator"] for e in themes[0]["evidence"]) == ["Nate", "Shane"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_crosscreator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'youtube_brain.observations.crosscreator'`

**Step 3: Write minimal implementation**

```python
# src/youtube_brain/observations/crosscreator.py
"""Cross-creator consensus over multiple brains' observations.

One shared implementation for both the skill bridge (`report --brains`) and the
categories API. Clusters observations across brains IN MEMORY (so a theme can
span creators without touching stored per-brain clusters) and counts consensus
at CREATOR granularity — one creator across several of their own videos is never
mistaken for agreement.
"""

from __future__ import annotations

from collections import defaultdict

from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.observations.cluster import greedy_cluster
from youtube_brain.observations.refresh import CLUSTER_THRESHOLD
from youtube_brain.observations.report import build_intelligence
from youtube_brain.storage.observations import (
    get_observation_embeddings,
    get_observations_by_brain,
    store_observation_embedding,
)


def _creator(o) -> str:
    return o.creator or o.youtube_id or "?"


def creator_consensus(observations: list) -> list[dict]:
    """Consensus themes counted by distinct CREATOR (>=2 to qualify)."""
    creators_total = len({_creator(o) for o in observations})
    clusters: dict = defaultdict(list)
    for o in observations:
        clusters[o.cluster_id if o.cluster_id is not None else -1].append(o)

    themes = []
    for obs in clusters.values():
        by_creator: dict = {}
        for o in obs:
            c = _creator(o)
            if c not in by_creator or (o.confidence or 0) > (by_creator[c].confidence or 0):
                by_creator[c] = o
        if len(by_creator) < 2:
            continue
        rep = max(obs, key=lambda o: ((o.confidence or 0), -len(o.claim)))
        evidence = [
            {
                "creator": c,
                "quote": (o.evidence_quote or "").strip(),
                "youtube_id": o.youtube_id,
                "start_time": o.start_time,
                "obs_type": o.obs_type,
            }
            for c, o in sorted(by_creator.items())
        ]
        themes.append({
            "label": rep.claim,
            "founders": len(by_creator),
            "total_founders": creators_total,
            "evidence": evidence,
        })
    themes.sort(key=lambda t: (-t["founders"], -len(t["evidence"])))
    return themes


async def cross_creator_intelligence(name: str, brain_ids: list[str], client: GeminiClient) -> dict:
    """Intelligence-shaped payload across several brains (creator-level consensus).

    Embeds any unembedded claims (zero generate), clusters in memory, then counts
    consensus by creator. rollups/by_type/total come from build_intelligence over
    the same observations; `founders` and `consensus` are overridden to be
    creator-level.
    """
    obs: list = []
    for bid in brain_ids:
        obs.extend(await get_observations_by_brain(bid))

    emb: dict[str, list[float]] = {}
    for bid in brain_ids:
        for oid, vec in await get_observation_embeddings(bid):
            emb[oid] = vec
    missing = [o for o in obs if str(o.id) not in emb]
    if missing:
        vecs = await client.embed_texts([o.claim for o in missing])
        for o, vec in zip(missing, vecs):
            await store_observation_embedding(o.id, client.embed_model, client.embed_dims, vec)
            emb[str(o.id)] = vec

    pairs = [(str(o.id), emb[str(o.id)]) for o in obs if str(o.id) in emb]
    assignments = greedy_cluster(pairs, threshold=CLUSTER_THRESHOLD)
    for o in obs:
        o.cluster_id = assignments.get(str(o.id))

    intel = build_intelligence(name, obs)
    intel["founders"] = len({_creator(o) for o in obs})
    intel["consensus"] = creator_consensus(obs)
    return intel
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_crosscreator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/youtube_brain/observations/crosscreator.py tests/test_crosscreator.py
git commit -m "feat: shared cross-creator consensus module (creator-level)"
```

---

## Task 2: Point the skill bridge at the shared module (DRY)

Replace the inline multi-brain clustering in `scripts/skill_bridge.py` with the shared module so there is one implementation.

**Files:**
- Modify: `scripts/skill_bridge.py` (`cmd_report` multi-brain branch; `_render_cross_report`)

**Step 1: Refactor `cmd_report` multi-brain branch**

In `cmd_report`, replace the `else:` (multi-brain) block that gathers obs, embeds, calls `greedy_cluster`, and assigns `cluster_id` with:

```python
    else:
        from youtube_brain.observations.crosscreator import cross_creator_intelligence
        name = " + ".join(name_map.get(b, b) for b in brain_ids)
        client = GeminiClient()
        try:
            intel = await cross_creator_intelligence(name, brain_ids, client)
        finally:
            await client.close()
        # cross_creator_intelligence already produced creator-level consensus.
```

Then update the report rendering: for the cross case, render Markdown from `intel["consensus"]` (creator-level) instead of re-clustering in `_render_cross_report`. Simplest: keep a thin `_render_cross_report_md(name, intel)` that iterates `intel["consensus"]` and `intel["by_type"]`. Single-brain path is unchanged (`recluster` + `build_report`).

**Step 2: Verify the bridge still works on real data**

Run:
```bash
python scripts/skill_bridge.py report --brains 51043dce-3f52-4369-81cf-55478aef8af0,aa374dc5-72ec-48b2-855a-d1ed114604ef --out data/_check.md
```
Expected: JSON with `cross_creator: true`, `consensus_count: 1` (Nate + Shane "AI skills pay" theme), identical to before the refactor. Then `rm data/_check.md`.

**Step 3: Run the suite**

Run: `pytest -k "not integration" --ignore=tests/test_e2e.py -q`
Expected: PASS (same count as before + Task 1's test).

**Step 4: Commit**

```bash
git add scripts/skill_bridge.py
git commit -m "refactor: bridge report uses shared cross-creator module"
```

---

## Task 3: Category config + loader

Categories live in the package (committed, not under gitignored `data/`). The loader reads them and matches creators to brains by `channel_id`.

**Files:**
- Create: `src/youtube_brain/config/categories.json`
- Create: `src/youtube_brain/categories.py`
- Test: `tests/test_categories.py`

**Step 1: Seed the config (minimal, real channel_ids we already have)**

```json
[
  {
    "slug": "making-money-with-ai",
    "name": "Making Money with AI",
    "description": "Operators on turning AI skills into income.",
    "creators": [
      { "handle": "@nateherk", "url": "https://youtube.com/@nateherk", "channel_id": "UC2ojq-nuP8ceeHqiroeKhBA" },
      { "handle": "@shanehummus", "url": "https://youtube.com/@shanehummus", "channel_id": "UCLKZ20yD2tNMBOkSDZo4FeQ" },
      { "handle": "@LiamOttley", "url": "https://youtube.com/@LiamOttley", "channel_id": null },
      { "handle": "@nicksaraev", "url": "https://youtube.com/@nicksaraev", "channel_id": null }
    ]
  }
]
```

(Starter Story's channel_id is `UChhw6DlKKTQ9mYSpTfXUYqA` — add it here if you want it in this category too; the seed script in Task 5 fills the `null`s.)

**Step 2: Write the failing test**

```python
# tests/test_categories.py
import json
import pytest
from youtube_brain.core.models import Brain, Source
from youtube_brain.core.enums import SourceType, SourceStatus
from youtube_brain.storage.brains import insert_brain
from youtube_brain.categories import load_categories, get_category, brains_by_channel_id


async def _seed_brain_with_channel(channel_id, name):
    brain = Brain(name=name)
    await insert_brain(brain)
    src = Source(brain_id=brain.id, source_type=SourceType.CHANNEL,
                 source_url="x", source_id=channel_id, status=SourceStatus.READY)
    from youtube_brain.storage.database import get_session, sources as sources_table
    from sqlalchemy.dialects.sqlite import insert
    async with get_session() as s:
        await s.execute(insert(sources_table).values(
            id=str(src.id), brain_id=str(brain.id), source_type="channel",
            source_url="x", source_id=channel_id, status="ready",
            created_at=src.created_at))
    return str(brain.id)


def test_load_categories_from_path(tmp_path):
    p = tmp_path / "categories.json"
    p.write_text(json.dumps([{"slug": "s", "name": "S", "description": "d",
                              "creators": [{"handle": "@a", "url": "u", "channel_id": "UC1"}]}]))
    cats = load_categories(p)
    assert cats[0].slug == "s"
    assert cats[0].creators[0].channel_id == "UC1"
    assert get_category("s", p).name == "S"


async def test_brains_by_channel_id_matches_pulled():
    bid = await _seed_brain_with_channel("UCZZZ", "Creator Z")
    found = await brains_by_channel_id(["UCZZZ", "UCNOPE"])
    assert "UCZZZ" in found and found["UCZZZ"]["brain_id"] == bid
    assert "UCNOPE" not in found
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_categories.py -v`
Expected: FAIL (`No module named 'youtube_brain.categories'`)

**Step 4: Write the loader**

```python
# src/youtube_brain/categories.py
"""Category config loader + creator->brain matching (no DB tables)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import text

from youtube_brain.storage.database import get_session

_CONFIG_PATH = Path(__file__).parent / "config" / "categories.json"


class Creator(BaseModel):
    handle: str
    url: str
    channel_id: str | None = None


class Category(BaseModel):
    slug: str
    name: str
    description: str = ""
    creators: list[Creator] = []


def load_categories(path: Path | None = None) -> list[Category]:
    p = path or _CONFIG_PATH
    if not p.exists():
        return []
    return [Category(**c) for c in json.loads(p.read_text(encoding="utf-8"))]


def get_category(slug: str, path: Path | None = None) -> Category | None:
    return next((c for c in load_categories(path) if c.slug == slug), None)


async def brains_by_channel_id(channel_ids: list[str]) -> dict[str, dict]:
    """{channel_id: brain summary} for channel_ids that have a pulled brain."""
    ids = [c for c in channel_ids if c]
    if not ids:
        return {}
    binds = {f"c{i}": c for i, c in enumerate(ids)}
    inc = ",".join(f":c{i}" for i in range(len(ids)))
    sql = text(
        "SELECT s.source_id AS cid, b.id AS bid, b.name AS name, b.status AS status, "
        "       b.video_count AS video_count "
        f"FROM sources s JOIN brains b ON b.id = s.brain_id WHERE s.source_id IN ({inc})"
    )
    async with get_session() as session:
        rows = (await session.execute(sql, binds)).fetchall()
    return {
        r.cid: {"brain_id": r.bid, "name": r.name, "status": r.status,
                "video_count": r.video_count}
        for r in rows
    }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_categories.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/youtube_brain/config/categories.json src/youtube_brain/categories.py tests/test_categories.py
git commit -m "feat: category config loader + channel_id->brain matching"
```

---

## Task 4: Category API endpoints

**Files:**
- Modify: `src/youtube_brain/api/routes.py` (add 3 endpoints + imports)
- Test: `tests/test_api.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_api.py` (it already has a `client` fixture that runs `init_database(tmp_settings)`):

```python
import json


async def test_categories_endpoints(client, tmp_path, monkeypatch):
    # Point the loader at a temp config with one known creator channel_id.
    cfg = tmp_path / "categories.json"
    cfg.write_text(json.dumps([{
        "slug": "ai-money", "name": "AI Money", "description": "d",
        "creators": [
            {"handle": "@a", "url": "u", "channel_id": "UCPULLED"},
            {"handle": "@b", "url": "u2", "channel_id": "UCPENDING"},
        ],
    }]))
    import youtube_brain.categories as cats_mod
    monkeypatch.setattr(cats_mod, "_CONFIG_PATH", cfg)

    # Seed one brain whose source carries UCPULLED.
    from youtube_brain.core.models import Brain, Source
    from youtube_brain.storage.brains import insert_brain
    from youtube_brain.storage.database import get_session, sources as st
    from sqlalchemy.dialects.sqlite import insert as sqlins
    brain = Brain(name="Creator A")
    await insert_brain(brain)
    async with get_session() as s:
        await s.execute(sqlins(st).values(
            id="s-a", brain_id=str(brain.id), source_type="channel", source_url="u",
            source_id="UCPULLED", status="ready", created_at=brain.created_at))

    lst = client.get("/api/categories").json()
    assert lst[0]["slug"] == "ai-money"
    assert lst[0]["creator_count"] == 2 and lst[0]["pulled_count"] == 1

    detail = client.get("/api/categories/ai-money").json()
    by_handle = {c["handle"]: c for c in detail["creators"]}
    assert by_handle["@a"]["pulled"] is True and by_handle["@a"]["brain_id"] == str(brain.id)
    assert by_handle["@b"]["pulled"] is False

    assert client.get("/api/categories/nope").status_code == 404
```

(If `test_api.py`'s `client` fixture is a `TestClient`, calls are sync as above. If it's an async httpx client, `await client.get(...)`. Match the existing fixture style in the file.)

**Step 2: Run to verify it fails**

Run: `pytest tests/test_api.py::test_categories_endpoints -v`
Expected: FAIL (404 for `/api/categories`)

**Step 3: Add the endpoints**

In `src/youtube_brain/api/routes.py`, add imports near the top:

```python
from youtube_brain.categories import brains_by_channel_id, get_category, load_categories
from youtube_brain.observations.crosscreator import cross_creator_intelligence
```

Add endpoints (after `api_intelligence`):

```python
@router.get("/api/categories")
async def api_categories():
    cats = load_categories()
    cids = [cr.channel_id for c in cats for cr in c.creators if cr.channel_id]
    resolved = await brains_by_channel_id(cids)
    return [
        {
            "slug": c.slug, "name": c.name, "description": c.description,
            "creator_count": len(c.creators),
            "pulled_count": sum(1 for cr in c.creators
                                if cr.channel_id and cr.channel_id in resolved),
        }
        for c in cats
    ]


@router.get("/api/categories/{slug}")
async def api_category(slug: str):
    cat = get_category(slug)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    resolved = await brains_by_channel_id([cr.channel_id for cr in cat.creators if cr.channel_id])
    creators = []
    for cr in cat.creators:
        b = resolved.get(cr.channel_id) if cr.channel_id else None
        if b:
            creators.append({"handle": cr.handle, "pulled": True, **b})
        else:
            creators.append({"handle": cr.handle, "url": cr.url,
                             "channel_id": cr.channel_id, "pulled": False})
    return {"slug": cat.slug, "name": cat.name, "description": cat.description,
            "creators": creators}


@router.get("/api/categories/{slug}/consensus")
async def api_category_consensus(slug: str):
    cat = get_category(slug)
    if cat is None:
        raise HTTPException(status_code=404, detail="Category not found")
    resolved = await brains_by_channel_id([cr.channel_id for cr in cat.creators if cr.channel_id])
    brain_ids = [b["brain_id"] for b in resolved.values()]
    if not brain_ids:
        return build_intelligence(cat.name, [])
    client = GeminiClient()
    try:
        return await cross_creator_intelligence(cat.name, brain_ids, client)
    finally:
        await client.close()
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_api.py::test_categories_endpoints -v`
Expected: PASS

**Step 5: Full suite + commit**

Run: `pytest -k "not integration" --ignore=tests/test_e2e.py -q` → PASS

```bash
git add src/youtube_brain/api/routes.py tests/test_api.py
git commit -m "feat: category API endpoints (list, detail, consensus)"
```

---

## Task 5: Seed script (fill channel_ids, report pulled/pending)

**Files:**
- Create: `scripts/seed_categories.py`

**Step 1: Write the script**

```python
"""Resolve channel_ids for categories.json and report pulled vs pending.

Usage:  python scripts/seed_categories.py            # report only
        python scripts/seed_categories.py --write     # also fill missing channel_ids
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from youtube_brain.categories import _CONFIG_PATH, brains_by_channel_id, load_categories
from youtube_brain.ingest.resolver import _resolve_single_video, parse_youtube_url, resolve_video_ids
from youtube_brain.storage.database import init_database

WRITE = "--write" in sys.argv


def _resolve_channel_id(url: str) -> str | None:
    try:
        metas = resolve_video_ids(parse_youtube_url(url))
        if not metas:
            return None
        full = _resolve_single_video(metas[0]["video_id"])
        return full.get("channel_id")
    except Exception:
        return None


async def main() -> None:
    await init_database()
    raw = json.loads(Path(_CONFIG_PATH).read_text(encoding="utf-8"))
    for cat in raw:
        for cr in cat["creators"]:
            if not cr.get("channel_id"):
                cid = _resolve_channel_id(cr["url"])
                if cid:
                    cr["channel_id"] = cid
                    print(f"  resolved {cr['handle']} -> {cid}")
    if WRITE:
        Path(_CONFIG_PATH).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote channel_ids back to categories.json")

    cats = load_categories()
    cids = [cr.channel_id for c in cats for cr in c.creators if cr.channel_id]
    resolved = await brains_by_channel_id(cids)
    for c in cats:
        print(f"\n{c.name}:")
        for cr in c.creators:
            state = "PULLED " if cr.channel_id and cr.channel_id in resolved else "pending"
            print(f"  [{state}] {cr.handle}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify**

Run: `python scripts/seed_categories.py`
Expected: prints "Making Money with AI" with `[PULLED ]` for @nateherk and @shanehummus, `[pending]` for the others.

**Step 3: Commit**

```bash
git add scripts/seed_categories.py
git commit -m "feat: seed_categories script (resolve channel_ids, report status)"
```

---

## Task 6: Frontend API client

**Files:**
- Modify: `frontend/src/api.ts` (append types + functions)

**Step 1: Append**

```typescript
export interface CategorySummary {
  slug: string; name: string; description: string;
  creator_count: number; pulled_count: number;
}

export interface CategoryCreator {
  handle: string; pulled: boolean;
  brain_id?: string; name?: string; status?: string; video_count?: number;
  url?: string; channel_id?: string | null;
}

export interface CategoryDetail {
  slug: string; name: string; description: string; creators: CategoryCreator[];
}

export async function listCategories(): Promise<CategorySummary[]> {
  const res = await fetch(`${BASE}/api/categories`);
  if (!res.ok) throw new Error(`Failed to list categories: ${res.statusText}`);
  return res.json();
}

export async function getCategory(slug: string): Promise<CategoryDetail> {
  const res = await fetch(`${BASE}/api/categories/${slug}`);
  if (!res.ok) throw new Error(`Failed to get category: ${res.statusText}`);
  return res.json();
}

export async function getCategoryConsensus(slug: string): Promise<Intelligence> {
  const res = await fetch(`${BASE}/api/categories/${slug}/consensus`);
  if (!res.ok) throw new Error(`Failed to load consensus: ${res.statusText}`);
  return res.json();
}
```

**Step 2: Typecheck + commit**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

```bash
git add frontend/src/api.ts
git commit -m "feat: frontend category API client"
```

---

## Task 7: Frontend views (Home → Category → Creator) + Pull card

**Files:**
- Create: `frontend/src/components/CategoryGrid.tsx`
- Create: `frontend/src/components/CategoryPage.tsx`
- Create: `frontend/src/components/PullCard.tsx`
- Modify: `frontend/src/App.tsx` (3-view router)

**Step 1: CategoryGrid** (Home)

```tsx
import { useEffect, useState } from 'react';
import { listCategories, type CategorySummary } from '../api';

export default function CategoryGrid({ onSelect }: { onSelect: (slug: string) => void }) {
  const [cats, setCats] = useState<CategorySummary[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { listCategories().then(setCats).catch(e => setErr(String(e))); }, []);
  if (err) return <div className="error">{err}</div>;
  return (
    <div className="category-grid">
      <h1>YouTube Brain</h1>
      {cats.map(c => (
        <button key={c.slug} className="category-card" onClick={() => onSelect(c.slug)}>
          <h2>{c.name}</h2>
          <p>{c.description}</p>
          <span>{c.pulled_count}/{c.creator_count} creators pulled</span>
        </button>
      ))}
    </div>
  );
}
```

**Step 2: PullCard** (pending creator → ingest)

```tsx
import { useState } from 'react';
import { ingestUrl } from '../api';

export default function PullCard({ handle, url }: { handle: string; url: string }) {
  const [state, setState] = useState<'idle' | 'pulling' | 'done' | 'error'>('idle');
  async function pull() {
    setState('pulling');
    try { await ingestUrl(url); setState('done'); }
    catch { setState('error'); }
  }
  return (
    <div className="creator-card pending">
      <h3>{handle}</h3>
      <p className="muted">not pulled yet</p>
      {state === 'idle' && <button onClick={pull}>Pull</button>}
      {state === 'pulling' && <span>ingesting…</span>}
      {state === 'done' && <span>queued — refresh shortly</span>}
      {state === 'error' && <button onClick={pull}>retry</button>}
    </div>
  );
}
```

**Step 3: CategoryPage** (creator browser + consensus tab)

```tsx
import { useEffect, useState } from 'react';
import { getCategory, getCategoryConsensus, type CategoryDetail, type Intelligence } from '../api';
import PullCard from './PullCard';
import IntelligencePanel from './IntelligencePanel';

export default function CategoryPage(
  { slug, onBack, onSelectCreator }:
  { slug: string; onBack: () => void; onSelectCreator: (brainId: string) => void },
) {
  const [cat, setCat] = useState<CategoryDetail | null>(null);
  const [tab, setTab] = useState<'creators' | 'consensus'>('creators');
  const [intel, setIntel] = useState<Intelligence | null>(null);

  useEffect(() => { getCategory(slug).then(setCat); }, [slug]);
  useEffect(() => {
    if (tab === 'consensus' && !intel) getCategoryConsensus(slug).then(setIntel);
  }, [tab, slug, intel]);

  if (!cat) return <div>Loading…</div>;
  return (
    <div className="category-page">
      <button onClick={onBack}>← categories</button>
      <h1>{cat.name}</h1>
      <div className="tabs">
        <button className={tab === 'creators' ? 'active' : ''} onClick={() => setTab('creators')}>Creators</button>
        <button className={tab === 'consensus' ? 'active' : ''} onClick={() => setTab('consensus')}>Consensus</button>
      </div>
      {tab === 'creators' && (
        <div className="creator-grid">
          {cat.creators.map(cr => cr.pulled ? (
            <button key={cr.handle} className="creator-card" onClick={() => onSelectCreator(cr.brain_id!)}>
              <h3>{cr.name}</h3>
              <span>{cr.video_count} videos · {cr.status}</span>
            </button>
          ) : (
            <PullCard key={cr.handle} handle={cr.handle} url={cr.url!} />
          ))}
        </div>
      )}
      {tab === 'consensus' && (intel
        ? <IntelligencePanel intelligence={intel} />
        : <div>Computing consensus…</div>)}
    </div>
  );
}
```

> Note: check `IntelligencePanel`'s actual prop name (e.g. `intelligence` vs `data`) and pass accordingly — read `frontend/src/components/IntelligencePanel.tsx` first.

**Step 4: App.tsx 3-view router**

```tsx
import { useState } from 'react';
import './App.css';
import CategoryGrid from './components/CategoryGrid';
import CategoryPage from './components/CategoryPage';
import BrainDetail from './components/BrainDetail';

type View =
  | { v: 'home' }
  | { v: 'category'; slug: string }
  | { v: 'creator'; brainId: string; slug: string };

function App() {
  const [view, setView] = useState<View>({ v: 'home' });
  return (
    <div className="app">
      {view.v === 'home' && (
        <CategoryGrid onSelect={(slug) => setView({ v: 'category', slug })} />
      )}
      {view.v === 'category' && (
        <CategoryPage
          slug={view.slug}
          onBack={() => setView({ v: 'home' })}
          onSelectCreator={(brainId) => setView({ v: 'creator', brainId, slug: view.slug })}
        />
      )}
      {view.v === 'creator' && (
        <BrainDetail brainId={view.brainId} onBack={() => setView({ v: 'category', slug: view.slug })} />
      )}
    </div>
  );
}

export default App;
```

**Step 5: Typecheck + minimal styles + commit**

Run (from `frontend/`): `npx tsc --noEmit` → no errors. Add minimal CSS for `.category-grid/.category-card/.creator-grid/.creator-card/.tabs` in `App.css` (copy spacing from existing `BrainList` styles).

```bash
git add frontend/src/components/CategoryGrid.tsx frontend/src/components/CategoryPage.tsx frontend/src/components/PullCard.tsx frontend/src/App.tsx frontend/src/App.css
git commit -m "feat: category Home/Category views + Pull card"
```

---

## Task 8: End-to-end smoke

**Step 1: Resolve channel_ids**

Run: `python scripts/seed_categories.py --write` (fills @LiamOttley/@nicksaraev ids), then `git add src/youtube_brain/config/categories.json && git commit -m "chore: fill category channel_ids"`.

**Step 2: Start backend + frontend**

Run: `python -m youtube_brain.cli serve` (terminal 1) and `cd frontend && npm run dev` (terminal 2).

**Step 3: Manual check**
- Home shows "Making Money with AI" with "2/4 creators pulled".
- Open it → Nate Herk + Shane Hummus as creator cards; Liam Ottley + Nick Saraev as **Pull cards**.
- Click a creator → existing `BrainDetail`.
- **Consensus** tab → renders the Nate+Shane "AI skills pay" theme via `IntelligencePanel`.

**Step 4: Final suite**

Run: `pytest -k "not integration" --ignore=tests/test_e2e.py -q` → PASS.

---

## Notes for the executor
- The autouse DB-isolation fixture means every test gets a throwaway DB — seed what each test needs.
- Cross-creator consensus is thin until creators have observations (Workflow D / `save`). The endpoint correctly returns few/zero themes otherwise — that's honest, not a bug.
- `data/` is gitignored; the category config lives in `src/youtube_brain/config/` precisely so it's committed.
- DRY: do Task 1+2 before the API so there is one cross-creator implementation.
