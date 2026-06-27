"""Zero-generate bridge for the global `youtube-brain` Claude skill.

Exposes three subcommands that a Claude-in-loop skill drives. NONE of them
spend Gemini *generate* quota (the free tier is only ~20 generate calls/day);
they use transcript fetching + chunking + the much-larger *embed* quota, and
leave the summarization / question-answering to Claude in-session.

  pull <url> [--limit N] [--max-chars N]
      Lite-ingest a video / channel / playlist (transcript + chunk + embed,
      ZERO generate). Single videos are grouped under their channel's brain so
      the channel accumulates. Prints JSON with the targeted videos' transcripts
      for Claude to summarize.

  brains
      List existing brains (id, name, video_count, status) as JSON so Claude can
      match a question to data already pulled.

  context <brain_id> "<question>" [--k N]
      Dense (embedding) + FTS retrieval over a brain, one embed call, zero
      generate. Prints the top-k timestamped chunks with youtu.be citations for
      Claude to synthesize a cited answer.

All human-readable logging goes to stderr; stdout is always a single JSON value.

Run from the project root (the DB path in settings is relative):
    python scripts/skill_bridge.py <subcommand> ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

from sqlalchemy import text

from youtube_brain.core.models import Observation
from youtube_brain.embed import cosine, embed_query
from youtube_brain.ingest.pull import pull_creator
from youtube_brain.observations.crosscreator import cross_creator_intelligence
from youtube_brain.observations.extractor import attribute
from youtube_brain.observations.lint import lint_candidates
from youtube_brain.observations.report import build_intelligence, build_report
from youtube_brain.storage.brains import list_brains
from youtube_brain.storage.database import get_session, init_database
from youtube_brain.storage.observations import (
    get_observations_by_brain,
    insert_observations,
)

# Default caps, bound cost on channels and keep stdout manageable for Claude.
DEFAULT_MAX_CHARS = 0  # 0 = no truncation: hand Claude the full transcript to summarize
DEFAULT_K = 12


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(obj) -> None:
    """Write a single JSON value to stdout (utf-8, non-ascii preserved)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _citation(youtube_id: str, start: float) -> str:
    return f"https://youtu.be/{youtube_id}?t={int(start)}"


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


async def _transcripts_for(brain_id: str, youtube_ids: list[str], max_chars: int) -> list[dict]:
    """Read back the (possibly truncated) transcripts for the targeted videos."""
    if not youtube_ids:
        return []
    placeholders = ",".join(f":id{i}" for i in range(len(youtube_ids)))
    params = {f"id{i}": yid for i, yid in enumerate(youtube_ids)}
    params["bid"] = brain_id
    sql = text(
        "SELECT v.video_id, v.title, v.channel_name, v.transcript_clean, "
        "       v.transcript_source, v.caption_kind, v.published_at, "
        "       (SELECT COUNT(*) FROM chunks c WHERE c.video_id = v.id) AS n_chunks "
        "FROM videos v "
        f"WHERE v.brain_id = :bid AND v.video_id IN ({placeholders}) "
        "ORDER BY v.created_at"
    )
    out: list[dict] = []
    async with get_session() as session:
        rows = (await session.execute(sql, params)).fetchall()
    for r in rows:
        transcript = r.transcript_clean or ""
        truncated = max_chars > 0 and len(transcript) > max_chars
        out.append(
            {
                "youtube_id": r.video_id,
                "title": r.title,
                "channel_name": r.channel_name,
                "published": str(r.published_at)[:10] if r.published_at else None,
                "url": f"https://youtu.be/{r.video_id}",
                "n_chunks": r.n_chunks,
                "transcript_source": r.transcript_source,
                "caption_kind": r.caption_kind,
                "transcript_truncated": truncated,
                "transcript_chars": len(transcript),
                "transcript": transcript[:max_chars] if truncated else transcript,
            }
        )
    return out


async def cmd_pull(url: str, limit: int | None, max_chars: int, topic_brain: str | None = None) -> None:
    _log(f"[pull] resolving: {url}" + (f"  -> topic '{topic_brain}'" if topic_brain else ""))
    result = await pull_creator(url, limit, topic_brain=topic_brain)
    if result.get("error"):
        _emit(result)
        return

    _log(f"[pull] brain='{result['brain_name']}'  "
         f"videos={len(result['targeted_ids'])}  (zero generate)")
    videos = await _transcripts_for(result["brain_id"], result["targeted_ids"], max_chars)
    _emit(
        {
            "brain_id": result["brain_id"],
            "brain_name": result["brain_name"],
            "channel_id": result["channel_id"],
            "source_type": result["source_type"],
            "videos_found": result["videos_found"],
            "videos_processed": result["videos_processed"],
            "chunks_created": result["chunks_created"],
            "errors": result["errors"],
            "videos": videos,
            "note": "Zero Gemini generate calls used. Summarize the transcript(s) above in-session.",
        }
    )


# ---------------------------------------------------------------------------
# search, discover candidate videos for a research question (zero generate)
# ---------------------------------------------------------------------------


# YouTube "Upload date" search-filter codes (the `sp=` param). Lets YouTube
# return only recent videos, important for fast-moving topics where old advice
# is stale.
_SP_UPLOAD_DATE = {
    "today": "EgIIAg%3D%3D",
    "week": "EgIIAw%3D%3D",
    "month": "EgIIBA%3D%3D",
    "year": "EgIIBQ%3D%3D",
}


async def cmd_search(query: str, limit: int, recent: str | None = None) -> None:
    """Search YouTube for candidate videos (yt-dlp, no API, zero generate).

    With ``recent`` (today/week/month/year), uses YouTube's upload-date filter so
    only recent videos come back. Returns candidates (id, title, channel,
    duration, views) for Claude to curate; the chosen ones are then pulled into a
    topic brain via `pull --brain`.
    """
    n = max(1, limit)
    if recent in _SP_UPLOAD_DATE:
        url = ("https://www.youtube.com/results?search_query="
               f"{urllib.parse.quote_plus(query)}&sp={_SP_UPLOAD_DATE[recent]}")
        cmd = ["yt-dlp", url, "--flat-playlist", "--dump-json", "--no-warnings",
               "--quiet", "--playlist-end", str(n)]
    else:
        cmd = ["yt-dlp", f"ytsearch{n}:{query}", "--flat-playlist",
               "--dump-json", "--no-warnings", "--quiet"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception as exc:
        _emit({"error": f"search failed: {exc}", "query": query})
        return
    results = []
    for line in out.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        vid = e.get("id")
        if not vid:
            continue
        dur = e.get("duration")
        results.append({
            "youtube_id": vid,
            "title": e.get("title"),
            "channel": e.get("channel") or e.get("uploader"),
            "duration_min": round(dur / 60, 1) if dur else None,
            "view_count": e.get("view_count"),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    _emit({"query": query, "recent": recent, "count": len(results), "results": results})


# ---------------------------------------------------------------------------
# brains
# ---------------------------------------------------------------------------


async def cmd_brains() -> None:
    await init_database()
    brains = await list_brains()
    _emit(
        [
            {
                "id": str(b.id),
                "name": b.name,
                "video_count": b.video_count,
                "status": b.status.value,
            }
            for b in brains
        ]
    )


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


async def _fts_chunk_ids(brain_ids: list[str], query: str, limit: int) -> set[str]:
    """Chunk ids matching the query lexically, for a small dense-rank boost."""
    terms = [t.replace('"', "") for t in query.strip().split() if t.strip()]
    if not terms:
        return set()
    fts_query = " OR ".join(f'"{t}"' for t in terms)
    binds = {f"b{i}": b for i, b in enumerate(brain_ids)}
    inclause = ",".join(f":b{i}" for i in range(len(brain_ids)))
    binds["q"] = fts_query
    binds["lim"] = limit
    sql = text(
        "SELECT c.id FROM chunks c "
        "JOIN chunks_fts ON chunks_fts.rowid = c.rowid "
        f"WHERE chunks_fts MATCH :q AND c.brain_id IN ({inclause}) "
        "ORDER BY bm25(chunks_fts) LIMIT :lim"
    )
    async with get_session() as session:
        try:
            rows = (await session.execute(sql, binds)).fetchall()
        except Exception as exc:  # malformed FTS query
            _log(f"[context] FTS boost skipped: {exc}")
            return set()
    return {r.id for r in rows}


async def _fts_ranked(brain_ids: list[str], query: str, limit: int) -> list:
    """FTS5/BM25 keyword search over chunks across brains, best match first."""
    terms = [t.replace('"', "") for t in query.strip().split() if t.strip()]
    if not terms:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in terms)
    binds = {f"b{i}": b for i, b in enumerate(brain_ids)}
    inclause = ",".join(f":b{i}" for i in range(len(brain_ids)))
    binds["q"] = fts_query
    binds["lim"] = limit
    sql = text(
        "SELECT c.id, c.brain_id AS bid, c.start_time, c.text, v.video_id AS yt, "
        "       v.title, v.channel_name AS creator, b.name AS brain_name, "
        "       bm25(chunks_fts) AS bm25 "
        "FROM chunks c "
        "JOIN chunks_fts ON chunks_fts.rowid = c.rowid "
        "JOIN videos v ON v.id = c.video_id "
        "JOIN brains b ON b.id = c.brain_id "
        f"WHERE chunks_fts MATCH :q AND c.brain_id IN ({inclause}) "
        "ORDER BY bm25(chunks_fts) LIMIT :lim"
    )
    async with get_session() as session:
        try:
            return (await session.execute(sql, binds)).fetchall()
        except Exception as exc:  # malformed FTS query
            _log(f"[context] FTS query failed: {exc}")
            return []


async def _dense_scored(brain_ids: list[str], query: str, k: int):
    """Cosine over stored chunk embeddings + a small lexical boost.

    Returns (scored, chunks_searched), or (None, 0) when no embeddings exist for
    these brains yet so the caller can fall back to lexical retrieval.
    """
    binds = {f"b{i}": b for i, b in enumerate(brain_ids)}
    inclause = ",".join(f":b{i}" for i in range(len(brain_ids)))
    sql = text(
        "SELECT c.id, c.brain_id AS bid, c.start_time, c.text, v.video_id AS yt, "
        "       v.title, v.channel_name AS creator, b.name AS brain_name, e.embedding "
        "FROM chunks c "
        "JOIN videos v ON v.id = c.video_id "
        "JOIN brains b ON b.id = c.brain_id "
        "JOIN chunk_embeddings e ON e.chunk_id = c.id "
        f"WHERE c.brain_id IN ({inclause})"
    )
    async with get_session() as session:
        rows = (await session.execute(sql, binds)).fetchall()
    if not rows:
        return None, 0

    qvec = await asyncio.to_thread(embed_query, query)
    fts_hits = await _fts_chunk_ids(brain_ids, query, k * 3)
    scored = []
    for r in rows:
        try:
            vec = json.loads(r.embedding)
        except Exception:
            continue
        sim = cosine(qvec, vec)
        if r.id in fts_hits:
            sim += 0.05  # small lexical boost for exact keyword matches
        scored.append((sim, r))
    return scored, len(rows)


async def cmd_context(brain_ids: list[str], query: str, k: int, all_brains: bool = False) -> None:
    await init_database()

    if all_brains:
        brain_ids = [str(b.id) for b in await list_brains()]
    if not brain_ids:
        _emit({"error": "no_brain_selected"})
        return

    # Dense (local embedding) + lexical boost when embeddings exist; otherwise
    # fall back to pure FTS5/BM25 (e.g. a brain whose embedding backfill hasn't
    # run). Both paths are zero external API.
    scored, chunks_searched = await _dense_scored(brain_ids, query, k)
    mode = "dense+fts"
    if scored is None:
        rows = await _fts_ranked(brain_ids, query, max(k * 4, 40))
        if not rows:
            _emit({
                "brain_ids": brain_ids, "query": query, "chunks_searched": 0, "results": [],
                "note": "No matches. Try broader/different terms, or `pull` the video "
                        "and read the full transcript.",
            })
            return
        scored = [(-(r.bm25 or 0.0), r) for r in rows]
        chunks_searched = len(rows)
        mode = "lexical"

    scored.sort(key=lambda t: t[0], reverse=True)

    # Across multiple brains, a global top-k gets swamped by the largest / most
    # on-topic corpus, defeating cross-creator synthesis. Give each brain a fair
    # share (its own best chunks), then backfill to k with the global remainder.
    n = len(brain_ids)
    if n > 1:
        per = max(1, k // n)
        counts: dict[str, int] = {}
        seen: set[int] = set()
        top: list = []
        for sc, r in scored:
            if counts.get(r.bid, 0) < per:
                counts[r.bid] = counts.get(r.bid, 0) + 1
                seen.add(id(r))
                top.append((sc, r))
        for sc, r in scored:  # backfill remaining slots with the global best
            if len(top) >= k:
                break
            if id(r) not in seen:
                seen.add(id(r))
                top.append((sc, r))
        top.sort(key=lambda t: t[0], reverse=True)
        top = top[:k]
    else:
        top = scored[:k]

    _emit(
        {
            "brain_ids": brain_ids,
            "query": query,
            "retrieval": mode,
            "chunks_searched": chunks_searched,
            "results": [
                {
                    # Fall back to the brain name (channel brains are creator-named)
                    # when a video's channel_name wasn't captured at ingest.
                    "creator": r.creator or r.brain_name,
                    "brain": r.brain_name,
                    "youtube_id": r.yt,
                    "video_title": r.title,
                    "start_seconds": int(r.start_time),
                    "citation": _citation(r.yt, r.start_time),
                    "score": round(sc, 4),
                    "text": r.text,
                }
                for sc, r in top
            ],
        }
    )


# ---------------------------------------------------------------------------
# save  (write-back), persist Claude-extracted summaries + observations
# ---------------------------------------------------------------------------


async def _brain_video_index(brain_id: str) -> dict:
    """{youtube_id: {uuid, title, chunks:[{id,start_time,text}]}} for attribution."""
    out: dict = {}
    async with get_session() as session:
        vrows = (await session.execute(
            text("SELECT id, video_id, title FROM videos WHERE brain_id = :b"),
            {"b": brain_id},
        )).fetchall()
        for vr in vrows:
            crows = (await session.execute(
                text("SELECT id, start_time, text FROM chunks WHERE video_id = :v "
                     "ORDER BY start_time"),
                {"v": vr.id},
            )).fetchall()
            out[vr.video_id] = {
                "uuid": vr.id,
                "title": vr.title,
                "chunks": [{"id": c.id, "start_time": c.start_time, "text": c.text}
                           for c in crows],
            }
    return out


async def cmd_save(path: str) -> None:
    """Persist Claude's in-session summaries + observations back into a brain.

    Input JSON: {"brain_id": "...", "videos": [{"youtube_id", "summary"?,
    "creator"?, "observations": [{type, claim, entity?, value?, evidence_quote?,
    confidence?}]}]}. Evidence quotes are attributed to chunks (recovering a
    youtu.be?t= citation), and claims are embedded, ZERO generate. This is what
    makes a brain COMPOUND instead of re-deriving every query; the persisted
    observations feed the existing observations report machinery.
    """
    await init_database()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    brain_id = data["brain_id"]
    vindex = await _brain_video_index(brain_id)

    models: list[Observation] = []
    summaries_set = 0
    attributed = 0
    for entry in data.get("videos", []):
        yt = entry.get("youtube_id")
        v = vindex.get(yt)
        if not v:
            _log(f"[save] no ingested video for {yt}; skipping")
            continue
        if entry.get("summary"):
            async with get_session() as session:
                await session.execute(
                    text("UPDATE videos SET video_summary = :s, status = 'summarized' "
                         "WHERE id = :i"),
                    {"s": entry["summary"], "i": v["uuid"]},
                )
            summaries_set += 1
        for o in entry.get("observations", []):
            if not o.get("claim"):
                continue
            attr = attribute(o.get("evidence_quote", ""), v["chunks"])
            if attr:
                attributed += 1
            models.append(Observation(
                brain_id=brain_id,
                video_id=v["uuid"],
                youtube_id=yt,
                creator=entry.get("creator") or v["title"] or yt,
                obs_type=o.get("type", "other"),
                claim=o["claim"],
                value=o.get("value") or None,
                entities=[o["entity"]] if o.get("entity") else [],
                evidence_quote=o.get("evidence_quote"),
                chunk_id=attr["chunk_id"] if attr else None,
                start_time=attr["start_time"] if attr else None,
                confidence=o.get("confidence"),
                domain=o.get("domain", "founders"),
            ))

    inserted = await insert_observations(models)

    _emit({
        "brain_id": brain_id,
        "summaries_updated": summaries_set,
        "observations_inserted": inserted,
        "chunk_attributed": attributed,
        "note": "Zero API. Observations persisted with citations; re-read with "
                "`observations`, or build the consensus `report` / `lint`.",
    })


async def cmd_observations(brain_ids: list[str], all_brains: bool, limit: int) -> None:
    """Read persisted observations (with citations), the durable, re-usable layer."""
    await init_database()
    if all_brains:
        brain_ids = [str(b.id) for b in await list_brains()]
    if not brain_ids:
        _emit({"error": "no_brain_selected"})
        return

    binds = {f"b{i}": b for i, b in enumerate(brain_ids)}
    inclause = ",".join(f":b{i}" for i in range(len(brain_ids)))
    binds["lim"] = limit
    sql = text(
        "SELECT o.creator, o.obs_type, o.claim, o.value, o.youtube_id, o.start_time, "
        "       o.confidence, o.evidence_quote, b.name AS brain_name "
        "FROM observations o JOIN brains b ON b.id = o.brain_id "
        f"WHERE o.brain_id IN ({inclause}) ORDER BY o.created_at LIMIT :lim"
    )
    async with get_session() as session:
        rows = (await session.execute(sql, binds)).fetchall()

    _emit({
        "brain_ids": brain_ids,
        "count": len(rows),
        "observations": [
            {
                "creator": r.creator,
                "brain": r.brain_name,
                "type": r.obs_type,
                "claim": r.claim,
                "value": r.value,
                "confidence": r.confidence,
                "citation": _citation(r.youtube_id, r.start_time) if r.start_time is not None else None,
                "evidence_quote": r.evidence_quote,
            }
            for r in rows
        ],
    })


# ---------------------------------------------------------------------------
# report, deterministic consensus/disagreement from persisted observations
# ---------------------------------------------------------------------------


def _render_cross_report(name: str, intel: dict) -> str:
    """Render the cross-creator Markdown report from a shared Intelligence dict.

    Consensus is counted by CREATOR (a theme only qualifies when >=2 distinct
    creators independently land in it), so one creator across several of their
    own videos is never mistaken for agreement. Clustering already happened in
    `cross_creator_intelligence`; this only renders the result."""
    themes = intel["consensus"]
    n_c = intel["founders"]

    lines = [
        f"# {name}: Cross-Creator Intelligence Report", "",
        f"*Consensus across {n_c} creators, from {intel['total_observations']} "
        f"attributable observations.*", "",
        "## Consensus: what multiple creators independently said", "",
    ]
    if not themes:
        lines.append("_No theme reached ≥2 distinct creators._")
    for theme in themes:
        lines.append(f"### {theme['label']}: {theme['founders']}/{n_c} creators")
        for e in theme["evidence"]:
            q = (e["quote"] or "").strip()
            if len(q) > 160:
                q = q[:157] + "…"
            if e["start_time"] is not None:
                m, s = divmod(int(e["start_time"]), 60)
                loc = f" ([{m}:{s:02d}]({_citation(e['youtube_id'], e['start_time'])}))"
            else:
                loc = ""
            lines.append(f'- **{e["creator"]}**: "{q}"{loc}')
        lines.append("")

    lines += ["## Observation coverage by type", ""]
    for t, n in intel["by_type"].items():
        lines.append(f"- **{t}**: {n}")
    lines.append("")
    return "\n".join(lines)


async def cmd_report(brain_ids: list[str], all_brains: bool, out_path: str | None) -> None:
    """Build an Intelligence Report (consensus + rollups) from saved observations.

    Counts are computed deterministically from the observations' entities, never
    asked of an LLM (mirrors the project's trust rule): a theme is an entity that
    >= 2 distinct sources land on (videos for a single brain, channels across
    brains). Zero API.
    """
    await init_database()
    all_list = await list_brains()
    name_map = {str(b.id): b.name for b in all_list}
    if all_brains:
        brain_ids = [str(b.id) for b in all_list]
    brain_ids = [b for b in brain_ids if b]
    if not brain_ids:
        _emit({"error": "no_brain_selected"})
        return

    out = out_path or "data/report.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cross = len(brain_ids) > 1

    if not cross:
        bid = brain_ids[0]
        obs = await get_observations_by_brain(bid)
        name = name_map.get(bid, bid)
        intel = build_intelligence(name, obs)
        Path(out).write_text(build_report(name, obs), encoding="utf-8")
        sources = intel["founders"]
        top = [
            {"label": t["label"], "sources": t["founders"],
             "creators": sorted({e["creator"] for e in t["evidence"]})}
            for t in intel["consensus"]
        ][:10]
        consensus_count = len(top)
    else:
        name = " + ".join(name_map.get(b, b) for b in brain_ids)
        intel = await cross_creator_intelligence(name, brain_ids)
        Path(out).write_text(_render_cross_report(name, intel), encoding="utf-8")
        sources = intel["founders"]
        top = [
            {"label": t["label"], "creators": sorted({e["creator"] for e in t["evidence"]})}
            for t in intel["consensus"]
        ][:10]
        consensus_count = len(intel["consensus"])

    _emit({
        "name": name,
        "report_path": out,
        "cross_creator": cross,
        "total_observations": intel["total_observations"],
        "sources": sources,
        "consensus_count": consensus_count,
        "top_consensus": top,
    })


# ---------------------------------------------------------------------------
# lint, contradiction / evolution / staleness candidates (dual of report)
# ---------------------------------------------------------------------------


async def cmd_lint(brain_ids: list[str], all_brains: bool, max_groups: int) -> None:
    """Surface tension candidates from persisted observations for Claude to judge.

    The deterministic narrowing step: groups observations by shared entity, joins
    each video's published date, and emits only entity groups with real conflict
    potential (>=2 distinct sources, or spanning >=2 dates). Claude then reads the
    candidates and labels each contradiction / evolution / stale / consistent and
    writes a cited lint report. Zero generate (a DB read + pure regrouping).
    """
    await init_database()
    all_list = await list_brains()
    name_map = {str(b.id): b.name for b in all_list}
    if all_brains:
        brain_ids = [str(b.id) for b in all_list]
    brain_ids = [b for b in brain_ids if b]
    if not brain_ids:
        _emit({"error": "no_brain_selected"})
        return

    observations: list[Observation] = []
    for bid in brain_ids:
        observations.extend(await get_observations_by_brain(bid))
    if not observations:
        _emit({
            "brain_ids": brain_ids, "total_observations": 0,
            "candidate_count": 0, "candidates": [],
            "note": "No observations yet, run Workflow D (save) first.",
        })
        return

    # youtube_id -> published date ("YYYY-MM-DD"), for chronological ordering.
    binds = {f"b{i}": b for i, b in enumerate(brain_ids)}
    inclause = ",".join(f":b{i}" for i in range(len(brain_ids)))
    async with get_session() as session:
        rows = (await session.execute(
            text(f"SELECT video_id, published_at FROM videos WHERE brain_id IN ({inclause})"),
            binds,
        )).fetchall()
    date_map = {r.video_id: (str(r.published_at)[:10] if r.published_at else None) for r in rows}

    candidates, total = lint_candidates(observations, date_map, max_groups=max_groups)
    if total > len(candidates):
        _log(f"lint: {total} tension candidates; emitting top {len(candidates)} "
             f"(raise --max-groups to see more)")

    _emit({
        "brain_ids": brain_ids,
        "names": [name_map.get(b, b) for b in brain_ids],
        "scope": "cross-creator" if len(brain_ids) > 1 else "single-brain",
        "total_observations": len(observations),
        "candidate_count": len(candidates),
        "candidates_total": total,
        "candidates": candidates,
        "note": "Tension candidates grouped by shared entity. Adjudicate each: "
                "CONTRADICTION (2+ creators clash), EVOLUTION (one creator's stance "
                "changed across dates), STALE (newer supersedes older), or CONSISTENT "
                "(drop). Cite both sides with the youtu.be links.",
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zero-generate bridge for the youtube-brain skill.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pull", help="Lite-ingest a URL and return transcripts to summarize.")
    pp.add_argument("url")
    pp.add_argument("--limit", type=int, default=None, help="Max videos for channel/playlist pulls.")
    pp.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Per-video transcript cap.")
    pp.add_argument("--brain", default=None,
                    help="Collect into a named TOPIC brain (question-driven research) "
                         "instead of the creator's channel brain.")

    qp = sub.add_parser("search", help="Search YouTube for candidate videos (zero generate, no API).")
    qp.add_argument("query")
    qp.add_argument("--limit", type=int, default=15, help="Max search results.")
    qp.add_argument("--recent", choices=["today", "week", "month", "year"], default=None,
                    help="Only videos uploaded within this window (YouTube upload-date filter).")

    sub.add_parser("brains", help="List existing brains as JSON.")

    cp = sub.add_parser("context", help="Retrieve top chunks for a question (zero generate).")
    cp.add_argument("brain_id", nargs="?", default=None,
                    help="Single brain id. Omit when using --brains or --all.")
    cp.add_argument("query")
    cp.add_argument("--brains", default=None,
                    help="Comma-separated brain ids to search ACROSS for cross-creator synthesis.")
    cp.add_argument("--all", action="store_true", help="Search across every brain.")
    cp.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to return.")

    sp = sub.add_parser("save", help="Persist Claude-extracted summaries + observations (write-back).")
    sp.add_argument("path", help="Path to the observations JSON Claude wrote.")

    op = sub.add_parser("observations", help="Read persisted observations with citations.")
    op.add_argument("brain_id", nargs="?", default=None, help="Single brain id. Omit with --brains/--all.")
    op.add_argument("--brains", default=None, help="Comma-separated brain ids.")
    op.add_argument("--all", action="store_true", help="Across every brain.")
    op.add_argument("--limit", type=int, default=200, help="Max observations to return.")

    rp = sub.add_parser("report", help="Build a consensus/disagreement intelligence report.")
    rp.add_argument("brain_id", nargs="?", default=None, help="Single brain id. Omit with --brains/--all.")
    rp.add_argument("--brains", default=None, help="Comma-separated brain ids (cross-creator).")
    rp.add_argument("--all", action="store_true", help="Across every brain.")
    rp.add_argument("--out", default=None, help="Markdown output path (default data/report.md).")

    lp = sub.add_parser("lint", help="Surface contradiction/evolution/staleness candidates to judge.")
    lp.add_argument("brain_id", nargs="?", default=None, help="Single brain id. Omit with --brains/--all.")
    lp.add_argument("--brains", default=None, help="Comma-separated brain ids (cross-creator).")
    lp.add_argument("--all", action="store_true", help="Across every brain.")
    lp.add_argument("--max-groups", type=int, default=40, help="Max tension groups to emit.")

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "pull":
        asyncio.run(cmd_pull(args.url, args.limit, args.max_chars, topic_brain=args.brain))
    elif args.cmd == "search":
        asyncio.run(cmd_search(args.query, args.limit, recent=args.recent))
    elif args.cmd == "brains":
        asyncio.run(cmd_brains())
    elif args.cmd == "context":
        if args.brains:
            ids = [b.strip() for b in args.brains.split(",") if b.strip()]
        elif args.brain_id:
            ids = [args.brain_id]
        else:
            ids = []
        asyncio.run(cmd_context(ids, args.query, args.k, all_brains=args.all))
    elif args.cmd == "save":
        asyncio.run(cmd_save(args.path))
    elif args.cmd == "observations":
        if args.brains:
            ids = [b.strip() for b in args.brains.split(",") if b.strip()]
        elif args.brain_id:
            ids = [args.brain_id]
        else:
            ids = []
        asyncio.run(cmd_observations(ids, args.all, args.limit))
    elif args.cmd == "report":
        if args.brains:
            ids = [b.strip() for b in args.brains.split(",") if b.strip()]
        elif args.brain_id:
            ids = [args.brain_id]
        else:
            ids = []
        asyncio.run(cmd_report(ids, args.all, args.out))
    elif args.cmd == "lint":
        if args.brains:
            ids = [b.strip() for b in args.brains.split(",") if b.strip()]
        elif args.brain_id:
            ids = [args.brain_id]
        else:
            ids = []
        asyncio.run(cmd_lint(ids, args.all, args.max_groups))


if __name__ == "__main__":
    main()
