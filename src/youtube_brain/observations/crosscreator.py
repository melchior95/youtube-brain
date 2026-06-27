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


def _channel(o) -> str:
    """Stable per-CHANNEL identity for counting consensus.

    A category brain == one channel, so brain_id is the right unit. Counting by
    the free-text `creator` field is fragile: one channel's videos can carry
    inconsistent creator text (e.g. cmd_save defaults a missing creator to the
    video title), which would make a single channel look like several creators
    agreeing. brain_id can't.
    """
    return str(o.brain_id)


def _display(o) -> str:
    return o.creator or o.youtube_id or "?"


def creator_consensus(observations: list) -> list[dict]:
    """Consensus themes counted by distinct CHANNEL/creator (>=2 to qualify)."""
    channels_total = len({_channel(o) for o in observations})
    clusters: dict = defaultdict(list)
    for o in observations:
        clusters[o.cluster_id if o.cluster_id is not None else -1].append(o)

    themes = []
    for obs in clusters.values():
        by_channel: dict = {}
        for o in obs:
            key = _channel(o)
            if key not in by_channel or (o.confidence or 0) > (by_channel[key].confidence or 0):
                by_channel[key] = o
        if len(by_channel) < 2:
            continue
        rep = max(obs, key=lambda o: ((o.confidence or 0), -len(o.claim)))
        evidence = [
            {
                "creator": _display(o),
                "quote": (o.evidence_quote or "").strip(),
                "youtube_id": o.youtube_id,
                "start_time": o.start_time,
                "obs_type": o.obs_type,
            }
            for _, o in sorted(by_channel.items(), key=lambda kv: _display(kv[1]))
        ]
        themes.append({
            "label": rep.claim,
            "founders": len(by_channel),
            "total_founders": channels_total,
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
    intel["founders"] = len({_channel(o) for o in obs})
    intel["consensus"] = creator_consensus(obs)
    return intel
