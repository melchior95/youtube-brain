"""Cross-creator consensus over multiple brains' observations.

Consensus is entity-based and deterministic: observations are grouped by the
canonical `entities` Claude extracted at save time, and a theme counts only when
>= 2 distinct CREATORS (channels) land on the same entity. One creator across
several of their own videos is never mistaken for agreement. No embeddings, no
external API: the counts are computed from persisted fields, never guessed.
"""

from __future__ import annotations

from youtube_brain.observations.report import build_intelligence, consensus_themes
from youtube_brain.storage.observations import get_observations_by_brain


def _channel(o) -> str:
    """Stable per-CHANNEL identity for counting consensus.

    A category brain == one channel, so brain_id is the right unit. Counting by
    the free-text `creator` field is fragile: one channel's videos can carry
    inconsistent creator text (e.g. cmd_save defaults a missing creator to the
    video title), which would make a single channel look like several creators
    agreeing. brain_id can't.
    """
    return str(o.brain_id)


def creator_consensus(observations: list) -> list[dict]:
    """Consensus themes counted by distinct CHANNEL (>=2 to qualify).

    Same entity-grouping as the single-brain report, but the unit of agreement is
    the channel (brain_id), not the individual video.
    """
    return consensus_themes(observations, ident=_channel)


async def cross_creator_intelligence(name: str, brain_ids: list[str]) -> dict:
    """Intelligence-shaped payload across several brains (creator-level consensus).

    Pools the brains' persisted observations and counts consensus by channel.
    rollups/by_type/total come from build_intelligence; `founders` and
    `consensus` are overridden to be channel-level. Zero external API.
    """
    obs: list = []
    for bid in brain_ids:
        obs.extend(await get_observations_by_brain(bid))

    intel = build_intelligence(name, obs)
    intel["founders"] = len({_channel(o) for o in obs})
    intel["consensus"] = creator_consensus(obs)
    return intel
