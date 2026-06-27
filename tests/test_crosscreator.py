from youtube_brain.core.models import Observation
from youtube_brain.observations.crosscreator import creator_consensus

B1 = "11111111-1111-1111-1111-111111111111"
B2 = "22222222-2222-2222-2222-222222222222"


def _obs(brain_id, creator, claim, yt, entity, conf=0.9):
    return Observation(brain_id=brain_id, youtube_id=yt, creator=creator,
                       obs_type="principle", claim=claim, entities=[entity],
                       confidence=conf)


def test_consensus_counts_distinct_channels():
    # Two different channels (brains) land on the same entity -> real consensus.
    # The same channel repeating an entity across its own videos does not.
    obs = [
        _obs(B1, "Nate Herk", "AI skills pay a lot", "v1", "AI skills"),
        _obs(B2, "Shane Hummus", "AI skills can pay more than a degree", "v2", "AI skills"),
        _obs(B2, "Shane Hummus", "pick a niche", "v3", "niche"),
        _obs(B2, "Shane Hummus", "pick a niche", "v4", "niche"),
    ]
    themes = creator_consensus(obs)
    assert len(themes) == 1
    assert themes[0]["founders"] == 2
    assert sorted(e["creator"] for e in themes[0]["evidence"]) == ["Nate Herk", "Shane Hummus"]


def test_one_channel_with_inconsistent_creator_text_is_not_consensus():
    # Same channel (one brain), but per-video `creator` text differs (e.g.
    # cmd_save defaulted a missing creator to the video title). This MUST NOT be
    # mistaken for multiple creators agreeing, identity is the channel, not text.
    obs = [
        _obs(B1, "Video A title", "pick a niche", "v1", "niche"),
        _obs(B1, "Video B title", "pick a niche", "v2", "niche"),
    ]
    assert creator_consensus(obs) == []
