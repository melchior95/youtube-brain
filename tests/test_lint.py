from youtube_brain.core.models import Observation
from youtube_brain.observations.lint import lint_candidates

B1 = "11111111-1111-1111-1111-111111111111"
B2 = "22222222-2222-2222-2222-222222222222"


def _obs(brain_id, creator, claim, yt, entity, value=None, start=10.0):
    return Observation(
        brain_id=brain_id, youtube_id=yt, creator=creator, obs_type="market",
        claim=claim, value=value, entities=[entity], start_time=start,
        evidence_quote=claim, confidence=0.9,
    )


def test_cross_source_disagreement_is_a_candidate():
    # Two creators, same entity, opposite stance -> one tension candidate.
    obs = [
        _obs(B1, "Alice", "NVDA is a buy here", "v1", "NVDA", value="bullish"),
        _obs(B2, "Bob", "NVDA is overheated", "v2", "NVDA", value="bearish"),
    ]
    dates = {"v1": "2026-01-10", "v2": "2026-01-12"}
    cands, total = lint_candidates(obs, dates)
    assert total == 1 and len(cands) == 1
    c = cands[0]
    assert c["entity"] == "NVDA"
    assert c["distinct_sources"] == 2
    assert c["values"] == ["bearish", "bullish"]
    assert [o["citation"] for o in c["observations"]] == [
        "https://youtu.be/v1?t=10", "https://youtu.be/v2?t=10"
    ]


def test_single_creator_over_time_is_a_candidate_and_date_ordered():
    # One creator/brain, same entity, two dates -> evolution candidate, oldest first.
    obs = [
        _obs(B1, "Alice", "TSLA looks weak", "vNew", "TSLA", value="bearish"),
        _obs(B1, "Alice", "TSLA to the moon", "vOld", "TSLA", value="bullish"),
    ]
    dates = {"vOld": "2025-03-01", "vNew": "2026-02-01"}
    cands, total = lint_candidates(obs, dates)
    assert total == 1
    c = cands[0]
    assert c["distinct_sources"] == 2 and c["distinct_dates"] == 2
    # Members sorted chronologically so Claude reads the flip in order.
    assert [o["published"] for o in c["observations"]] == ["2025-03-01", "2026-02-01"]


def test_lone_entity_mention_is_not_a_candidate():
    obs = [_obs(B1, "Alice", "mentioned AMD once", "v1", "AMD")]
    cands, total = lint_candidates(obs, {"v1": "2026-01-01"})
    assert total == 0 and cands == []


def test_same_source_same_date_restatement_is_not_a_candidate():
    # Same video twice -> one source, one date -> no tension.
    obs = [
        _obs(B1, "Alice", "buy NVDA", "v1", "NVDA"),
        _obs(B1, "Alice", "yeah buy NVDA", "v1", "NVDA"),
    ]
    cands, total = lint_candidates(obs, {"v1": "2026-01-01"})
    assert total == 0 and cands == []


def test_entity_normalization_groups_case_and_whitespace():
    obs = [
        _obs(B1, "Alice", "bullish nvda", "v1", "NVDA"),
        _obs(B2, "Bob", "bearish nvda", "v2", "  nvda  "),
    ]
    cands, total = lint_candidates(obs, {"v1": "2026-01-01", "v2": "2026-01-02"})
    assert total == 1
    assert cands[0]["distinct_sources"] == 2


def test_missing_start_time_yields_no_citation():
    obs = [
        _obs(B1, "Alice", "buy NVDA", "v1", "NVDA", start=None),
        _obs(B2, "Bob", "sell NVDA", "v2", "NVDA", start=None),
    ]
    cands, _ = lint_candidates(obs, {"v1": "2026-01-01", "v2": "2026-01-02"})
    assert all(o["citation"] is None for o in cands[0]["observations"])


def test_max_groups_caps_but_reports_total():
    obs = []
    for i in range(5):
        obs.append(_obs(B1, "Alice", f"claim {i}", f"a{i}", f"E{i}", value="x"))
        obs.append(_obs(B2, "Bob", f"counter {i}", f"b{i}", f"E{i}", value="y"))
    dates = {o.youtube_id: f"2026-01-{i:02d}" for i, o in enumerate(obs, 1)}
    cands, total = lint_candidates(obs, dates, max_groups=3)
    assert total == 5 and len(cands) == 3
