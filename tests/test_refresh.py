"""Tests for the watchlist changelog diff (pure logic)."""

from youtube_brain.observations.refresh import diff_intelligence


def _intel(total, founders, rollups):
    return {"total_observations": total, "founders": founders, "rollups": rollups}


def test_diff_counts_new_observations_and_founders():
    before = _intel(100, 9, {})
    after = _intel(115, 10, {})
    d = diff_intelligence(before, after)
    assert d["new_observations"] == 15
    assert d["new_founders"] == 1


def test_diff_detects_added_creator_to_existing_entity():
    before = _intel(10, 2, {"Tools": [
        {"value": "Supabase", "founders": 2,
         "evidence": [{"creator": "A"}, {"creator": "B"}]},
    ]})
    after = _intel(13, 3, {"Tools": [
        {"value": "Supabase", "founders": 3,
         "evidence": [{"creator": "A"}, {"creator": "B"}, {"creator": "C"}]},
    ]})
    d = diff_intelligence(before, after)
    chg = d["rollup_changes"]
    assert len(chg) == 1
    assert chg[0]["value"] == "Supabase"
    assert chg[0]["before"] == 2 and chg[0]["after"] == 3
    assert chg[0]["new_creators"] == ["C"]
    assert chg[0]["is_new"] is False


def test_diff_detects_brand_new_entity():
    before = _intel(10, 2, {"Channels": []})
    after = _intel(12, 3, {"Channels": [
        {"value": "Reddit", "founders": 1, "evidence": [{"creator": "C"}]},
    ]})
    d = diff_intelligence(before, after)
    assert d["rollup_changes"][0]["value"] == "Reddit"
    assert d["rollup_changes"][0]["is_new"] is True


def test_diff_no_changes():
    intel = _intel(10, 2, {"Tools": [
        {"value": "Cursor", "founders": 1, "evidence": [{"creator": "A"}]},
    ]})
    d = diff_intelligence(intel, intel)
    assert d["new_observations"] == 0
    assert d["rollup_changes"] == []
