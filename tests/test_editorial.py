"""Tests for the editorial layer's deterministic scaffolding."""

from youtube_brain.core.models import Observation
from youtube_brain.observations.editorial import (
    build_brief,
    compute_absences,
    signal_tier,
)


def test_signal_tiers():
    assert signal_tier(5, 10) == "Strong signal"
    assert signal_tier(3, 10) == "Emerging pattern"
    assert signal_tier(1, 10) == "Outlier"
    assert signal_tier(0, 0) == "Outlier"


def _obs(yt, claim, otype="principle", value=None):
    return Observation(
        brain_id="b", youtube_id=yt, creator=yt, obs_type=otype,
        claim=claim, value=value,
    )


def test_absences_flags_unmentioned_expected_topics():
    obs = [
        _obs("v1", "Used Reddit for first users", "acquisition_channel", "Reddit"),
        _obs("v2", "Charges a monthly subscription", "monetization", "subscription"),
    ]
    absences = compute_absences(obs)
    topics = {a["topic"] for a in absences}
    # Nobody mentioned venture capital or hiring -> flagged absent.
    assert "Raising venture capital" in topics
    assert "Hiring a team" in topics
    assert all(a["founders"] == 0 for a in absences if a["topic"] == "Raising venture capital")


def test_absences_not_flagged_when_present():
    obs = [
        _obs("v1", "Raised a seed round from an investor", "tactic", "fundraising"),
        _obs("v2", "Closed a venture seed round", "tactic", "venture"),
    ]
    absences = compute_absences(obs)
    assert "Raising venture capital" not in {a["topic"] for a in absences}


def test_build_brief_structure():
    obs = [
        _obs("v1", "Copy a proven idea", "business_model"),
        _obs("v2", "Copy a proven idea and niche down", "principle"),
    ]
    brief = build_brief("Test", obs)
    assert brief["total_founders"] == 2
    assert brief["total_observations"] == 2
    assert "consensus" in brief and "rollups" in brief
    assert "absences" in brief and "opinions" in brief
