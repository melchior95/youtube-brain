"""Tests for the cosine helper (the embedder itself needs a model download)."""

import pytest

from youtube_brain.embed import cosine


def test_cosine_identical():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert cosine([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_mismatched_dims():
    assert cosine([1, 0, 0], [1, 0]) == 0.0


def test_cosine_arbitrary():
    # [3,4] . [4,3] = 24, |[3,4]|=5, |[4,3]|=5 => 24/25 = 0.96
    assert cosine([3, 4], [4, 3]) == pytest.approx(0.96)
