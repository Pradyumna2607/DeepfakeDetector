"""Tests for the offline heuristic baseline and config."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.baseline import compute_fake_probability, score_face_crop  # noqa: E402


def _random_face(seed=0, size=128):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_score_face_crop_returns_dict():
    out = score_face_crop(_random_face())
    assert isinstance(out, dict) and out


def test_probability_in_range():
    crops = [_random_face(i) for i in range(5)]
    result = compute_fake_probability(crops)
    assert isinstance(result, dict)
    p = result.get("fake_probability", result.get("probability"))
    assert p is not None and 0.0 <= float(p) <= 1.0
