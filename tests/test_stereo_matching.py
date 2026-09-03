"""Sparse stereo block matching on a synthetic rectified pair."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.frontend.stereo import StereoConfig, match_stereo_epipolar

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _textured_pair(rng, disparity: int, size=(240, 480)):
    """A random-texture pair with a known constant disparity.

    For a rectified pair a left pixel at ``u`` matches the right pixel at
    ``u - d``, so the right image must carry the same texture shifted left by
    ``d``.  Shifting random texture is the cleanest possible test pair: the
    correct match is unambiguous and known exactly, so anything the matcher gets
    wrong is the matcher's fault rather than the scene's.
    """
    height, width = size
    base = rng.integers(0, 256, (height, width + disparity), dtype=np.uint8)
    left = base[:, :width]
    right = base[:, disparity:disparity + width]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def test_recovers_a_known_constant_disparity(rng, pinhole):
    disparity = 24
    left, right = _textured_pair(rng, disparity)
    points = np.column_stack([
        rng.integers(60, 460, 40).astype(float),
        rng.integers(20, 220, 40).astype(float),
    ])
    config = StereoConfig(min_disparity=2.0, max_disparity=64, max_depth=1e6,
                          max_relative_depth_sigma=1.0)
    matches = match_stereo_epipolar(
        left, right, points, config,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert len(matches) >= 20
    assert np.abs(matches.disparity - disparity).max() < 1.0
    assert np.median(np.abs(matches.disparity - disparity)) < 0.2


def test_depths_are_consistent_with_the_disparity(rng, pinhole):
    disparity = 32
    left, right = _textured_pair(rng, disparity)
    points = np.column_stack([
        rng.integers(60, 460, 30).astype(float),
        rng.integers(20, 220, 30).astype(float),
    ])
    config = StereoConfig(min_disparity=2.0, max_disparity=64, max_depth=1e6,
                          max_relative_depth_sigma=1.0)
    matches = match_stereo_epipolar(
        left, right, points, config,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    expected = pinhole["fx"] * pinhole["baseline"] / disparity
    assert np.allclose(matches.depth, expected, rtol=0.05)
    assert np.allclose(matches.points_cam[:, 2], matches.depth)


def test_far_points_are_rejected_by_the_uncertainty_gate(rng, pinhole):
    """A small true disparity is a distant point; the default config drops it."""
    disparity = 4
    left, right = _textured_pair(rng, disparity)
    points = np.column_stack([
        rng.integers(60, 460, 40).astype(float),
        rng.integers(20, 220, 40).astype(float),
    ])
    strict = StereoConfig()  # defaults: max_relative_depth_sigma 0.05
    matches = match_stereo_epipolar(
        left, right, points, strict,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert len(matches) == 0

    permissive = StereoConfig(min_disparity=1.0, max_relative_depth_sigma=1.0,
                              max_depth=1e6)
    kept = match_stereo_epipolar(
        left, right, points, permissive,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert len(kept) > 0


def test_uniform_region_produces_no_matches(pinhole):
    """A flat wall has no unique match anywhere along the epipolar line."""
    flat = np.full((160, 320), 128, dtype=np.uint8)
    points = np.column_stack([np.linspace(60, 260, 20), np.full(20, 80.0)])
    config = StereoConfig(min_disparity=2.0, max_disparity=48, max_depth=1e6,
                          max_relative_depth_sigma=1.0)
    matches = match_stereo_epipolar(
        flat, flat, points, config,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert len(matches) == 0


def test_no_input_points_returns_empty(pinhole):
    image = np.zeros((100, 200), dtype=np.uint8)
    matches = match_stereo_epipolar(
        image, image, np.zeros((0, 2)), StereoConfig(),
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert len(matches) == 0
    assert matches.points_cam.shape == (0, 3)


def test_result_indices_point_back_into_the_input(rng, pinhole):
    left, right = _textured_pair(rng, 20)
    points = np.column_stack([
        rng.integers(60, 460, 25).astype(float),
        rng.integers(20, 220, 25).astype(float),
    ])
    config = StereoConfig(min_disparity=2.0, max_disparity=64, max_depth=1e6,
                          max_relative_depth_sigma=1.0)
    matches = match_stereo_epipolar(
        left, right, points, config,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    assert np.all(matches.index < points.shape[0])
    assert np.allclose(matches.uv_left, points[matches.index])
