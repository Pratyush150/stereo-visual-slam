"""Stereo triangulation and depth-uncertainty rejection."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.frontend.stereo import (
    StereoConfig,
    depth_from_disparity,
    depth_sigma,
    disparity_to_points,
    min_disparity_for,
)
from svslam.reprojection import project_stereo


def test_triangulation_recovers_a_known_point(pinhole):
    """Project a known 3D point into a rectified pair, then invert it exactly."""
    fx, fy = pinhole["fx"], pinhole["fy"]
    cx, cy = pinhole["cx"], pinhole["cy"]
    b = pinhole["baseline"]
    truth = np.array([[1.7, -0.6, 12.5], [-4.0, 1.2, 8.0], [0.0, 0.0, 20.0]])
    uvur = project_stereo(truth, fx, fy, cx, cy, b)
    disparity = uvur[:, 0] - uvur[:, 2]
    recovered = disparity_to_points(uvur[:, :2], disparity, fx, fy, cx, cy, b)
    assert np.abs(recovered - truth).max() < 1e-9


def test_disparity_equals_fx_baseline_over_depth(pinhole):
    depths = np.array([5.0, 10.0, 40.0, 80.0])
    disparity = pinhole["fx"] * pinhole["baseline"] / depths
    assert np.allclose(
        depth_from_disparity(disparity, pinhole["fx"], pinhole["baseline"]), depths
    )


def test_depth_uncertainty_grows_quadratically(pinhole):
    """sigma_Z = Z^2 sigma_d / (fx b): doubling the depth quadruples the error."""
    fx, b, sigma_d = pinhole["fx"], pinhole["baseline"], 0.25
    near = depth_from_disparity(fx * b / 10.0, fx, b)
    d10 = fx * b / 10.0
    d20 = fx * b / 20.0
    s10 = depth_sigma(d10, fx, b, sigma_d)
    s20 = depth_sigma(d20, fx, b, sigma_d)
    assert np.isclose(near, 10.0)
    assert np.isclose(s20 / s10, 4.0, rtol=1e-9)
    # And the concrete numbers quoted in the module docstring.
    assert 0.05 < s10 < 0.08
    assert depth_sigma(fx * b / 80.0, fx, b, sigma_d) > 4.0


def test_min_disparity_follows_from_the_uncertainty_budget(pinhole):
    """sigma_Z/Z == sigma_d/d exactly, so the threshold has a closed form."""
    d_min = min_disparity_for(pinhole["fx"], pinhole["baseline"], 0.25, 0.05, None)
    assert np.isclose(d_min, 0.25 / 0.05)
    relative = depth_sigma(d_min, pinhole["fx"], pinhole["baseline"], 0.25) / depth_from_disparity(
        d_min, pinhole["fx"], pinhole["baseline"]
    )
    assert np.isclose(relative, 0.05)


def test_depth_ceiling_tightens_the_disparity_floor(pinhole):
    loose = min_disparity_for(pinhole["fx"], pinhole["baseline"], 0.25, 0.05, None)
    tight = min_disparity_for(pinhole["fx"], pinhole["baseline"], 0.25, 0.05, 20.0)
    assert tight > loose
    assert np.isclose(tight, pinhole["fx"] * pinhole["baseline"] / 20.0)


def test_non_positive_disparity_is_infinite_depth():
    assert np.isinf(depth_from_disparity(np.array([0.0, -1.0]), 700.0, 0.5)).all()


def test_default_config_rejects_beyond_its_depth_ceiling(pinhole):
    config = StereoConfig()
    d_min = min_disparity_for(
        pinhole["fx"], pinhole["baseline"], config.disparity_sigma,
        config.max_relative_depth_sigma, config.max_depth,
    )
    depth_at_floor = depth_from_disparity(d_min, pinhole["fx"], pinhole["baseline"])
    assert depth_at_floor <= config.max_depth + 1e-9
