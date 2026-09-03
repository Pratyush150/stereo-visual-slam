"""Analytic reprojection Jacobians against central differences.

This is the credibility anchor of the whole repository.  Every optimiser here --
motion-only Gauss-Newton, windowed bundle adjustment -- descends on these
derivatives.  If they are wrong the system still runs, still produces a
trajectory, and is silently worse; nothing crashes.  So they are checked
numerically, for both residual flavours and both parameter blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from svslam.reprojection import (
    project,
    project_stereo,
    projection_jacobian,
    reprojection_jacobians,
    reprojection_residual,
    stereo_projection_jacobian,
    transform_points,
)
from svslam.se3 import se3_exp


def _scene(rng, n=9):
    T = se3_exp(np.concatenate([rng.normal(size=3) * 0.4, rng.normal(size=3) * 0.2]))
    points = np.column_stack([
        rng.uniform(-6.0, 6.0, n),
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(6.0, 30.0, n),
    ])
    return T, points


@pytest.mark.parametrize("stereo", [False, True])
def test_pose_jacobian_matches_central_differences(rng, pinhole, stereo):
    baseline = pinhole["baseline"] if stereo else None
    T, points = _scene(rng)
    k = 3 if stereo else 2
    observations = rng.normal(size=(points.shape[0], k)) * 2.0
    if stereo:
        observations += project_stereo(
            transform_points(T, points), pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"], pinhole["baseline"])
    else:
        observations += project(
            transform_points(T, points), pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"])

    analytic, _ = reprojection_jacobians(T, points, pinhole["fx"], pinhole["fy"], baseline)
    eps = 1e-7
    numeric = np.zeros_like(analytic)
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        plus = reprojection_residual(
            se3_exp(d) @ T, points, observations, pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"], baseline)
        minus = reprojection_residual(
            se3_exp(-d) @ T, points, observations, pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"], baseline)
        numeric[:, :, j] = (plus - minus) / (2.0 * eps)

    scale = max(float(np.abs(analytic).max()), 1.0)
    assert np.abs(numeric - analytic).max() / scale < 1e-6


@pytest.mark.parametrize("stereo", [False, True])
def test_point_jacobian_matches_central_differences(rng, pinhole, stereo):
    baseline = pinhole["baseline"] if stereo else None
    T, points = _scene(rng)
    k = 3 if stereo else 2
    observations = rng.normal(size=(points.shape[0], k)) * 2.0

    _, analytic = reprojection_jacobians(T, points, pinhole["fx"], pinhole["fy"], baseline)
    eps = 1e-7
    numeric = np.zeros_like(analytic)
    for j in range(3):
        d = np.zeros_like(points)
        d[:, j] = eps
        plus = reprojection_residual(
            T, points + d, observations, pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"], baseline)
        minus = reprojection_residual(
            T, points - d, observations, pinhole["fx"], pinhole["fy"],
            pinhole["cx"], pinhole["cy"], baseline)
        numeric[:, :, j] = (plus - minus) / (2.0 * eps)

    scale = max(float(np.abs(analytic).max()), 1.0)
    assert np.abs(numeric - analytic).max() / scale < 1e-6


def test_rotation_block_sign_is_not_symmetric(rng, pinhole):
    """A flipped sign on the rotation block still looks plausible; catch it here.

    The rotation columns of the pose Jacobian come from ``-[p_c]_x``.  Writing
    ``+[p_c]_x`` produces a Jacobian of the right shape and magnitude that sends
    every optimiser uphill, so the test asserts the sign explicitly.
    """
    T, points = _scene(rng, n=4)
    J_pose, _ = reprojection_jacobians(T, points, pinhole["fx"], pinhole["fy"])
    p_cam = transform_points(T, points)
    J_proj = projection_jacobian(p_cam, pinhole["fx"], pinhole["fy"])
    for i in range(points.shape[0]):
        expected = J_proj[i] @ np.array([
            [0.0, p_cam[i, 2], -p_cam[i, 1]],
            [-p_cam[i, 2], 0.0, p_cam[i, 0]],
            [p_cam[i, 1], -p_cam[i, 0], 0.0],
        ])
        assert np.allclose(J_pose[i, :, 3:], expected, atol=1e-12)


def test_projection_jacobian_shapes_and_values(rng, pinhole):
    points = np.array([[1.0, 2.0, 10.0], [-3.0, 0.5, 25.0]])
    J = projection_jacobian(points, pinhole["fx"], pinhole["fy"])
    assert J.shape == (2, 2, 3)
    assert np.isclose(J[0, 0, 0], pinhole["fx"] / 10.0)
    assert np.isclose(J[0, 1, 2], -pinhole["fy"] * 2.0 / 100.0)

    Js = stereo_projection_jacobian(points, pinhole["fx"], pinhole["fy"], pinhole["baseline"])
    assert Js.shape == (2, 3, 3)
    # The third row differs from the first only by the baseline term.
    assert np.isclose(
        Js[0, 2, 2] - Js[0, 0, 2],
        pinhole["fx"] * pinhole["baseline"] / 100.0,
    )


def test_residual_is_zero_at_the_truth(rng, pinhole):
    T, points = _scene(rng)
    observations = project_stereo(
        transform_points(T, points), pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"], pinhole["baseline"])
    residual = reprojection_residual(
        T, points, observations, pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"], pinhole["baseline"])
    assert np.abs(residual).max() < 1e-9
