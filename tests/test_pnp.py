"""RANSAC PnP and the motion-only Gauss-Newton refinement."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.frontend.odometry import (
    OdometryConfig,
    estimate_essential_motion,
    estimate_pose_pnp,
    huber_weights,
    refine_pose_gauss_newton,
)
from svslam.reprojection import project, project_stereo, transform_points
from svslam.se3 import se3_exp, se3_inverse

from conftest import requires_cv2


def _K(pinhole):
    return np.array([
        [pinhole["fx"], 0.0, pinhole["cx"]],
        [0.0, pinhole["fy"], pinhole["cy"]],
        [0.0, 0.0, 1.0],
    ])


def _scene(rng, pinhole, n=140):
    """A known pose and a cloud of points that all project inside the image."""
    truth = se3_exp(np.array([0.4, -0.1, -1.6, 0.01, 0.03, -0.02]))
    points = np.column_stack([
        rng.uniform(-12.0, 12.0, n),
        rng.uniform(-3.0, 3.0, n),
        rng.uniform(6.0, 40.0, n),
    ])
    pixels = project(transform_points(truth, points), pinhole["fx"], pinhole["fy"],
                     pinhole["cx"], pinhole["cy"])
    inside = (
        (pixels[:, 0] > 5) & (pixels[:, 0] < 1220)
        & (pixels[:, 1] > 5) & (pixels[:, 1] < 365)
    )
    return truth, points[inside], pixels[inside]


@requires_cv2
def test_pnp_recovers_a_known_pose(rng, pinhole):
    truth, points, pixels = _scene(rng, pinhole)
    estimate = estimate_pose_pnp(points, pixels, _K(pinhole))
    assert estimate.success
    error = se3_inverse(truth) @ estimate.T_cw
    assert np.linalg.norm(error[:3, 3]) < 1e-3
    assert np.abs(error[:3, :3] - np.eye(3)).max() < 1e-4
    assert estimate.inliers.sum() > 0.95 * points.shape[0]


@requires_cv2
def test_pnp_survives_a_known_outlier_fraction(rng, pinhole):
    """Thirty percent of the correspondences are pure noise; RANSAC finds them."""
    truth, points, pixels = _scene(rng, pinhole)
    n = points.shape[0]
    n_outliers = int(0.3 * n)
    corrupted = pixels.copy()
    outlier_index = rng.choice(n, n_outliers, replace=False)
    corrupted[outlier_index] += rng.uniform(30.0, 200.0, (n_outliers, 2)) * rng.choice(
        [-1.0, 1.0], (n_outliers, 2)
    )

    estimate = estimate_pose_pnp(points, corrupted, _K(pinhole))
    assert estimate.success
    error = se3_inverse(truth) @ estimate.T_cw
    assert np.linalg.norm(error[:3, 3]) < 0.05

    truth_mask = np.ones(n, dtype=bool)
    truth_mask[outlier_index] = False
    # Every accepted correspondence should be a true inlier, and most true
    # inliers should be accepted.
    assert not np.any(estimate.inliers & ~truth_mask)
    assert estimate.inliers.sum() > 0.9 * truth_mask.sum()


@requires_cv2
def test_pnp_reports_failure_with_too_few_points(pinhole):
    estimate = estimate_pose_pnp(np.zeros((3, 3)), np.zeros((3, 2)), _K(pinhole))
    assert not estimate.success
    assert estimate.n_correspondences == 3


def test_gauss_newton_refines_a_perturbed_pose(rng, pinhole):
    truth, points, pixels = _scene(rng, pinhole)
    noisy = pixels + rng.normal(scale=0.3, size=pixels.shape)
    start = se3_exp(np.concatenate([rng.normal(scale=0.05, size=3),
                                    rng.normal(scale=0.01, size=3)])) @ truth

    before = np.linalg.norm(
        project(transform_points(start, points), pinhole["fx"], pinhole["fy"],
                pinhole["cx"], pinhole["cy"]) - noisy, axis=1
    ).mean()
    refined, _, after = refine_pose_gauss_newton(
        start, points, noisy, pinhole["fx"], pinhole["fy"], pinhole["cx"], pinhole["cy"],
        iterations=20,
    )
    assert after < before
    assert after < 0.6
    error = se3_inverse(truth) @ refined
    assert np.linalg.norm(error[:3, 3]) < 0.05


def test_gauss_newton_uses_the_stereo_residual_for_scale(rng, pinhole):
    """The stereo residual constrains depth, so scale cannot drift."""
    truth, points, _ = _scene(rng, pinhole)
    observations = project_stereo(
        transform_points(truth, points), pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"], pinhole["baseline"],
    )
    start = se3_exp(np.array([0.2, 0.05, 0.3, 0.0, 0.0, 0.0])) @ truth
    refined, _, error = refine_pose_gauss_newton(
        start, points, observations, pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"], baseline=pinhole["baseline"], iterations=25,
    )
    assert error < 1e-3
    assert np.abs(se3_inverse(truth) @ refined - np.eye(4)).max() < 1e-5


def test_gauss_newton_is_robust_to_a_few_gross_outliers(rng, pinhole):
    truth, points, pixels = _scene(rng, pinhole)
    corrupted = pixels.copy()
    corrupted[:8] += 300.0
    refined, _, _ = refine_pose_gauss_newton(
        truth, points, corrupted, pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"], iterations=20, huber_delta=2.0,
    )
    drift = np.linalg.norm((se3_inverse(truth) @ refined)[:3, 3])
    assert drift < 0.1


def test_huber_weights_bound_the_influence_of_an_outlier():
    weights = huber_weights(np.array([0.0, 1.0, 2.0, 4.0, 100.0]), 2.0)
    assert np.allclose(weights[:3], 1.0)
    assert np.isclose(weights[3], 0.5)
    assert np.isclose(weights[4], 0.02)
    # Contribution to the gradient is w * r, which is bounded above by delta.
    residuals = np.array([2.0, 10.0, 1000.0])
    assert np.allclose(huber_weights(residuals, 2.0) * residuals, 2.0)


def test_gauss_newton_with_no_points_is_a_no_op(pinhole):
    T = np.eye(4)
    refined, norms, error = refine_pose_gauss_newton(
        T, np.zeros((0, 3)), np.zeros((0, 2)), pinhole["fx"], pinhole["fy"],
        pinhole["cx"], pinhole["cy"],
    )
    assert np.allclose(refined, T)
    assert norms.size == 0
    assert np.isnan(error)


@requires_cv2
def test_essential_matrix_recovers_rotation_but_only_a_direction(rng, pinhole):
    """Monocular geometry cannot observe scale; the returned t is a unit vector."""
    truth, points, pixels_prev = _scene(rng, pinhole, n=300)
    motion = se3_exp(np.array([0.3, 0.0, -2.0, 0.0, 0.02, 0.0]))
    pixels_curr = project(
        transform_points(motion, transform_points(truth, points)),
        pinhole["fx"], pinhole["fy"], pinhole["cx"], pinhole["cy"],
    )
    inside = (
        (pixels_curr[:, 0] > 5) & (pixels_curr[:, 0] < 1220)
        & (pixels_curr[:, 1] > 5) & (pixels_curr[:, 1] < 365)
    )
    R, t, mask = estimate_essential_motion(
        pixels_prev[inside], pixels_curr[inside], _K(pinhole), OdometryConfig()
    )
    assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-6)
    assert np.abs(R - motion[:3, :3]).max() < 5e-3
    direction = motion[:3, 3] / np.linalg.norm(motion[:3, 3])
    assert np.abs(np.dot(t, direction)) > 0.99
    assert mask.sum() > 0.8 * inside.sum()
