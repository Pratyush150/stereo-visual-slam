"""Figure rendering.

The figures in the README are produced by these functions from real run output,
so the tests check that each one writes a non-trivial PNG rather than failing
silently or producing an empty canvas.
"""

from __future__ import annotations

import numpy as np
import pytest

from svslam.map import SlamMap
from svslam.se3 import se3_exp, se3_inverse

matplotlib = pytest.importorskip("matplotlib")

from svslam import viz  # noqa: E402


def _trajectory(n=120, drift=0.0):
    poses = np.tile(np.eye(4), (n, 1, 1))
    t = np.arange(n)
    poses[:, 0, 3] = 20.0 * np.sin(t / 25.0) + drift * t / n
    poses[:, 2, 3] = 20.0 * (1.0 - np.cos(t / 25.0))
    return poses


def _is_a_real_png(path, min_bytes=3000):
    data = path.read_bytes()
    return data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > min_bytes


def test_trajectory_figure(tmp_path):
    path = viz.plot_trajectory(_trajectory(drift=3.0), _trajectory(),
                               tmp_path / "trajectory.png", subtitle="0.9% translation")
    assert _is_a_real_png(path)


def test_trajectory_figure_without_ground_truth(tmp_path):
    path = viz.plot_trajectory(_trajectory(), None, tmp_path / "no_gt.png")
    assert _is_a_real_png(path)


def test_loop_comparison_figure(tmp_path):
    path = viz.plot_loop_comparison(
        _trajectory(drift=6.0), _trajectory(drift=0.5), _trajectory(),
        tmp_path / "loop.png", ate_before=4.2, ate_after=0.8,
    )
    assert _is_a_real_png(path)


def test_kitti_error_figure(tmp_path):
    from svslam.evaluation.kitti_metrics import evaluate

    truth = np.tile(np.eye(4), (1200, 1, 1))
    truth[:, 2, 3] = np.arange(1200, dtype=float)
    estimate = truth.copy()
    estimate[:, 2, 3] *= 1.015
    report = evaluate(estimate, truth)
    path = viz.plot_kitti_errors(report, tmp_path / "errors.png")
    assert _is_a_real_png(path)


def test_tracked_features_figure(tmp_path, rng):
    image = rng.integers(0, 255, (370, 1226), dtype=np.uint8)
    points = np.column_stack([rng.uniform(0, 1226, 300), rng.uniform(0, 370, 300)])
    inliers = rng.random(300) > 0.25
    path = viz.plot_tracked_features(image, points, inliers, tmp_path / "features.png")
    assert _is_a_real_png(path)


def test_ba_sparsity_figure(tmp_path, rng, pinhole):
    from svslam.backend.ba import BAProblem

    n_cameras, n_points = 6, 200
    cam_idx, pt_idx = [], []
    for i in range(n_cameras):
        for j in range(i * 20, min(i * 20 + 90, n_points)):
            cam_idx.append(i)
            pt_idx.append(j)
    problem = BAProblem(
        poses_cw=np.tile(np.eye(4), (n_cameras, 1, 1)),
        points=rng.normal(size=(n_points, 3)) + np.array([0.0, 0.0, 15.0]),
        camera_index=np.array(cam_idx), point_index=np.array(pt_idx),
        observations=rng.normal(size=(len(cam_idx), 3)) * 100.0,
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    path = viz.plot_ba_sparsity(problem, tmp_path / "sparsity.png")
    assert _is_a_real_png(path)


def test_covisibility_figure(tmp_path):
    slam_map = SlamMap()
    for i in range(12):
        slam_map.add_keyframe(
            i, se3_inverse(se3_exp(np.array([0.0, 0.0, float(i), 0.0, 0.1 * i, 0.0]))),
            np.zeros((40, 2)), np.zeros((40, 32), np.uint8),
        )
    for j in range(40):
        landmark = slam_map.add_landmark(np.array([float(j), 0.0, 10.0]),
                                         np.zeros(32, np.uint8))
        for kf_id in range(12):
            slam_map.add_observation(kf_id, j, landmark.id)
    path = viz.plot_covisibility(slam_map, tmp_path / "covis.png", min_shared=15)
    assert _is_a_real_png(path)


def test_palette_is_the_documented_one():
    """Three categorical slots, used in a fixed order across every figure."""
    assert viz.SERIES == ("#2a78d6", "#eb6834", "#1baf7a")
    assert len(set(viz.SERIES)) == 3
    assert viz.SURFACE.startswith("#") and viz.INK.startswith("#")
