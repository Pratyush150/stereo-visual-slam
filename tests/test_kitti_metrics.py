"""The official KITTI odometry metrics, ATE and RPE.

The metric definition is as much a part of "1% translation error" as the
pipeline is.  These tests pin it to values that can be worked out by hand, so a
change in the metric cannot silently change the headline number.
"""

from __future__ import annotations

import numpy as np

from svslam.evaluation.kitti_metrics import (
    KITTI_LENGTHS,
    absolute_trajectory_error,
    evaluate,
    kitti_odometry_errors,
    kitti_summary,
    last_frame_from_segment_length,
    relative_pose_error,
    trajectory_distances,
    umeyama_alignment,
)
from svslam.se3 import se3_exp


def straight_line(n: int = 1200, step: float = 1.0) -> np.ndarray:
    """n poses one metre apart along +z, the camera's forward axis."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n) * step
    return poses


def test_trajectory_distances_are_cumulative():
    poses = straight_line(5, 2.0)
    assert np.allclose(trajectory_distances(poses), [0.0, 2.0, 4.0, 6.0, 8.0])
    assert trajectory_distances(np.zeros((0, 4, 4))).size == 0


def test_segment_end_lookup():
    distances = trajectory_distances(straight_line(300))
    assert last_frame_from_segment_length(distances, 0, 100.0) == 100
    assert last_frame_from_segment_length(distances, 50, 100.0) == 150
    # No frame is far enough away, so the sub-sequence is skipped, not truncated.
    assert last_frame_from_segment_length(distances, 250, 100.0) == -1


def test_a_perfect_trajectory_scores_zero():
    poses = straight_line()
    report = evaluate(poses, poses)
    assert report.translation_percent == 0.0
    assert report.rotation_deg_per_m == 0.0
    assert report.ate["rmse"] == 0.0
    assert report.n_segments > 0


def test_a_one_percent_scale_error_scores_exactly_one_percent():
    """The hand-computable case: every sub-sequence is 1% too long."""
    truth = straight_line()
    estimate = truth.copy()
    estimate[:, 2, 3] *= 1.01
    report = evaluate(estimate, truth)
    assert np.isclose(report.translation_percent, 1.0, atol=1e-9)
    assert np.isclose(report.rotation_deg_per_m, 0.0, atol=1e-12)
    for length in KITTI_LENGTHS:
        assert np.isclose(report.per_length[length]["translation_percent"], 1.0, atol=1e-9)


def test_a_known_constant_yaw_error_scores_the_expected_rotation_error():
    """One degree of yaw over every 100 m is 0.01 deg/m by definition."""
    truth = straight_line(900)
    estimate = truth.copy()
    per_metre = np.deg2rad(1.0) / 100.0
    for i in range(estimate.shape[0]):
        estimate[i, :3, :3] = se3_exp(
            np.array([0.0, 0.0, 0.0, 0.0, per_metre * i, 0.0])
        )[:3, :3]
    errors = kitti_odometry_errors(estimate, truth, lengths=(100,), step=10)
    summary = kitti_summary(errors)
    assert np.isclose(summary["rotation_deg_per_m"], 0.01, rtol=1e-3)


def test_segment_counts_follow_the_official_stride():
    """Start frames step by 10; a sub-sequence needing more path is skipped."""
    truth = straight_line(1200)
    errors = kitti_odometry_errors(truth, truth, lengths=(100,), step=10)
    # Starts 0, 10, ... 1090 all reach 100 m; 1100 and beyond do not.
    assert len(errors) == 110


def test_summary_of_no_segments_is_not_a_crash():
    summary = kitti_summary([])
    assert np.isnan(summary["translation_percent"])
    assert summary["n_segments"] == 0


def test_umeyama_recovers_a_known_similarity(rng):
    source = rng.normal(size=(60, 3)) * 4.0
    R_true = se3_exp(np.array([0.0, 0.0, 0.0, 0.3, -0.2, 0.1]))[:3, :3]
    t_true = np.array([2.0, -5.0, 1.5])
    scale_true = 1.7
    target = scale_true * (source @ R_true.T) + t_true

    R, t, s = umeyama_alignment(source, target, with_scale=True)
    assert np.abs(R - R_true).max() < 1e-9
    assert np.abs(t - t_true).max() < 1e-8
    assert np.isclose(s, scale_true)

    R2, t2, s2 = umeyama_alignment(source, target, with_scale=False)
    assert np.isclose(s2, 1.0)
    assert np.abs(R2 - R_true).max() < 1e-9


def test_umeyama_handles_empty_input():
    R, t, s = umeyama_alignment(np.zeros((0, 3)), np.zeros((0, 3)))
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, 0.0)
    assert s == 1.0


def test_ate_is_invariant_to_a_rigid_transform_when_aligned(rng):
    truth = straight_line(200)
    T = se3_exp(np.array([3.0, -1.0, 2.0, 0.1, 0.2, -0.3]))
    moved = np.einsum("ij,njk->nik", T, truth)
    result = absolute_trajectory_error(moved, truth, align=True)
    assert result["rmse"] < 1e-9
    unaligned = absolute_trajectory_error(moved, truth, align=False)
    assert unaligned["rmse"] > 1.0


def test_ate_reports_a_scale_when_asked(rng):
    truth = straight_line(200)
    scaled = truth.copy()
    scaled[:, :3, 3] *= 0.8
    result = absolute_trajectory_error(scaled, truth, align=True, with_scale=True)
    assert np.isclose(result["scale"], 1.25, rtol=1e-6)
    assert result["rmse"] < 1e-9


def test_rpe_measures_local_drift_not_global_drift():
    truth = straight_line(300)
    # A constant offset is a global error with zero relative error.
    offset = truth.copy()
    offset[:, 0, 3] += 50.0
    result = relative_pose_error(offset, truth, delta=10)
    assert result["translation_rmse"] < 1e-9
    assert result["n_pairs"] == 290

    # A per-step scale error is a genuine local error.
    scaled = truth.copy()
    scaled[:, 2, 3] *= 1.05
    result = relative_pose_error(scaled, truth, delta=10)
    assert np.isclose(result["translation_rmse"], 0.5, rtol=1e-9)


def test_rpe_with_too_few_poses():
    result = relative_pose_error(straight_line(5), straight_line(5), delta=10)
    assert result["n_pairs"] == 0
    assert np.isnan(result["translation_rmse"])


def test_report_table_renders_every_length():
    truth = straight_line()
    estimate = truth.copy()
    estimate[:, 2, 3] *= 1.02
    table = evaluate(estimate, truth).format_table()
    for length in KITTI_LENGTHS:
        assert f"{length}" in table
    assert "trans err (%)" in table


def test_report_serialises_to_plain_types():
    truth = straight_line(400)
    data = evaluate(truth, truth).as_dict()
    assert set(data) >= {"translation_percent", "rotation_deg_per_m", "ate", "rpe"}
    assert isinstance(data["ate"]["rmse"], float)
