"""Bundle adjustment: the Schur complement, and that it actually converges."""

from __future__ import annotations

import numpy as np

from svslam.backend.ba import (
    BAConfig,
    BAProblem,
    build_normal_equations,
    bundle_adjust,
    solve_dense,
    solve_schur,
)
from svslam.reprojection import project_stereo, transform_points
from svslam.se3 import se3_exp, se3_inverse


def _problem(rng, pinhole, n_cameras=5, n_points=60, point_noise=0.08,
             pose_noise=0.01, pixel_noise=0.3):
    """A small forward-driving stereo window with known poses and points."""
    truth_poses = np.array([
        se3_inverse(se3_exp(np.array([0.0, 0.0, 1.4 * i, 0.0, 0.02 * i, 0.0])))
        for i in range(n_cameras)
    ])
    truth_points = np.column_stack([
        rng.uniform(-10.0, 10.0, n_points),
        rng.uniform(-2.5, 2.5, n_points),
        rng.uniform(6.0, 35.0, n_points),
    ])

    cam_idx, pt_idx, observations = [], [], []
    for i in range(n_cameras):
        p_cam = transform_points(truth_poses[i], truth_points)
        uv = project_stereo(p_cam, pinhole["fx"], pinhole["fy"], pinhole["cx"],
                            pinhole["cy"], pinhole["baseline"])
        visible = (
            (p_cam[:, 2] > 2.0)
            & (uv[:, 0] > 0) & (uv[:, 0] < 1226)
            & (uv[:, 1] > 0) & (uv[:, 1] < 370)
        )
        for j in np.flatnonzero(visible):
            cam_idx.append(i)
            pt_idx.append(int(j))
            observations.append(uv[j] + rng.normal(scale=pixel_noise, size=3))

    poses = np.array(truth_poses)
    for i in range(1, n_cameras):
        poses[i] = se3_exp(np.concatenate([
            rng.normal(scale=pose_noise, size=3),
            rng.normal(scale=pose_noise * 0.1, size=3),
        ])) @ poses[i]

    problem = BAProblem(
        poses_cw=poses,
        points=truth_points + rng.normal(scale=point_noise, size=truth_points.shape),
        camera_index=np.array(cam_idx),
        point_index=np.array(pt_idx),
        observations=np.array(observations),
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
        fixed_cameras=(0,),
    )
    return problem, truth_poses, truth_points


def test_schur_solve_equals_the_dense_solve(rng, pinhole):
    """The whole point of the Schur complement: same answer, far less work."""
    problem, _, _ = _problem(rng, pinhole)
    normal = build_normal_equations(problem, 2.0)
    for lam in (0.0, 1e-4, 1e-1, 10.0):
        dc_schur, dp_schur = solve_schur(normal, problem, lam)
        dc_dense, dp_dense = solve_dense(normal, problem, lam)
        assert np.abs(dc_schur - dc_dense).max() < 1e-9
        assert np.abs(dp_schur - dp_dense).max() < 1e-8


def test_schur_solve_equals_dense_with_mono_residuals(rng, pinhole):
    problem, _, _ = _problem(rng, pinhole)
    mono = BAProblem(
        poses_cw=problem.poses_cw, points=problem.points,
        camera_index=problem.camera_index, point_index=problem.point_index,
        observations=problem.observations[:, :2],
        fx=problem.fx, fy=problem.fy, cx=problem.cx, cy=problem.cy,
        baseline=None, fixed_cameras=(0,),
    )
    normal = build_normal_equations(mono, 2.0)
    dc_schur, dp_schur = solve_schur(normal, mono, 1e-3)
    dc_dense, dp_dense = solve_dense(normal, mono, 1e-3)
    assert np.abs(dc_schur - dc_dense).max() < 1e-9
    assert np.abs(dp_schur - dp_dense).max() < 1e-8


def test_fixed_camera_receives_no_update(rng, pinhole):
    """Without a fixed camera the problem has a six-dimensional gauge freedom."""
    problem, _, _ = _problem(rng, pinhole)
    normal = build_normal_equations(problem, 2.0)
    dc, _ = solve_schur(normal, problem, 1e-4)
    assert np.allclose(dc[0], 0.0)
    assert np.linalg.norm(dc[1:]) > 0.0


def test_normal_equations_have_the_expected_block_structure(rng, pinhole):
    problem, _, _ = _problem(rng, pinhole)
    normal = build_normal_equations(problem, 2.0)
    assert normal["U"].shape == (problem.n_cameras, 6, 6)
    assert normal["V"].shape == (problem.n_points, 3, 3)
    assert normal["b_c"].shape == (problem.n_cameras, 6)
    assert normal["b_p"].shape == (problem.n_points, 3)
    # W is sparse: at most one block per (camera, landmark) that co-occur.
    assert len(normal["W"]) <= problem.n_observations
    for block in normal["W"].values():
        assert block.shape == (6, 3)
    # Blocks are symmetric positive semi-definite by construction.
    for i in range(problem.n_cameras):
        assert np.allclose(normal["U"][i], normal["U"][i].T, atol=1e-9)
        assert np.linalg.eigvalsh(normal["U"][i]).min() > -1e-9


def test_bundle_adjustment_reduces_cost_and_recovers_geometry(rng, pinhole):
    problem, truth_poses, truth_points = _problem(rng, pinhole)
    result = bundle_adjust(problem, BAConfig(max_iterations=25))

    assert result.final_cost < result.initial_cost
    assert result.final_rmse < result.initial_rmse
    assert result.final_rmse < 1.0

    estimated_centres = np.array([se3_inverse(T)[:3, 3] for T in result.poses_cw])
    truth_centres = np.array([se3_inverse(T)[:3, 3] for T in truth_poses])
    assert np.abs(estimated_centres - truth_centres).max() < 0.02
    assert np.abs(result.points - truth_points).mean() < 0.1


def test_bundle_adjustment_is_monotone(rng, pinhole):
    """Levenberg-Marquardt only accepts steps that lower the cost."""
    problem, _, _ = _problem(rng, pinhole, pose_noise=0.03, point_noise=0.25)
    result = bundle_adjust(problem, BAConfig(max_iterations=25))
    assert len(result.history) >= 2
    assert all(b <= a + 1e-9 for a, b in zip(result.history, result.history[1:]))


def test_bundle_adjustment_handles_an_empty_problem(pinhole):
    problem = BAProblem(
        poses_cw=np.eye(4).reshape(1, 4, 4), points=np.zeros((0, 3)),
        camera_index=np.zeros(0, int), point_index=np.zeros(0, int),
        observations=np.zeros((0, 3)),
        fx=pinhole["fx"], fy=pinhole["fy"], cx=pinhole["cx"], cy=pinhole["cy"],
        baseline=pinhole["baseline"],
    )
    result = bundle_adjust(problem)
    assert np.all(np.isfinite(result.poses_cw))
    assert result.final_cost == 0.0
