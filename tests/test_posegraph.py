"""Pose-graph optimisation: Jacobians and drift removal."""

from __future__ import annotations

import numpy as np

from svslam.backend.posegraph import (
    PoseGraphConfig,
    PoseGraphEdge,
    edge_error,
    edge_jacobians,
    have_scipy,
    optimise_pose_graph,
)
from svslam.se3 import se3_exp, se3_inverse


def circular_trajectory(n: int = 40, step: float = 1.0) -> np.ndarray:
    """A closed loop: n equal steps that turn a full circle."""
    poses = [np.eye(4)]
    increment = se3_exp(np.array([step, 0.0, 0.0, 0.0, 2.0 * np.pi / n, 0.0]))
    for _ in range(n - 1):
        poses.append(poses[-1] @ increment)
    return np.array(poses)


def drifted_graph(rng, n=40, sigma_t=0.01, sigma_r=0.004, information=100.0):
    """Ground truth, a drifted odometry estimate, and the edges between them."""
    truth = circular_trajectory(n)
    edges, estimate = [], [truth[0]]
    for i in range(1, n):
        relative = se3_inverse(truth[i - 1]) @ truth[i]
        noisy = relative @ se3_exp(np.concatenate([
            rng.normal(scale=sigma_t, size=3), rng.normal(scale=sigma_r, size=3)
        ]))
        estimate.append(estimate[-1] @ noisy)
        edges.append(PoseGraphEdge(i - 1, i, noisy, np.eye(6) * information))
    return truth, np.array(estimate), edges


def ate(poses: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((poses[:, :3, 3] - truth[:, :3, 3]) ** 2, axis=1))))


def test_edge_error_is_zero_for_a_consistent_measurement(rng):
    for _ in range(20):
        Ti = se3_exp(rng.normal(size=6))
        Tj = se3_exp(rng.normal(size=6))
        measurement = se3_inverse(Ti) @ Tj
        assert np.abs(edge_error(Ti, Tj, measurement)).max() < 1e-9


def test_edge_jacobians_match_central_differences(rng):
    eps = 1e-5
    worst = 0.0
    for _ in range(30):
        Ti = se3_exp(rng.normal(size=6))
        Tj = se3_exp(rng.normal(size=6))
        measurement = se3_exp(rng.normal(size=6) * 0.3)
        _, J_i, J_j = edge_jacobians(Ti, Tj, measurement)
        numeric_i = np.zeros((6, 6))
        numeric_j = np.zeros((6, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            numeric_i[:, k] = (
                edge_error(Ti @ se3_exp(d), Tj, measurement)
                - edge_error(Ti @ se3_exp(-d), Tj, measurement)
            ) / (2.0 * eps)
            numeric_j[:, k] = (
                edge_error(Ti, Tj @ se3_exp(d), measurement)
                - edge_error(Ti, Tj @ se3_exp(-d), measurement)
            ) / (2.0 * eps)
        worst = max(worst, float(np.abs(numeric_i - J_i).max()),
                    float(np.abs(numeric_j - J_j).max()))
    assert worst < 1e-6


def test_loop_closure_reduces_drift(rng):
    truth, estimate, edges = drifted_graph(rng)
    n = truth.shape[0]
    edges.append(PoseGraphEdge(
        n - 1, 0, se3_inverse(truth[n - 1]) @ truth[0], np.eye(6) * 100.0, is_loop=True
    ))
    before = ate(estimate, truth)
    result = optimise_pose_graph(estimate, edges, PoseGraphConfig(kernel="none"))
    after = ate(result.poses, truth)
    assert after < before
    assert after < 0.8 * before
    assert result.final_cost < result.initial_cost


def test_first_node_is_held_fixed(rng):
    truth, estimate, edges = drifted_graph(rng)
    result = optimise_pose_graph(estimate, edges, PoseGraphConfig(kernel="none"))
    assert np.allclose(result.poses[0], estimate[0])


def test_odometry_only_graph_is_already_optimal(rng):
    """With no loop edge the odometry chain explains itself exactly."""
    truth, estimate, edges = drifted_graph(rng)
    result = optimise_pose_graph(estimate, edges, PoseGraphConfig(kernel="none"))
    assert result.initial_cost < 1e-12
    assert np.abs(result.poses - estimate).max() < 1e-6


def test_empty_graph_is_handled():
    poses = np.tile(np.eye(4), (3, 1, 1))
    result = optimise_pose_graph(poses, [], PoseGraphConfig())
    assert np.allclose(result.poses, poses)
    assert result.converged


def test_scipy_flag_is_a_boolean():
    """The sparse path is optional; the dense fallback must stay available."""
    assert isinstance(have_scipy(), bool)
