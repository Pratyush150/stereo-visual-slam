"""A false loop closure must not destroy the map.

This is the failure that ends SLAM systems.  One appearance match between two
different but similar-looking places produces a strong, confident and completely
wrong constraint, and unweighted least squares will fold the trajectory in half
to satisfy it.  The tests here inject exactly that and check the map survives.
"""

from __future__ import annotations

import numpy as np
import pytest

from svslam.backend.posegraph import (
    PoseGraphConfig,
    PoseGraphEdge,
    optimise_pose_graph,
    robust_weight,
)
from svslam.se3 import se3_exp, se3_inverse

from test_posegraph import ate, drifted_graph


def _graph_with_true_loop(rng, n=40):
    truth, estimate, edges = drifted_graph(rng, n=n)
    edges.append(PoseGraphEdge(
        n - 1, 0, se3_inverse(truth[n - 1]) @ truth[0], np.eye(6) * 100.0, is_loop=True
    ))
    return truth, estimate, edges


def test_huber_weight_shape():
    assert robust_weight(0.0, "huber", 1.0, 1.0) == 1.0
    assert robust_weight(0.5, "huber", 1.0, 1.0) == 1.0
    assert np.isclose(robust_weight(4.0, "huber", 1.0, 1.0), 0.5)
    assert robust_weight(1e6, "huber", 1.0, 1.0) < 2e-3


def test_dcs_weight_decays_faster_than_huber():
    """DCS switches an edge off; Huber only leans on it."""
    chi2 = 400.0
    assert robust_weight(chi2, "dcs", 1.0, 1.0) < robust_weight(chi2, "huber", 1.0, 1.0)
    assert robust_weight(0.0, "dcs", 1.0, 1.0) == 1.0
    assert robust_weight(1.0, "dcs", 1.0, 1.0) == 1.0


def test_no_kernel_means_no_kernel():
    assert robust_weight(1e9, "none", 1.0, 1.0) == 1.0


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError):
        robust_weight(1.0, "cauchy", 1.0, 1.0)


def test_a_correct_loop_closure_keeps_full_weight(rng):
    """The kernel must not reject the good edges along with the bad ones."""
    truth, estimate, edges = _graph_with_true_loop(rng)
    result = optimise_pose_graph(estimate, edges, PoseGraphConfig(kernel="dcs"))
    assert result.edge_weights[-1] > 0.9
    assert ate(result.poses, truth) < ate(estimate, truth)


def test_false_loop_closure_destroys_an_unweighted_graph(rng):
    """Establish the damage first, so the robust result means something."""
    truth, estimate, edges = _graph_with_true_loop(rng)
    false_edge = PoseGraphEdge(
        30, 5, se3_exp(np.array([8.0, 3.0, 1.0, 0.4, 0.2, 0.1])),
        np.eye(6) * 100.0, is_loop=True,
    )
    result = optimise_pose_graph(estimate, edges + [false_edge],
                                 PoseGraphConfig(kernel="none"))
    assert ate(result.poses, truth) > 5.0 * ate(estimate, truth)


def test_dcs_survives_an_injected_false_loop_closure(rng):
    truth, estimate, edges = _graph_with_true_loop(rng)
    clean = optimise_pose_graph(estimate, edges, PoseGraphConfig(kernel="dcs"))
    false_edge = PoseGraphEdge(
        30, 5, se3_exp(np.array([8.0, 3.0, 1.0, 0.4, 0.2, 0.1])),
        np.eye(6) * 100.0, is_loop=True,
    )
    robust = optimise_pose_graph(estimate, edges + [false_edge],
                                 PoseGraphConfig(kernel="dcs"))

    # The false edge is switched off, the true one is kept.
    assert robust.edge_weights[-1] < 1e-3
    assert robust.edge_weights[-2] > 0.9
    # And the map is no worse than it was without the false edge at all.
    assert ate(robust.poses, truth) < 1.05 * ate(clean.poses, truth)
    assert ate(robust.poses, truth) < ate(estimate, truth)


def test_kernel_is_applied_only_to_loop_edges_by_default(rng):
    """Sequential odometry is trusted; only loop edges are candidates for rejection."""
    truth, estimate, edges = _graph_with_true_loop(rng)
    config = PoseGraphConfig(kernel="dcs", kernel_on_loops_only=True)
    result = optimise_pose_graph(estimate, edges, config)
    assert np.allclose(result.edge_weights[:-1], 1.0)
