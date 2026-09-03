"""SE(3) and SO(3) exponential, logarithm, adjoint and Jacobians."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.se3 import (
    normalise_rotation,
    rotation_angle,
    se3_adjoint,
    se3_exp,
    se3_inverse,
    se3_left_jacobian,
    se3_left_jacobian_inverse,
    se3_log,
    skew,
    so3_exp,
    so3_log,
    unskew,
)


def test_skew_unskew_round_trip(rng):
    for _ in range(20):
        v = rng.normal(size=3)
        assert np.allclose(unskew(skew(v)), v)
        assert np.allclose(skew(v), -skew(v).T)


def test_so3_exp_is_a_rotation(rng):
    for _ in range(50):
        R = so3_exp(rng.normal(size=3) * rng.uniform(0.0, 3.0))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_se3_exp_log_round_trip_generic(rng):
    """log(exp(xi)) == xi whenever the rotation angle is inside (0, pi)."""
    worst = 0.0
    for _ in range(300):
        phi = rng.normal(size=3)
        phi = phi / np.linalg.norm(phi) * rng.uniform(1e-3, np.pi - 1e-3)
        xi = np.concatenate([rng.normal(size=3) * 5.0, phi])
        worst = max(worst, float(np.abs(se3_log(se3_exp(xi)) - xi).max()))
    assert worst < 1e-9


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-9, 1e-7, 1e-4, 1e-2])
def test_se3_round_trip_near_zero(angle, rng):
    """The small-angle branch must stay exact, not merely finite."""
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi = np.concatenate([rng.normal(size=3), axis * angle])
        T = se3_exp(xi)
        assert np.all(np.isfinite(T))
        assert np.allclose(se3_exp(se3_log(T)), T, atol=1e-10)


@pytest.mark.parametrize("gap", [1e-12, 1e-9, 1e-6, 1e-3])
def test_se3_round_trip_near_pi(gap, rng):
    """Near pi the naive log formula divides by sin(theta) and loses everything."""
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi = np.concatenate([rng.normal(size=3) * 3.0, axis * (np.pi - gap)])
        T = se3_exp(xi)
        recovered = se3_log(T)
        assert np.isclose(np.linalg.norm(recovered[3:]), np.pi - gap, atol=1e-7)
        assert np.allclose(se3_exp(recovered), T, atol=1e-8)


def test_so3_log_at_exactly_pi(rng):
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = so3_exp(axis * np.pi)
        recovered = so3_log(R)
        assert np.isclose(np.linalg.norm(recovered), np.pi, atol=1e-6)
        assert np.allclose(so3_exp(recovered), R, atol=1e-7)


def test_se3_inverse_matches_matrix_inverse(rng):
    for _ in range(20):
        T = se3_exp(rng.normal(size=6))
        assert np.allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-12)
        assert np.allclose(se3_inverse(T), np.linalg.inv(T), atol=1e-10)


def test_adjoint_identity(rng):
    """T exp(xi) T^-1 == exp(Adj(T) xi) is the defining property of the adjoint."""
    for _ in range(30):
        T = se3_exp(rng.normal(size=6))
        xi = rng.normal(size=6) * 0.1
        left = T @ se3_exp(xi) @ se3_inverse(T)
        right = se3_exp(se3_adjoint(T) @ xi)
        assert np.allclose(left, right, atol=1e-11)


def test_left_jacobian_matches_central_differences(rng):
    """J_l is defined by exp((xi+d)^) ~ exp((J_l d)^) exp(xi^)."""
    eps = 1e-5
    worst = 0.0
    for scale in (0.05, 0.5, 1.5, 2.8):
        for _ in range(5):
            xi = rng.normal(size=6)
            xi[3:] = xi[3:] / np.linalg.norm(xi[3:]) * scale
            analytic = se3_left_jacobian(xi)
            T_inv = se3_inverse(se3_exp(xi))
            numeric = np.zeros((6, 6))
            for k in range(6):
                d = np.zeros(6)
                d[k] = eps
                numeric[:, k] = (
                    se3_log(se3_exp(xi + d) @ T_inv) - se3_log(se3_exp(xi - d) @ T_inv)
                ) / (2.0 * eps)
            worst = max(worst, float(np.abs(numeric - analytic).max()))
    assert worst < 1e-6


def test_left_jacobian_inverse(rng):
    for _ in range(50):
        xi = rng.normal(size=6) * rng.uniform(0.0, 3.0)
        product = se3_left_jacobian(xi) @ se3_left_jacobian_inverse(xi)
        assert np.allclose(product, np.eye(6), atol=1e-8)


def test_rotation_angle_and_normalisation(rng):
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0.0, np.pi)
        R = so3_exp(axis * theta)
        assert np.isclose(rotation_angle(R), theta, atol=1e-9)
        perturbed = R + rng.normal(scale=1e-4, size=(3, 3))
        fixed = normalise_rotation(perturbed)
        assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(fixed) > 0.0
