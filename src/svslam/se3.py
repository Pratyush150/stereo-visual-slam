"""Minimal SE(3)/SO(3) Lie group utilities.

Everything downstream (bundle adjustment, pose graph, odometry refinement)
optimises over the tangent space of SE(3), so the exponential and logarithm
maps here are the numerical foundation of the whole package.

Conventions
-----------
* A pose is a 4x4 homogeneous matrix ``T`` mapping *points expressed in the
  body/camera frame* into the world frame, i.e. ``p_world = T @ p_cam``.
* A tangent vector is ordered ``xi = [rho (3,), phi (3,)]`` -- translation
  part first, rotation part second.
* Increments are applied with a **left perturbation**::

      T <- exp(xi^) @ T

  which is what the analytic Jacobians in :mod:`svslam.backend.ba` assume.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "skew",
    "unskew",
    "so3_exp",
    "so3_log",
    "se3_exp",
    "se3_log",
    "se3_inverse",
    "se3_adjoint",
    "left_jacobian_inverse_so3",
    "normalise_rotation",
    "rotation_angle",
]

_EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def unskew(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew` (takes the skew-symmetric part)."""
    m = np.asarray(m, dtype=float)
    return np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) * 0.5


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map from a rotation vector to a 3x3 rotation.

    A Taylor expansion is used for small angles so the map stays accurate and
    finite as ``|phi| -> 0``.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        # exp(K) = I + K + K^2/2 + ... ; two terms are already at machine
        # precision for theta < 1e-7.
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def rotation_angle(R: np.ndarray) -> float:
    """Return the rotation angle of ``R`` in radians, clamped to [0, pi]."""
    c = (np.trace(np.asarray(R, dtype=float)) - 1.0) * 0.5
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map from a 3x3 rotation to a rotation vector.

    Implemented through a quaternion (Shepperd's method) rather than the
    textbook ``theta / (2 sin theta) * (R - R^T)`` form.  That form divides by
    ``sin theta`` and loses every significant digit as ``theta -> pi``, which
    is exactly where a vehicle doing a U-turn puts you.  The quaternion route
    stays accurate over the whole range, including ``theta = 0`` and
    ``theta = pi``.
    """
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        v = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        v = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        v = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    n = float(np.linalg.norm(v))
    if n < 1e-14:
        # Identity rotation (or numerically indistinguishable from it).
        return np.zeros(3)
    # atan2 keeps the angle well conditioned at both ends of [0, pi].
    theta = 2.0 * np.arctan2(n, abs(w))
    axis = v / n
    if w < 0.0:
        axis = -axis
    return axis * theta


def _left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian ``J`` of SO(3) such that ``exp((phi+d)^) ~ exp((J d)^) exp(phi^)``."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    a = (1.0 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) + a * K + b * (K @ K)


def left_jacobian_inverse_so3(phi: np.ndarray) -> np.ndarray:
    """Inverse of the SO(3) left Jacobian.

    The usual coefficient ``1/theta^2 - (1 + cos theta) / (2 theta sin theta)``
    is rewritten with a half-angle identity as
    ``1/theta^2 - cot(theta/2) / (2 theta)``.  Algebraically identical, but the
    original cancels ``1 + cos theta`` to zero in floating point near
    ``theta = pi``; the cotangent form is exact there.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    c = 1.0 / (theta * theta) - (np.cos(half) / np.sin(half)) / (2.0 * theta)
    return np.eye(3) - 0.5 * K + c * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from ``xi = [rho, phi]`` to a 4x4 SE(3) matrix."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    V = _left_jacobian_so3(phi)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map from a 4x4 SE(3) matrix to ``xi = [rho, phi]``."""
    T = np.asarray(T, dtype=float)
    phi = so3_log(T[:3, :3])
    rho = left_jacobian_inverse_so3(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform, without a general matrix inverse."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint matrix of ``T`` for the ``[rho, phi]`` tangent ordering."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[:3, 3:] = skew(t) @ R
    adj[3:, 3:] = R
    return adj


def normalise_rotation(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix back onto SO(3) via SVD.

    Repeated incremental updates slowly break orthonormality; calling this
    after every optimisation step keeps poses on the manifold.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    R_out = U @ Vt
    if np.linalg.det(R_out) < 0.0:
        U[:, -1] *= -1.0
        R_out = U @ Vt
    return R_out


def _q_matrix(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """The coupling block ``Q`` of the SE(3) left Jacobian.

    ``J_l(xi) = [[J, Q], [0, J]]`` for the ``[rho, phi]`` ordering, where ``J``
    is the SO(3) left Jacobian.  ``Q`` is the term that makes SE(3) genuinely
    different from SO(3) x R^3: it says how a rotation increment drags the
    translation part around.  Series coefficients are used below the threshold,
    where the closed forms cancel catastrophically.
    """
    rho = np.asarray(rho, dtype=float).reshape(3)
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    rx = skew(rho)
    px = skew(phi)

    # Series below the threshold: the closed forms all cancel to O(theta^4)
    # in a numerator of O(1), so they lose every digit near the origin.
    if theta < 0.1:
        t2 = theta * theta
        t4 = t2 * t2
        c1 = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0
        c2 = 1.0 / 24.0 - t2 / 720.0 + t4 / 40320.0
        c3 = 1.0 / 120.0 - t2 / 2520.0 + t4 / 60480.0
    else:
        s_t, c_t = np.sin(theta), np.cos(theta)
        t2, t3, t4, t5 = theta ** 2, theta ** 3, theta ** 4, theta ** 5
        c1 = (theta - s_t) / t3
        c2 = (t2 + 2.0 * c_t - 2.0) / (2.0 * t4)
        c3 = (2.0 * theta - 3.0 * s_t + theta * c_t) / (2.0 * t5)

    term1 = px @ rx + rx @ px + px @ rx @ px
    term2 = px @ px @ rx + rx @ px @ px - 3.0 * (px @ rx @ px)
    term3 = px @ rx @ px @ px + px @ px @ rx @ px
    return 0.5 * rx + c1 * term1 + c2 * term2 + c3 * term3


def se3_left_jacobian(xi: np.ndarray) -> np.ndarray:
    """6x6 left Jacobian of SE(3), defined by ``exp((xi + d)^) ~ exp((J d)^) exp(xi^)``."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    J = _left_jacobian_so3(phi)
    Q = _q_matrix(rho, phi)
    out = np.zeros((6, 6))
    out[:3, :3] = J
    out[:3, 3:] = Q
    out[3:, 3:] = J
    return out


def se3_left_jacobian_inverse(xi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`se3_left_jacobian`, in closed form.

    Block inversion of an upper block-triangular matrix::

        [[J, Q], [0, J]]^-1 = [[J^-1, -J^-1 Q J^-1], [0, J^-1]]
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    Ji = left_jacobian_inverse_so3(phi)
    Q = _q_matrix(rho, phi)
    out = np.zeros((6, 6))
    out[:3, :3] = Ji
    out[:3, 3:] = -Ji @ Q @ Ji
    out[3:, 3:] = Ji
    return out


def se3_right_jacobian_inverse(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian inverse, which is the left one evaluated at ``-xi``."""
    return se3_left_jacobian_inverse(-np.asarray(xi, dtype=float).reshape(6))
