"""Reprojection residuals and their analytic Jacobians.

Every optimiser in this package -- motion-only Gauss-Newton, windowed bundle
adjustment -- minimises the same quantity, so it is defined once here.

Conventions
-----------
The optimisation variable for a camera is ``T_cw``, the transform taking a
**world** point into the **camera** frame.  Increments are left perturbations::

    T_cw <- exp(delta^) . T_cw,   delta = [rho, phi]

Applying that to a camera-frame point gives, to first order::

    p_c' = p_c + rho + phi x p_c

so the derivative of the camera-frame point with respect to the increment is
the 3x6 block ``[I | -[p_c]_x]``.  Everything else is the chain rule through the
pinhole projection.

Two residual flavours are provided:

``mono``
    ``r = [u, v] - z``, two rows.  Cannot observe scale on its own.
``stereo``
    ``r = [u_l, v, u_r] - z``, three rows, with ``u_r = u_l - fx b / Z``.  The
    third row is what pins metric scale down: it depends on ``1/Z`` directly, so
    a landmark's depth cannot drift without paying for it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "project",
    "project_stereo",
    "projection_jacobian",
    "stereo_projection_jacobian",
    "reprojection_residual",
    "reprojection_jacobians",
    "transform_points",
]


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an ``(N, 3)`` array of points."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3]


def project(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole projection of camera-frame points to ``(N, 2)`` pixels."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = p[:, 2]
    return np.column_stack([fx * p[:, 0] / z + cx, fy * p[:, 1] / z + cy])


def project_stereo(
    points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float, baseline: float
) -> np.ndarray:
    """Rectified stereo projection to ``(N, 3)`` of ``[u_left, v, u_right]``."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = p[:, 2]
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    return np.column_stack([u, v, u - fx * baseline / z])


def projection_jacobian(
    points_cam: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """``d[u, v] / d p_c`` for each point, shape ``(N, 2, 3)``."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    J = np.zeros((p.shape[0], 2, 3))
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * x * inv_z2
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * y * inv_z2
    return J


def stereo_projection_jacobian(
    points_cam: np.ndarray, fx: float, fy: float, baseline: float
) -> np.ndarray:
    """``d[u_l, v, u_r] / d p_c`` for each point, shape ``(N, 3, 3)``."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    J = np.zeros((p.shape[0], 3, 3))
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * x * inv_z2
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * y * inv_z2
    J[:, 2, 0] = fx * inv_z
    J[:, 2, 2] = -fx * x * inv_z2 + fx * baseline * inv_z2
    return J


def reprojection_residual(
    T_cw: np.ndarray,
    points_world: np.ndarray,
    observations: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float | None = None,
) -> np.ndarray:
    """Predicted minus observed pixels, shape ``(N, 2)`` or ``(N, 3)``."""
    p_cam = transform_points(T_cw, points_world)
    obs = np.asarray(observations, dtype=float)
    if baseline is None:
        return project(p_cam, fx, fy, cx, cy) - obs.reshape(-1, 2)
    return project_stereo(p_cam, fx, fy, cx, cy, baseline) - obs.reshape(-1, 3)


def reprojection_jacobians(
    T_cw: np.ndarray,
    points_world: np.ndarray,
    fx: float,
    fy: float,
    baseline: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Jacobians of the reprojection residual.

    Returns
    -------
    ``(J_pose, J_point)`` with shapes ``(N, k, 6)`` and ``(N, k, 3)``, where
    ``k`` is 2 for the monocular residual and 3 for the stereo one.

    ``J_pose`` is with respect to a **left** increment on ``T_cw``, and
    ``J_point`` with respect to the world coordinates of the landmark.  Both are
    checked against central differences in ``tests/test_jacobians.py`` -- if
    those tests pass, the derivations below are right.
    """
    T_cw = np.asarray(T_cw, dtype=float)
    R = T_cw[:3, :3]
    p_cam = transform_points(T_cw, points_world)

    if baseline is None:
        J_proj = projection_jacobian(p_cam, fx, fy)
    else:
        J_proj = stereo_projection_jacobian(p_cam, fx, fy, baseline)

    n = J_proj.shape[0]
    # d p_c / d delta = [I | -[p_c]_x], built without a Python loop because this
    # runs once per observation per bundle-adjustment iteration.
    d_pc_d_delta = np.zeros((n, 3, 6))
    d_pc_d_delta[:, :, :3] = np.eye(3)
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    d_pc_d_delta[:, 0, 4] = z
    d_pc_d_delta[:, 0, 5] = -y
    d_pc_d_delta[:, 1, 3] = -z
    d_pc_d_delta[:, 1, 5] = x
    d_pc_d_delta[:, 2, 3] = y
    d_pc_d_delta[:, 2, 4] = -x

    J_pose = np.einsum("nij,njk->nik", J_proj, d_pc_d_delta)
    J_point = np.einsum("nij,jk->nik", J_proj, R)
    return J_pose, J_point
