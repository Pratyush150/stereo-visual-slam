"""Frame-to-frame motion estimation.

Three estimators live here, in the order the pipeline reaches for them:

1. **PnP + RANSAC** (primary).  Stereo gives metric 3D points from the previous
   frame; matching them into the current image gives 3D-2D correspondences, and
   PnP recovers the motion with true scale.  RANSAC is essential -- a handful of
   matches on a moving car or a bad stereo match will otherwise pull the
   solution several degrees off.
2. **Motion-only Gauss-Newton** (refinement).  RANSAC returns the model of the
   best consensus set, not the best fit to it.  A few Gauss-Newton iterations on
   the inliers with a Huber kernel typically halve the reprojection error.  This
   is implemented here rather than delegated, because it is the step where the
   analytic pose Jacobian earns its keep.
3. **Essential matrix** (monocular fallback).  Used when stereo depth is
   unavailable.  It recovers rotation and a *unit* translation direction -- the
   scale is genuinely unobservable from one camera, which is the reason this
   package is stereo in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reprojection import (
    project,
    reprojection_jacobians,
    reprojection_residual,
    transform_points,
)
from ..se3 import normalise_rotation, se3_exp, se3_inverse

try:  # pragma: no cover
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

__all__ = [
    "OdometryConfig",
    "is_plausible_motion",
    "PoseEstimate",
    "huber_weights",
    "refine_pose_gauss_newton",
    "estimate_pose_pnp",
    "estimate_essential_motion",
    "KeyframePolicy",
]


@dataclass(frozen=True)
class OdometryConfig:
    """Thresholds for PnP, RANSAC and the Gauss-Newton refinement."""

    ransac_reprojection_error: float = 2.0
    ransac_iterations: int = 200
    ransac_confidence: float = 0.999
    min_inliers: int = 20
    #: Largest believable camera translation between consecutive frames, metres.
    max_translation_per_frame: float = 4.0
    #: Largest believable camera rotation between consecutive frames, radians.
    max_rotation_per_frame: float = 0.35
    gn_iterations: int = 10
    gn_huber_delta: float = 2.0
    gn_convergence: float = 1e-8
    #: Inlier gate applied after refinement, in pixels.
    final_inlier_threshold: float = 3.0


@dataclass
class PoseEstimate:
    """Estimated ``T_cw`` plus the diagnostics that say whether to trust it."""

    T_cw: np.ndarray
    inliers: np.ndarray
    success: bool
    mean_reprojection_error: float = float("nan")
    n_correspondences: int = 0

    @property
    def T_wc(self) -> np.ndarray:
        """Camera pose in the world frame."""
        return se3_inverse(self.T_cw)


def is_plausible_motion(
    relative: np.ndarray, max_translation: float, max_rotation: float
) -> bool:
    """Reject a frame-to-frame motion that physics rules out.

    PnP occasionally returns a confidently wrong pose -- a sharp turn leaves few
    matches, RANSAC finds a small consensus set among them, and the answer is a
    jump of several metres in one 100 ms frame.  Nothing downstream notices:
    reprojection error is low on the consensus set, the pose is finite, and the
    trajectory is destroyed from that frame onwards.  Measuring one such failure
    on KITTI drive 0027 cost more than ten percentage points of translation
    error over the whole sequence.

    The gate is deliberately loose -- 4 m in a 10 Hz frame is 144 km/h -- because
    its job is to catch the impossible, not to smooth the merely surprising.
    """
    T = np.asarray(relative, dtype=float)
    if not np.all(np.isfinite(T)):
        return False
    if float(np.linalg.norm(T[:3, 3])) > max_translation:
        return False
    cos_angle = np.clip((np.trace(T[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_angle)) <= max_rotation


def huber_weights(residual_norms: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights for the Huber kernel.

    Below ``delta`` the cost is quadratic and the weight is 1; above it the cost
    grows linearly and the weight decays as ``delta / |r|``.  A gross outlier
    therefore contributes a bounded gradient instead of dominating the normal
    equations.
    """
    r = np.asarray(residual_norms, dtype=float)
    return np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))


def refine_pose_gauss_newton(
    T_cw_init: np.ndarray,
    points_world: np.ndarray,
    observations: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    baseline: float | None = None,
    iterations: int = 10,
    huber_delta: float = 2.0,
    convergence: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Motion-only refinement by robust Gauss-Newton on SE(3).

    Solves ``(J^T W J) delta = -J^T W r`` and applies the left update
    ``T <- exp(delta) T``, iterating until the step is negligible.  ``W`` comes
    from :func:`huber_weights`, recomputed each iteration (IRLS).

    A small ridge term is added to the 6x6 normal matrix.  With few points, or
    points that are nearly coplanar and far away, ``J^T W J`` is close to
    singular and an unregularised solve produces an enormous step that throws
    the pose away entirely.

    Returns
    -------
    ``(T_cw, per_point_residual_norm, mean_error)``.
    """
    T = np.array(T_cw_init, dtype=float)
    points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
    observations = np.asarray(observations, dtype=float)
    n = points_world.shape[0]
    if n == 0:
        return T, np.zeros(0), float("nan")

    norms = np.zeros(n)
    for _ in range(max(int(iterations), 1)):
        residual = reprojection_residual(
            T, points_world, observations, fx, fy, cx, cy, baseline
        )
        J_pose, _ = reprojection_jacobians(T, points_world, fx, fy, baseline)
        norms = np.linalg.norm(residual, axis=1)
        w = huber_weights(norms, huber_delta)

        # H = sum_i w_i J_i^T J_i ; g = sum_i w_i J_i^T r_i
        H = np.einsum("n,nki,nkj->ij", w, J_pose, J_pose)
        g = np.einsum("n,nki,nk->i", w, J_pose, residual)
        H += np.eye(6) * (1e-9 + 1e-9 * np.trace(H))
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            break
        T = se3_exp(delta) @ T
        T[:3, :3] = normalise_rotation(T[:3, :3])
        if float(np.linalg.norm(delta)) < convergence:
            break

    residual = reprojection_residual(T, points_world, observations, fx, fy, cx, cy, baseline)
    norms = np.linalg.norm(residual, axis=1)
    return T, norms, float(np.mean(norms)) if n else float("nan")


def estimate_pose_pnp(
    points_world: np.ndarray,
    observations_uv: np.ndarray,
    K: np.ndarray,
    config: OdometryConfig | None = None,
    *,
    T_cw_guess: np.ndarray | None = None,
    baseline: float | None = None,
    observations_stereo: np.ndarray | None = None,
) -> PoseEstimate:
    """Robust 3D-2D pose estimation: RANSAC PnP, then Gauss-Newton on the inliers.

    ``cv2.solvePnPRansac`` is used only to find the consensus set and a starting
    pose.  The pose that is actually returned comes from
    :func:`refine_pose_gauss_newton` over those inliers, using the stereo
    residual when a baseline is supplied.

    Note the deliberate two-stage inlier logic: RANSAC's set is recomputed after
    refinement against ``final_inlier_threshold``, because refinement usually
    pulls a few borderline points back in.
    """
    config = config or OdometryConfig()
    points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
    observations_uv = np.asarray(observations_uv, dtype=float).reshape(-1, 2)
    n = points_world.shape[0]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if n < 4 or cv2 is None:
        return PoseEstimate(np.eye(4), np.zeros(n, bool), False, n_correspondences=n)

    rvec0 = tvec0 = None
    use_guess = False
    if T_cw_guess is not None:
        T_guess = np.asarray(T_cw_guess, dtype=float)
        rvec0 = cv2.Rodrigues(np.ascontiguousarray(T_guess[:3, :3]))[0]
        tvec0 = np.ascontiguousarray(T_guess[:3, 3]).reshape(3, 1)
        use_guess = True

    ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
        points_world.astype(np.float64),
        observations_uv.astype(np.float64),
        np.asarray(K, dtype=np.float64),
        None,
        rvec=rvec0,
        tvec=tvec0,
        useExtrinsicGuess=use_guess,
        iterationsCount=config.ransac_iterations,
        reprojectionError=config.ransac_reprojection_error,
        confidence=config.ransac_confidence,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inlier_idx is None or len(inlier_idx) < config.min_inliers:
        return PoseEstimate(np.eye(4), np.zeros(n, bool), False, n_correspondences=n)

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)

    inliers = np.zeros(n, dtype=bool)
    inliers[np.asarray(inlier_idx).reshape(-1)] = True

    obs = observations_uv
    base = None
    if baseline is not None and observations_stereo is not None:
        obs = np.asarray(observations_stereo, dtype=float).reshape(-1, 3)
        base = baseline

    T, _, _ = refine_pose_gauss_newton(
        T,
        points_world[inliers],
        obs[inliers],
        fx, fy, cx, cy,
        baseline=base,
        iterations=config.gn_iterations,
        huber_delta=config.gn_huber_delta,
        convergence=config.gn_convergence,
    )

    # Re-gate every correspondence against the refined pose.
    predicted = project(transform_points(T, points_world), fx, fy, cx, cy)
    err = np.linalg.norm(predicted - observations_uv, axis=1)
    inliers = err < config.final_inlier_threshold
    if int(inliers.sum()) < config.min_inliers:
        return PoseEstimate(T, inliers, False, float(np.mean(err)), n)
    return PoseEstimate(T, inliers, True, float(np.mean(err[inliers])), n)


def estimate_essential_motion(
    uv_prev: np.ndarray,
    uv_curr: np.ndarray,
    K: np.ndarray,
    config: OdometryConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monocular relative motion from the essential matrix.

    Returns ``(R, unit_t, inlier_mask)`` for the transform taking the previous
    camera frame into the current one.  ``unit_t`` has norm 1 by construction:
    a single camera cannot observe how far it travelled, only in which
    direction.  Any monocular pipeline has to get scale from somewhere else, and
    whatever it uses will drift.  That is the failure mode stereo removes.
    """
    config = config or OdometryConfig()
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for the essential-matrix path")

    uv_prev = np.asarray(uv_prev, dtype=np.float64).reshape(-1, 2)
    uv_curr = np.asarray(uv_curr, dtype=np.float64).reshape(-1, 2)
    if uv_prev.shape[0] < 5:
        return np.eye(3), np.zeros(3), np.zeros(uv_prev.shape[0], bool)

    E, mask = cv2.findEssentialMat(
        uv_prev, uv_curr, np.asarray(K, dtype=np.float64),
        method=cv2.RANSAC,
        prob=config.ransac_confidence,
        threshold=config.ransac_reprojection_error,
    )
    if E is None or E.shape[0] < 3:
        return np.eye(3), np.zeros(3), np.zeros(uv_prev.shape[0], bool)
    E = E[:3, :3]
    _, R, t, pose_mask = cv2.recoverPose(E, uv_prev, uv_curr, np.asarray(K, dtype=np.float64), mask=mask)
    t = t.reshape(3)
    norm = float(np.linalg.norm(t))
    return R, (t / norm if norm > 0 else t), (pose_mask.reshape(-1) > 0)


@dataclass
class KeyframePolicy:
    """Decides when the current frame becomes a keyframe.

    Keyframes are inserted when *any* of these fires:

    * the tracked-feature count falls below ``min_tracked`` -- the map is about
      to lose its grip on the scene;
    * the ratio of currently tracked features to the last keyframe's count drops
      below ``max_tracked_ratio``, meaning the view has substantially changed;
    * the camera has moved more than ``translation_threshold`` metres or rotated
      more than ``rotation_threshold`` radians since the last keyframe;
    * ``max_frames`` frames have elapsed regardless, so a stationary vehicle at
      a junction still anchors the graph.

    Keyframing on distance alone is a common mistake: stop at a red light and no
    keyframe is ever inserted, so when you move again the frontend is matching
    against a very old frame.
    """

    min_tracked: int = 60
    max_tracked_ratio: float = 0.5
    translation_threshold: float = 3.0
    #: Rotation is kept tighter than translation on purpose: a turn destroys
    #: overlap with the reference keyframe far faster than driving straight does,
    #: and a stale reference during a turn is where tracking actually breaks.
    rotation_threshold: float = 0.12
    max_frames: int = 8
    min_frames: int = 2

    def should_insert(
        self,
        frames_since_keyframe: int,
        n_tracked: int,
        n_keyframe_features: int,
        relative_pose: np.ndarray,
    ) -> bool:
        """Return True if the current frame should become a keyframe."""
        if frames_since_keyframe < self.min_frames:
            return False
        if frames_since_keyframe >= self.max_frames:
            return True
        if n_tracked < self.min_tracked:
            return True
        if n_keyframe_features > 0 and n_tracked < self.max_tracked_ratio * n_keyframe_features:
            return True
        T = np.asarray(relative_pose, dtype=float)
        translation = float(np.linalg.norm(T[:3, 3]))
        cos_angle = np.clip((np.trace(T[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        rotation = float(np.arccos(cos_angle))
        return translation > self.translation_threshold or rotation > self.rotation_threshold
