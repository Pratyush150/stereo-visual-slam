"""The official KITTI odometry metrics, plus ATE and RPE.

Why not just report the final position error?  Because it is meaningless.  A
trajectory that drifts steadily and one that is perfect for 90% of the sequence
and then jumps can end in the same place.  The KITTI benchmark instead measures
error *per unit of distance travelled*, over many sub-sequences of many lengths,
which is why its numbers transfer between sequences and are worth comparing.

The official definition
-----------------------
For every sub-sequence length in ``{100, 200, ..., 800}`` metres and every start
frame, find the end frame at which the ground-truth path length first reaches
that length.  Compute the relative pose error::

    E = (gt_i^-1 gt_j)^-1 . (est_i^-1 est_j)

Translation error is ``||trans(E)|| / length`` and rotation error is
``angle(rot(E)) / length``.  Average over every sub-sequence of every length.
The published figures are the translation error as a **percentage** and the
rotation error in **degrees per metre**.

Two details that are easy to get wrong and that change the answer:

* the start frames step through the sequence at a fixed *frame* stride (10 in
  the official tools), not a fixed distance;
* a sub-sequence is skipped entirely if the trajectory ends before the required
  length is reached, rather than being truncated -- truncating biases short
  sequences optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..se3 import rotation_angle, se3_inverse

__all__ = [
    "KITTI_LENGTHS",
    "trajectory_distances",
    "last_frame_from_segment_length",
    "SegmentError",
    "kitti_odometry_errors",
    "kitti_summary",
    "umeyama_alignment",
    "absolute_trajectory_error",
    "relative_pose_error",
    "MetricsReport",
    "evaluate",
]

#: Sub-sequence lengths in metres, exactly as the benchmark defines them.
KITTI_LENGTHS: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)

#: Frame stride between sub-sequence start points, as in the official tools.
DEFAULT_STEP = 10


def trajectory_distances(poses: np.ndarray) -> np.ndarray:
    """Cumulative path length at each pose, in metres. ``distances[0] == 0``."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    if poses.shape[0] == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def last_frame_from_segment_length(
    distances: np.ndarray, first_frame: int, length: float
) -> int:
    """First frame at or beyond ``length`` metres after ``first_frame``, or -1."""
    target = distances[first_frame] + length
    idx = int(np.searchsorted(distances, target, side="left"))
    return idx if idx < distances.size else -1


@dataclass
class SegmentError:
    """One sub-sequence's error, before averaging."""

    first_frame: int
    length: float
    #: Translation error per metre (multiply by 100 for the published percentage).
    translation: float
    #: Rotation error in radians per metre.
    rotation: float
    speed: float


def kitti_odometry_errors(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    lengths: tuple[int, ...] = KITTI_LENGTHS,
    step: int = DEFAULT_STEP,
    frame_rate: float = 10.0,
) -> list[SegmentError]:
    """Per-sub-sequence errors, following the official development kit."""
    estimated = np.asarray(estimated, dtype=float).reshape(-1, 4, 4)
    ground_truth = np.asarray(ground_truth, dtype=float).reshape(-1, 4, 4)
    n = min(estimated.shape[0], ground_truth.shape[0])
    estimated, ground_truth = estimated[:n], ground_truth[:n]
    distances = trajectory_distances(ground_truth)

    errors: list[SegmentError] = []
    for first in range(0, n, max(int(step), 1)):
        for length in lengths:
            last = last_frame_from_segment_length(distances, first, float(length))
            if last < 0 or last >= n:
                continue
            delta_gt = se3_inverse(ground_truth[first]) @ ground_truth[last]
            delta_est = se3_inverse(estimated[first]) @ estimated[last]
            error = se3_inverse(delta_gt) @ delta_est
            t_err = float(np.linalg.norm(error[:3, 3]))
            r_err = rotation_angle(error[:3, :3])
            n_frames = max(last - first, 1)
            errors.append(
                SegmentError(
                    first_frame=first,
                    length=float(length),
                    translation=t_err / length,
                    rotation=r_err / length,
                    speed=length / (n_frames / max(frame_rate, 1e-9)),
                )
            )
    return errors


def kitti_summary(errors: list[SegmentError]) -> dict[str, float]:
    """Average the sub-sequence errors into the two published numbers."""
    if not errors:
        return {"translation_percent": float("nan"), "rotation_deg_per_m": float("nan"),
                "n_segments": 0}
    t = float(np.mean([e.translation for e in errors]))
    r = float(np.mean([e.rotation for e in errors]))
    return {
        "translation_percent": t * 100.0,
        "rotation_deg_per_m": np.rad2deg(r),
        "n_segments": len(errors),
    }


def umeyama_alignment(
    source: np.ndarray, target: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Umeyama's closed-form solution.  ``with_scale=True`` also solves for a scale
    factor, which is what a monocular trajectory needs before it can be compared
    to anything -- for a stereo system, scale is observed, so leaving scale at 1
    is the honest comparison and a scale far from 1 is itself a finding.

    Returns ``(R, t, s)`` such that ``target ~ s R source + t``.
    """
    src = np.asarray(source, dtype=float).reshape(-1, 3)
    dst = np.asarray(target, dtype=float).reshape(-1, 3)
    n = min(src.shape[0], dst.shape[0])
    src, dst = src[:n], dst[:n]
    if n == 0:
        return np.eye(3), np.zeros(3), 1.0

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    sigma = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = 1.0
    if with_scale:
        var_s = float((sc ** 2).sum() / n)
        scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return R, t, scale


def absolute_trajectory_error(
    estimated: np.ndarray, ground_truth: np.ndarray, align: bool = True,
    with_scale: bool = False,
) -> dict[str, float]:
    """ATE: RMS position difference after an optional rigid alignment."""
    est = np.asarray(estimated, dtype=float).reshape(-1, 4, 4)
    gt = np.asarray(ground_truth, dtype=float).reshape(-1, 4, 4)
    n = min(est.shape[0], gt.shape[0])
    p_est, p_gt = est[:n, :3, 3], gt[:n, :3, 3]
    if n == 0:
        return {"rmse": float("nan"), "mean": float("nan"), "median": float("nan"),
                "max": float("nan"), "scale": 1.0}

    scale = 1.0
    if align:
        R, t, scale = umeyama_alignment(p_est, p_gt, with_scale)
        p_est = (scale * (R @ p_est.T).T) + t
    d = np.linalg.norm(p_est - p_gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
        "scale": float(scale),
    }


def relative_pose_error(
    estimated: np.ndarray, ground_truth: np.ndarray, delta: int = 1
) -> dict[str, float]:
    """RPE over a fixed frame gap: local drift rate, insensitive to global drift."""
    est = np.asarray(estimated, dtype=float).reshape(-1, 4, 4)
    gt = np.asarray(ground_truth, dtype=float).reshape(-1, 4, 4)
    n = min(est.shape[0], gt.shape[0])
    delta = max(int(delta), 1)
    if n <= delta:
        return {"translation_rmse": float("nan"), "rotation_rmse_deg": float("nan"),
                "n_pairs": 0}

    t_err, r_err = [], []
    for i in range(n - delta):
        j = i + delta
        d_gt = se3_inverse(gt[i]) @ gt[j]
        d_est = se3_inverse(est[i]) @ est[j]
        e = se3_inverse(d_gt) @ d_est
        t_err.append(np.linalg.norm(e[:3, 3]))
        r_err.append(rotation_angle(e[:3, :3]))
    t_err = np.array(t_err)
    r_err = np.array(r_err)
    return {
        "translation_rmse": float(np.sqrt(np.mean(t_err ** 2))),
        "rotation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(r_err ** 2)))),
        "n_pairs": int(t_err.size),
    }


@dataclass
class MetricsReport:
    """Everything :func:`evaluate` computed, ready to print or serialise."""

    translation_percent: float
    rotation_deg_per_m: float
    n_segments: int
    ate: dict[str, float]
    rpe: dict[str, float]
    path_length: float
    n_poses: int
    per_length: dict[int, dict[str, float]] = field(default_factory=dict)
    segments: list[SegmentError] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "translation_percent": self.translation_percent,
            "rotation_deg_per_m": self.rotation_deg_per_m,
            "n_segments": self.n_segments,
            "ate": self.ate,
            "rpe": self.rpe,
            "path_length_m": self.path_length,
            "n_poses": self.n_poses,
            "per_length": {str(k): v for k, v in self.per_length.items()},
        }

    def format_table(self) -> str:
        """The KITTI error-vs-length table as plain text."""
        lines = [
            "  length (m)   trans err (%)   rot err (deg/m)   segments",
            "  ----------   -------------   ---------------   --------",
        ]
        for length in sorted(self.per_length):
            row = self.per_length[length]
            lines.append(
                f"  {length:>10}   {row['translation_percent']:>13.3f}"
                f"   {row['rotation_deg_per_m']:>15.5f}   {int(row['n_segments']):>8}"
            )
        return "\n".join(lines)


def evaluate(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    *,
    lengths: tuple[int, ...] = KITTI_LENGTHS,
    step: int = DEFAULT_STEP,
    frame_rate: float = 10.0,
    align_ate: bool = True,
    ate_with_scale: bool = False,
    rpe_delta: int = 10,
) -> MetricsReport:
    """Run the full metric suite over an estimated and a ground-truth trajectory."""
    errors = kitti_odometry_errors(estimated, ground_truth, lengths, step, frame_rate)
    summary = kitti_summary(errors)

    per_length: dict[int, dict[str, float]] = {}
    for length in lengths:
        subset = [e for e in errors if int(e.length) == int(length)]
        if subset:
            per_length[int(length)] = kitti_summary(subset)

    gt = np.asarray(ground_truth, dtype=float).reshape(-1, 4, 4)
    return MetricsReport(
        translation_percent=summary["translation_percent"],
        rotation_deg_per_m=summary["rotation_deg_per_m"],
        n_segments=int(summary["n_segments"]),
        ate=absolute_trajectory_error(estimated, ground_truth, align_ate, ate_with_scale),
        rpe=relative_pose_error(estimated, ground_truth, rpe_delta),
        path_length=float(trajectory_distances(gt)[-1]) if gt.shape[0] else 0.0,
        n_poses=int(min(len(estimated), len(ground_truth))),
        per_length=per_length,
        segments=errors,
    )
