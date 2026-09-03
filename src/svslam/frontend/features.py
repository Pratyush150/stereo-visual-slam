"""Feature detection, spatial bucketing, matching and KLT tracking.

The single most common way a hand-rolled visual odometry pipeline goes wrong is
not the solver -- it is the feature distribution.  A corner detector run over a
street scene puts most of its strongest responses on one high-contrast region:
a brick wall, a row of parked cars, a tree line.  Feed those to PnP and the
pose is well constrained along one direction and nearly unconstrained along the
others, so the estimate wanders even though the reprojection error looks fine.

The fix is bucketing: divide the image into a grid and cap how many features
any one cell may contribute.  You keep fewer features overall and they are
individually weaker, but they span the image, and the pose is better
conditioned.  :func:`spatial_spread` quantifies the difference so the effect can
be measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

__all__ = [
    "FeatureConfig",
    "FeatureSet",
    "require_cv2",
    "bucket_indices",
    "bucket_keypoints",
    "spatial_spread",
    "OrbDetector",
    "match_descriptors",
    "track_klt",
    "keypoints_to_array",
]


def require_cv2() -> None:
    """Raise a clear error if OpenCV is missing, instead of an AttributeError."""
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for feature extraction; install opencv-python"
        )


@dataclass(frozen=True)
class FeatureConfig:
    """Tuning for the detector and the bucketing grid.

    ``max_features`` is the budget *after* bucketing.  ``grid_rows`` x
    ``grid_cols`` cells each keep at most ``max_per_cell`` features, ranked by
    detector response.
    """

    max_features: int = 1500
    grid_rows: int = 5
    grid_cols: int = 12
    max_per_cell: int = 30
    fast_threshold: int = 12
    scale_factor: float = 1.2
    n_levels: int = 8
    use_bucketing: bool = True
    ratio_test: float = 0.75
    max_hamming: int = 64


@dataclass
class FeatureSet:
    """Keypoint pixel coordinates plus their descriptors and responses."""

    points: np.ndarray  # (N, 2) float32, (u, v)
    descriptors: np.ndarray  # (N, 32) uint8 for ORB
    responses: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def subset(self, idx: np.ndarray) -> "FeatureSet":
        idx = np.asarray(idx, dtype=int)
        return FeatureSet(self.points[idx], self.descriptors[idx], self.responses[idx])


def keypoints_to_array(keypoints) -> np.ndarray:
    """cv2 KeyPoint sequence to an ``(N, 2)`` float32 array of ``(u, v)``."""
    if len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def bucket_indices(
    points: np.ndarray,
    responses: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int,
    grid_cols: int,
    max_per_cell: int,
    max_total: int | None = None,
) -> np.ndarray:
    """Select feature indices so no grid cell contributes more than a cap.

    Within a cell, features are ranked by detector response and the strongest
    ``max_per_cell`` survive.  If ``max_total`` is given the survivors are then
    trimmed globally, again by response, so the budget is honoured.

    Returns
    -------
    Sorted array of indices into ``points``.
    """
    points = np.asarray(points, dtype=float)
    responses = np.asarray(responses, dtype=float)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=int)

    height, width = image_shape[:2]
    cell_h = max(height / float(grid_rows), 1e-9)
    cell_w = max(width / float(grid_cols), 1e-9)
    row = np.clip((points[:, 1] / cell_h).astype(int), 0, grid_rows - 1)
    col = np.clip((points[:, 0] / cell_w).astype(int), 0, grid_cols - 1)
    cell = row * grid_cols + col

    # Sort by (cell, -response) so each cell's best features come first, then
    # take the first max_per_cell of every run.
    order = np.lexsort((-responses, cell))
    sorted_cells = cell[order]
    # Rank within each cell run.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1]
    rank = np.arange(order.size) - np.repeat(starts, np.diff(np.r_[starts, order.size]))
    keep = order[rank < max_per_cell]

    if max_total is not None and keep.size > max_total:
        keep = keep[np.argsort(-responses[keep])[:max_total]]
    return np.sort(keep)


def bucket_keypoints(
    features: FeatureSet, image_shape: tuple[int, int], config: FeatureConfig
) -> FeatureSet:
    """Apply :func:`bucket_indices` to a :class:`FeatureSet`."""
    idx = bucket_indices(
        features.points,
        features.responses,
        image_shape,
        config.grid_rows,
        config.grid_cols,
        config.max_per_cell,
        config.max_features,
    )
    return features.subset(idx)


def spatial_spread(
    points: np.ndarray,
    image_shape: tuple[int, int],
    grid_rows: int = 5,
    grid_cols: int = 12,
) -> dict[str, float]:
    """Measure how evenly a point set covers the image.

    Returns three numbers over the same grid used for bucketing:

    ``occupancy``
        Fraction of cells holding at least one point.  1.0 means full coverage.
    ``normalised_entropy``
        Shannon entropy of the per-cell counts divided by ``log(n_cells)``.
        1.0 is a perfectly uniform spread; a detector dumping everything into
        one cell scores 0.
    ``max_cell_fraction``
        Share of all points falling in the single busiest cell.  This is the
        number that spikes when features clump on one textured wall.
    """
    points = np.asarray(points, dtype=float)
    n_cells = grid_rows * grid_cols
    if points.shape[0] == 0:
        return {"occupancy": 0.0, "normalised_entropy": 0.0, "max_cell_fraction": 0.0}

    height, width = image_shape[:2]
    row = np.clip((points[:, 1] / max(height / grid_rows, 1e-9)).astype(int), 0, grid_rows - 1)
    col = np.clip((points[:, 0] / max(width / grid_cols, 1e-9)).astype(int), 0, grid_cols - 1)
    counts = np.bincount(row * grid_cols + col, minlength=n_cells).astype(float)

    total = counts.sum()
    p = counts[counts > 0] / total
    entropy = float(-(p * np.log(p)).sum())
    return {
        "occupancy": float((counts > 0).sum() / n_cells),
        "normalised_entropy": entropy / float(np.log(n_cells)) if n_cells > 1 else 0.0,
        "max_cell_fraction": float(counts.max() / total),
    }


class OrbDetector:
    """ORB detection with optional bucketing.

    A large raw budget is requested from ORB and then thinned by the bucketer,
    which works far better than asking ORB for few features directly -- ORB's
    own retention is by response only, so a small budget clumps.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        require_cv2()
        self.config = config or FeatureConfig()
        # Ask for headroom so bucketing has candidates in the weak cells.
        raw_budget = max(self.config.max_features * 3, 2000)
        self._orb = cv2.ORB_create(
            nfeatures=raw_budget,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            fastThreshold=self.config.fast_threshold,
        )

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> FeatureSet:
        """Detect and describe, then bucket if enabled."""
        keypoints, descriptors = self._orb.detectAndCompute(image, mask)
        if descriptors is None or len(keypoints) == 0:
            return FeatureSet(
                np.zeros((0, 2), np.float32),
                np.zeros((0, 32), np.uint8),
                np.zeros(0, np.float32),
            )
        features = FeatureSet(
            keypoints_to_array(keypoints),
            descriptors,
            np.array([kp.response for kp in keypoints], dtype=np.float32),
        )
        if not self.config.use_bucketing:
            if len(features) > self.config.max_features:
                idx = np.argsort(-features.responses)[: self.config.max_features]
                features = features.subset(np.sort(idx))
            return features
        return bucket_keypoints(features, image.shape, self.config)


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.75,
    max_distance: int = 64,
    cross_check: bool = True,
) -> np.ndarray:
    """Brute-force Hamming matching with Lowe's ratio test.

    Two filters, both cheap and both worth having:

    * **Ratio test** -- reject a match whose best and second-best distances are
      similar, because that means the descriptor is ambiguous (repeated texture,
      road markings, railings).
    * **Cross-check** -- keep a match only if it is mutually best.  This removes
      the many-to-one matches that otherwise drag RANSAC around.

    Returns
    -------
    ``(M, 2)`` int array of index pairs into ``desc_a`` and ``desc_b``.
    """
    require_cv2()
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return np.zeros((0, 2), dtype=int)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    pairs: list[tuple[int, int]] = []
    for candidates in knn:
        if not candidates:
            continue
        best = candidates[0]
        if best.distance > max_distance:
            continue
        if len(candidates) > 1 and best.distance > ratio * candidates[1].distance:
            continue
        pairs.append((best.queryIdx, best.trainIdx))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    matches = np.array(pairs, dtype=int)

    if cross_check:
        knn_back = matcher.knnMatch(desc_b, desc_a, k=1)
        back = {m[0].queryIdx: m[0].trainIdx for m in knn_back if m}
        keep = np.array([back.get(int(j), -1) == int(i) for i, j in matches])
        matches = matches[keep]
    return matches


def track_klt(
    prev_image: np.ndarray,
    next_image: np.ndarray,
    prev_points: np.ndarray,
    *,
    window: int = 21,
    levels: int = 3,
    fb_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade tracking with a forward-backward consistency check.

    Tracking forwards then backwards and requiring the round trip to land within
    ``fb_threshold`` pixels removes most of the tracks that slid onto an
    occluding edge or drifted along a repeated texture.  Those are exactly the
    tracks that survive the status flag but are geometrically wrong.

    Returns
    -------
    ``(next_points, valid_mask)`` where ``next_points`` has the same length as
    ``prev_points`` and ``valid_mask`` is a boolean array.
    """
    require_cv2()
    prev_points = np.asarray(prev_points, dtype=np.float32).reshape(-1, 1, 2)
    if prev_points.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    params = dict(winSize=(window, window), maxLevel=levels, criteria=criteria)

    fwd, status_f, _ = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points, None, **params)
    back, status_b, _ = cv2.calcOpticalFlowPyrLK(next_image, prev_image, fwd, None, **params)

    error = np.linalg.norm(prev_points - back, axis=2).reshape(-1)
    valid = (
        (status_f.reshape(-1) == 1)
        & (status_b.reshape(-1) == 1)
        & (error < fb_threshold)
    )
    h, w = next_image.shape[:2]
    pts = fwd.reshape(-1, 2)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    return pts, valid & inside
