"""Sparse stereo matching and triangulation.

The stereo pair is rectified, so the search for a left feature's partner is one
dimensional: same row, some disparity to the left.  That is cheap, but it is
also where a stereo pipeline quietly poisons itself, for one reason.

Depth from disparity is ``Z = fx * b / d``.  Differentiate it::

    dZ/dd = -fx * b / d^2   =>   sigma_Z = Z^2 / (fx * b) * sigma_d

The uncertainty grows with the **square** of the depth.  With KITTI's numbers
(``fx = 707 px``, ``b = 0.54 m``) and a realistic ``sigma_d`` of a quarter
pixel, a point at 10 m has about 6 cm of depth uncertainty; the same point at
80 m has about 4 m.  Those far points look like perfectly good matches, they
have low reprojection error in the image, and they will drag a PnP solution
around by metres.  They have to be rejected, and the threshold has to come from
the geometry rather than from a magic number -- see :func:`min_disparity_for`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StereoConfig",
    "StereoMatches",
    "depth_from_disparity",
    "depth_sigma",
    "min_disparity_for",
    "disparity_to_points",
    "match_stereo_epipolar",
    "triangulate_features",
]


@dataclass(frozen=True)
class StereoConfig:
    """Search window and rejection thresholds for sparse stereo."""

    patch_radius: int = 5
    min_disparity: float = 1.0
    max_disparity: int = 96
    lr_max_diff: float = 1.0
    #: Assumed standard deviation of a disparity measurement, in pixels.
    disparity_sigma: float = 0.25
    #: Reject a point whose relative depth uncertainty exceeds this.
    max_relative_depth_sigma: float = 0.05
    #: Hard depth ceiling in metres, independent of the uncertainty test.
    max_depth: float = 60.0
    #: Reject a match whose SAD cost is above this (per pixel, 0-255 scale).
    max_mean_abs_diff: float = 30.0
    #: Reject a match whose best cost is not clearly better than the runner-up.
    uniqueness_ratio: float = 0.85


def depth_from_disparity(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """``Z = fx * b / d``, with non-positive disparities mapped to infinity."""
    d = np.asarray(disparity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(d > 0.0, fx * baseline / np.where(d > 0.0, d, 1.0), np.inf)
    return z


def depth_sigma(
    disparity: np.ndarray, fx: float, baseline: float, disparity_sigma: float
) -> np.ndarray:
    """First-order depth uncertainty ``sigma_Z = Z^2 sigma_d / (fx b)``."""
    z = depth_from_disparity(disparity, fx, baseline)
    return z * z * disparity_sigma / (fx * baseline)


def min_disparity_for(
    fx: float,
    baseline: float,
    disparity_sigma: float,
    max_relative_sigma: float,
    max_depth: float | None = None,
) -> float:
    """Smallest usable disparity, derived from the depth-uncertainty budget.

    Since ``sigma_Z / Z = sigma_d / d`` exactly, requiring a relative depth
    uncertainty below ``max_relative_sigma`` is the same as requiring::

        d >= sigma_d / max_relative_sigma

    which is a threshold with a meaning rather than a tuned constant.  A depth
    ceiling, if given, tightens it further via ``d >= fx * b / Z_max``.
    """
    d_min = disparity_sigma / max(max_relative_sigma, 1e-9)
    if max_depth is not None and max_depth > 0.0:
        d_min = max(d_min, fx * baseline / max_depth)
    return float(d_min)


def disparity_to_points(
    uv: np.ndarray,
    disparity: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> np.ndarray:
    """Back-project rectified pixels plus disparity to camera-frame 3D points."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d = np.asarray(disparity, dtype=float).reshape(-1)
    z = fx * baseline / d
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.column_stack([x, y, z])


@dataclass
class StereoMatches:
    """Result of sparse stereo on one frame.

    ``index`` refers back into the left :class:`~svslam.frontend.features.FeatureSet`
    that was passed in, so descriptors and track identity survive the filter.
    """

    index: np.ndarray
    uv_left: np.ndarray
    disparity: np.ndarray
    points_cam: np.ndarray
    depth: np.ndarray
    depth_sigma: np.ndarray

    def __len__(self) -> int:
        return int(self.index.shape[0])


def _gather_patches(
    image: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int
) -> np.ndarray:
    """Gather ``(2r+1)^2`` patches around integer pixel centres, as ``(N, P)``."""
    offsets = np.arange(-radius, radius + 1)
    dv, du = np.meshgrid(offsets, offsets, indexing="ij")
    rows = v[:, None] + dv.reshape(1, -1)
    cols = u[:, None] + du.reshape(1, -1)
    return image[rows, cols].astype(np.float32)


def _sad_search(
    ref_patches: np.ndarray,
    search_image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius: int,
    disparities: np.ndarray,
    sign: int,
) -> np.ndarray:
    """SAD cost of every candidate disparity, as ``(N, D)``.

    ``sign = -1`` searches leftwards (left image -> right image, the normal
    direction); ``sign = +1`` searches rightwards for the reverse check.
    """
    costs = np.full((ref_patches.shape[0], disparities.size), np.inf, dtype=np.float32)
    for k, d in enumerate(disparities):
        uu = u + sign * int(d)
        ok = (uu >= radius) & (uu < search_image.shape[1] - radius)
        if not np.any(ok):
            continue
        cand = _gather_patches(search_image, uu[ok], v[ok], radius)
        costs[ok, k] = np.abs(ref_patches[ok] - cand).mean(axis=1)
    return costs


def match_stereo_epipolar(
    left_image: np.ndarray,
    right_image: np.ndarray,
    uv_left: np.ndarray,
    config: StereoConfig,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> StereoMatches:
    """Match left features into the right image along their epipolar rows.

    The search is a plain SAD block match over the integer disparity range,
    refined to sub-pixel accuracy by fitting a parabola through the cost minimum
    and its two neighbours.  Four filters are applied, in this order:

    1. **Uniqueness** -- the best cost must beat the best cost outside a small
       neighbourhood of the winner by a margin.  Repeated structure (railings,
       kerbs, lane markings) fails this and is dropped.
    2. **Absolute cost** -- a winner that is still a poor match is not a match.
    3. **Left-right consistency** -- re-search from the right image back to the
       left; the two disparities must agree within ``lr_max_diff``.  This is what
       catches occlusions, where a left pixel has no true partner at all.
    4. **Depth uncertainty** -- reject anything below the disparity implied by
       :func:`min_disparity_for`, and anything beyond ``max_depth``.

    Returns
    -------
    :class:`StereoMatches` containing only the survivors.
    """
    uv_left = np.asarray(uv_left, dtype=float).reshape(-1, 2)
    n = uv_left.shape[0]
    empty = StereoMatches(
        np.zeros(0, int), np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3)),
        np.zeros(0), np.zeros(0),
    )
    if n == 0:
        return empty

    radius = config.patch_radius
    height, width = left_image.shape[:2]
    u = np.round(uv_left[:, 0]).astype(int)
    v = np.round(uv_left[:, 1]).astype(int)

    d_min = min_disparity_for(
        fx, baseline, config.disparity_sigma,
        config.max_relative_depth_sigma, config.max_depth,
    )
    d_lo = max(int(np.floor(max(config.min_disparity, d_min))), 1)
    d_hi = int(config.max_disparity)
    if d_hi <= d_lo:
        return empty
    disparities = np.arange(d_lo, d_hi + 1)

    in_bounds = (
        (u >= radius + d_lo)
        & (u < width - radius)
        & (v >= radius)
        & (v < height - radius)
    )
    idx = np.flatnonzero(in_bounds)
    if idx.size == 0:
        return empty

    ul, vl = u[idx], v[idx]
    left_patches = _gather_patches(left_image, ul, vl, radius)
    costs = _sad_search(left_patches, right_image, ul, vl, radius, disparities, -1)

    best_k = np.argmin(costs, axis=1)
    rows = np.arange(idx.size)
    best_cost = costs[rows, best_k]
    finite = np.isfinite(best_cost)

    # Uniqueness: mask a +/-2 disparity window around the winner and look at the
    # next-best cost outside it.
    masked = costs.copy()
    for shift in (-2, -1, 0, 1, 2):
        k = np.clip(best_k + shift, 0, disparities.size - 1)
        masked[rows, k] = np.inf
    runner_up = masked.min(axis=1)
    unique = ~np.isfinite(runner_up) | (best_cost < config.uniqueness_ratio * runner_up)

    keep = finite & unique & (best_cost <= config.max_mean_abs_diff)

    # Sub-pixel refinement: parabola through (k-1, k, k+1).
    d_int = disparities[best_k].astype(float)
    km = np.clip(best_k - 1, 0, disparities.size - 1)
    kp = np.clip(best_k + 1, 0, disparities.size - 1)
    c0, c1, c2 = costs[rows, km], best_cost, costs[rows, kp]
    # A neighbour can be +inf when the winner sits at the edge of the search
    # range; replace those with the winner's own cost so the parabola is flat
    # there and the refinement contributes nothing rather than a NaN.
    usable = np.isfinite(c0) & np.isfinite(c2)
    c0 = np.where(usable, c0, c1)
    c2 = np.where(usable, c2, c1)
    denom = c0 - 2.0 * c1 + c2
    safe = np.abs(denom) > 1e-6
    delta = np.where(safe, 0.5 * (c0 - c2) / np.where(safe, denom, 1.0), 0.0)
    disparity = d_int + np.clip(delta, -1.0, 1.0)

    # Left-right consistency: search back from the right image.
    ur = np.clip(ul - disparities[best_k], radius, width - radius - 1)
    right_patches = _gather_patches(right_image, ur, vl, radius)
    back_costs = _sad_search(right_patches, left_image, ur, vl, radius, disparities, +1)
    back_k = np.argmin(back_costs, axis=1)
    back_d = disparities[back_k].astype(float)
    lr_ok = np.isfinite(back_costs[rows, back_k]) & (
        np.abs(back_d - d_int) <= config.lr_max_diff
    )
    keep &= lr_ok

    keep &= disparity >= d_min
    depth = depth_from_disparity(disparity, fx, baseline)
    keep &= (depth > 0.0) & (depth <= config.max_depth)

    sel = np.flatnonzero(keep)
    if sel.size == 0:
        return empty

    final_idx = idx[sel]
    final_disp = disparity[sel]
    points = disparity_to_points(uv_left[final_idx], final_disp, fx, fy, cx, cy, baseline)
    return StereoMatches(
        index=final_idx,
        uv_left=uv_left[final_idx],
        disparity=final_disp,
        points_cam=points,
        depth=depth[sel],
        depth_sigma=depth_sigma(final_disp, fx, baseline, config.disparity_sigma),
    )


def triangulate_features(
    left_image: np.ndarray,
    right_image: np.ndarray,
    features,
    calibration,
    config: StereoConfig | None = None,
) -> StereoMatches:
    """Convenience wrapper binding a :class:`StereoConfig` to a calibration."""
    config = config or StereoConfig()
    return match_stereo_epipolar(
        left_image,
        right_image,
        features.points,
        config,
        fx=calibration.fx,
        fy=calibration.fy,
        cx=calibration.cx,
        cy=calibration.cy,
        baseline=calibration.baseline,
    )
