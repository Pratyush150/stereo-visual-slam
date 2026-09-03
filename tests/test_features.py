"""Feature bucketing, spread metrics, matching and KLT tracking."""

from __future__ import annotations

import numpy as np

from svslam.frontend.features import (
    FeatureConfig,
    FeatureSet,
    OrbDetector,
    bucket_indices,
    bucket_keypoints,
    match_descriptors,
    spatial_spread,
    track_klt,
)

from conftest import requires_cv2

IMAGE_SHAPE = (370, 1226)


def _clumped_and_spread(rng, n_clump=900, n_spread=300):
    """A detector's worst case: most responses piled onto one textured region."""
    clump = np.column_stack([
        rng.normal(200.0, 20.0, n_clump), rng.normal(150.0, 15.0, n_clump)
    ])
    spread = np.column_stack([
        rng.uniform(0.0, IMAGE_SHAPE[1], n_spread),
        rng.uniform(0.0, IMAGE_SHAPE[0], n_spread),
    ])
    points = np.vstack([clump, spread])
    # The clumped features are also the strongest, which is why a
    # response-ranked budget keeps exactly the wrong ones.
    responses = np.concatenate([
        rng.uniform(0.5, 1.0, n_clump), rng.uniform(0.0, 0.5, n_spread)
    ])
    return points, responses


def test_bucketing_spreads_features_measurably_better(rng):
    points, responses = _clumped_and_spread(rng)
    before = spatial_spread(points, IMAGE_SHAPE)

    keep = bucket_indices(points, responses, IMAGE_SHAPE, 5, 12, 10)
    after = spatial_spread(points[keep], IMAGE_SHAPE)

    assert after["normalised_entropy"] > before["normalised_entropy"] + 0.2
    assert after["max_cell_fraction"] < 0.5 * before["max_cell_fraction"]
    assert after["occupancy"] >= before["occupancy"]


def test_bucketing_beats_a_plain_response_ranked_budget(rng):
    """Taking the strongest N features is what bucketing is competing against."""
    points, responses = _clumped_and_spread(rng)
    budget = 300
    ranked = np.argsort(-responses)[:budget]
    bucketed = bucket_indices(points, responses, IMAGE_SHAPE, 5, 12, 10, budget)

    ranked_spread = spatial_spread(points[ranked], IMAGE_SHAPE)
    bucketed_spread = spatial_spread(points[bucketed], IMAGE_SHAPE)
    assert bucketed_spread["normalised_entropy"] > ranked_spread["normalised_entropy"]
    assert bucketed_spread["occupancy"] > ranked_spread["occupancy"]


def test_bucket_cap_is_respected(rng):
    points, responses = _clumped_and_spread(rng)
    rows, cols, cap = 4, 8, 5
    keep = bucket_indices(points, responses, IMAGE_SHAPE, rows, cols, cap)
    kept = points[keep]
    row = np.clip((kept[:, 1] / (IMAGE_SHAPE[0] / rows)).astype(int), 0, rows - 1)
    col = np.clip((kept[:, 0] / (IMAGE_SHAPE[1] / cols)).astype(int), 0, cols - 1)
    counts = np.bincount(row * cols + col, minlength=rows * cols)
    assert counts.max() <= cap
    assert keep.size <= rows * cols * cap


def test_bucketing_keeps_the_strongest_within_each_cell():
    points = np.array([[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]])
    responses = np.array([0.1, 0.9, 0.5])
    keep = bucket_indices(points, responses, (100, 100), 1, 1, 2)
    assert set(keep.tolist()) == {1, 2}


def test_global_budget_is_applied_after_bucketing(rng):
    points, responses = _clumped_and_spread(rng)
    keep = bucket_indices(points, responses, IMAGE_SHAPE, 5, 12, 30, max_total=50)
    assert keep.size == 50


def test_spread_of_an_empty_set_is_zero():
    result = spatial_spread(np.zeros((0, 2)), IMAGE_SHAPE)
    assert result == {"occupancy": 0.0, "normalised_entropy": 0.0, "max_cell_fraction": 0.0}


def test_spread_of_a_single_cell_is_the_worst_possible():
    points = np.full((50, 2), 5.0)
    result = spatial_spread(points, IMAGE_SHAPE, 5, 12)
    assert result["normalised_entropy"] == 0.0
    assert result["max_cell_fraction"] == 1.0
    assert np.isclose(result["occupancy"], 1.0 / 60.0)


def test_bucket_keypoints_wraps_the_index_helper(rng):
    points, responses = _clumped_and_spread(rng)
    features = FeatureSet(points, np.zeros((points.shape[0], 32), np.uint8), responses)
    config = FeatureConfig(max_features=200, grid_rows=5, grid_cols=12, max_per_cell=8)
    reduced = bucket_keypoints(features, IMAGE_SHAPE, config)
    assert len(reduced) <= 200
    assert reduced.descriptors.shape[0] == len(reduced)


@requires_cv2
def test_orb_detector_with_bucketing_spreads_better_than_without(rng):
    """The same claim, on a real detector over a synthetic textured scene."""
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    image[:] = rng.integers(60, 90, IMAGE_SHAPE, dtype=np.uint8)
    # One very high-contrast patch that ORB will pile onto.
    image[120:220, 150:400] = rng.integers(0, 255, (100, 250), dtype=np.uint8)
    # Weaker texture everywhere else.
    image += (rng.integers(0, 30, IMAGE_SHAPE)).astype(np.uint8)

    plain = OrbDetector(FeatureConfig(max_features=400, use_bucketing=False))
    bucketed = OrbDetector(FeatureConfig(max_features=400, use_bucketing=True,
                                         grid_rows=5, grid_cols=12, max_per_cell=8))
    a = spatial_spread(plain.detect(image).points, IMAGE_SHAPE)
    b = spatial_spread(bucketed.detect(image).points, IMAGE_SHAPE)
    assert b["max_cell_fraction"] <= a["max_cell_fraction"]
    assert b["normalised_entropy"] >= a["normalised_entropy"]


@requires_cv2
def test_descriptor_matching_finds_the_identity_permutation(rng):
    descriptors = rng.integers(0, 256, (60, 32), dtype=np.uint8)
    permutation = rng.permutation(60)
    matches = match_descriptors(descriptors, descriptors[permutation], ratio=0.9)
    assert matches.shape[0] > 50
    lookup = {int(a): int(b) for a, b in matches}
    for a, b in lookup.items():
        assert permutation[b] == a


@requires_cv2
def test_descriptor_matching_with_empty_input():
    assert match_descriptors(None, None).shape == (0, 2)
    assert match_descriptors(np.zeros((0, 32), np.uint8),
                             np.zeros((5, 32), np.uint8)).shape == (0, 2)


@requires_cv2
def test_klt_tracks_a_known_translation(rng):
    image = rng.integers(0, 255, (200, 320), dtype=np.uint8)
    image = np.repeat(np.repeat(image[::4, ::4], 4, axis=0), 4, axis=1)[:200, :320]
    shift = 3
    shifted = np.roll(image, shift, axis=1)
    points = np.column_stack([
        rng.uniform(60, 260, 40), rng.uniform(40, 160, 40)
    ]).astype(np.float32)

    tracked, valid = track_klt(image, shifted, points)
    assert valid.sum() > 20
    delta = tracked[valid] - points[valid]
    assert np.abs(np.median(delta[:, 0]) - shift) < 0.6
    assert np.abs(np.median(delta[:, 1])) < 0.6


@requires_cv2
def test_klt_forward_backward_check_rejects_nonsense(rng):
    image = rng.integers(0, 255, (200, 320), dtype=np.uint8)
    unrelated = rng.integers(0, 255, (200, 320), dtype=np.uint8)
    points = np.column_stack([
        rng.uniform(60, 260, 40), rng.uniform(40, 160, 40)
    ]).astype(np.float32)
    _, valid = track_klt(image, unrelated, points, fb_threshold=0.5)
    assert valid.sum() < 0.5 * points.shape[0]


@requires_cv2
def test_klt_with_no_points():
    image = np.zeros((50, 50), dtype=np.uint8)
    points, valid = track_klt(image, image, np.zeros((0, 2)))
    assert points.shape == (0, 2)
    assert valid.size == 0
