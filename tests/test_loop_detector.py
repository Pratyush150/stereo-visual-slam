"""Bag-of-words vocabulary and the loop-detection gates."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.frontend.odometry import OdometryConfig
from svslam.loop.detector import LoopCandidate, LoopConfig, LoopDetector, build_vocabulary_from
from svslam.loop.vocabulary import BagOfWords, POPCOUNT8, hamming_distances, train_vocabulary
from svslam.reprojection import project, transform_points
from svslam.se3 import se3_exp, se3_inverse

from conftest import requires_cv2


def _clustered_descriptors(rng, n_clusters=8, per_cluster=60, flip_rate=0.02):
    """Descriptors drawn from a few well-separated binary prototypes."""
    prototypes = rng.integers(0, 256, (n_clusters, 32), dtype=np.uint8)
    blocks = []
    for prototype in prototypes:
        noise = (rng.random((per_cluster, 32)) < flip_rate).astype(np.uint8)
        noise *= rng.integers(0, 256, (per_cluster, 32), dtype=np.uint8)
        blocks.append(prototype ^ noise)
    return prototypes, np.vstack(blocks)


def test_popcount_table_is_correct():
    assert POPCOUNT8[0] == 0
    assert POPCOUNT8[255] == 8
    assert POPCOUNT8[0b10110000] == 3
    assert POPCOUNT8.sum() == 1024


def test_hamming_distances_match_a_bit_count(rng):
    a = rng.integers(0, 256, (7, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (5, 32), dtype=np.uint8)
    distances = hamming_distances(a, b)
    assert distances.shape == (7, 5)
    reference = np.array([
        [int(np.unpackbits(x ^ y).sum()) for y in b] for x in a
    ])
    assert np.array_equal(distances, reference)


def test_vocabulary_separates_known_clusters(rng):
    _, descriptors = _clustered_descriptors(rng)
    vocabulary = train_vocabulary(descriptors, n_words=8, seed=1)
    words = vocabulary.words(descriptors)
    counts = np.bincount(words, minlength=8)
    assert (counts > 0).sum() == 8
    assert counts.max() - counts.min() <= 10


def test_similarity_is_one_for_identical_distributions(rng):
    _, descriptors = _clustered_descriptors(rng)
    vocabulary = train_vocabulary(descriptors, n_words=8, seed=1)
    v = vocabulary.vector(descriptors)
    assert np.isclose(BagOfWords.similarity(v, v), 1.0)
    assert np.isclose(v.sum(), 1.0)


def test_similarity_is_zero_for_disjoint_word_sets(rng):
    _, descriptors = _clustered_descriptors(rng)
    vocabulary = train_vocabulary(descriptors, n_words=8, seed=1)
    a = vocabulary.vector(descriptors[:60])
    b = vocabulary.vector(descriptors[120:180])
    assert BagOfWords.similarity(a, b) < 0.05


def test_idf_suppresses_words_that_appear_everywhere(rng):
    """A word in every document carries no information and gets weight zero."""
    _, descriptors = _clustered_descriptors(rng, n_clusters=4, per_cluster=40)
    documents = [descriptors] * 6
    vocabulary = build_vocabulary_from(documents, n_words=4)
    assert np.allclose(vocabulary.idf, 0.0)


def test_empty_descriptor_set_gives_a_zero_vector(rng):
    _, descriptors = _clustered_descriptors(rng)
    vocabulary = train_vocabulary(descriptors, n_words=8, seed=1)
    assert np.allclose(vocabulary.vector(np.zeros((0, 32), np.uint8)), 0.0)
    assert vocabulary.words(None).size == 0


def test_build_vocabulary_rejects_empty_input():
    with pytest.raises(ValueError):
        build_vocabulary_from([])


def _detector(rng, n_documents=60):
    prototypes, descriptors = _clustered_descriptors(rng, n_clusters=10, per_cluster=80)
    documents = [
        descriptors[rng.choice(descriptors.shape[0], 60, replace=False)]
        for _ in range(n_documents)
    ]
    vocabulary = build_vocabulary_from(documents, n_words=10)
    return LoopDetector(vocabulary, LoopConfig()), descriptors


def test_temporal_exclusion_blocks_recent_keyframes(rng):
    detector, descriptors = _detector(rng)
    config = detector.config
    for i in range(config.temporal_exclusion):
        detector.add(i, descriptors[:60])
    assert detector.query(config.temporal_exclusion, descriptors[:60]) == []


def test_a_repeated_appearance_eventually_produces_a_candidate(rng):
    detector, descriptors = _detector(rng)
    signature = descriptors[:80]
    other = descriptors[400:480]
    detector.add(0, signature)
    for i in range(1, 60):
        detector.add(i, other)
    # Two consecutive queries, because the consistency gate needs support.
    detector.query(60, signature)
    candidates = detector.query(61, signature)
    assert any(c.candidate_id == 0 for c in candidates)
    assert detector.stats.queries >= 2


def test_single_frame_appearance_spike_is_rejected_by_consistency(rng):
    detector, descriptors = _detector(rng)
    signature = descriptors[:80]
    other = descriptors[400:480]
    detector.add(0, signature)
    for i in range(1, 60):
        detector.add(i, other)
    first = detector.query(60, signature)
    assert first == []
    assert detector.stats.rejected_consistency > 0


@requires_cv2
def test_geometric_verification_rejects_an_appearance_only_match(rng, pinhole):
    """Two places can look alike; they will not share a rigid transform."""
    detector, descriptors = _detector(rng)
    K = np.array([[pinhole["fx"], 0.0, pinhole["cx"]],
                  [0.0, pinhole["fy"], pinhole["cy"]],
                  [0.0, 0.0, 1.0]])
    n = 120
    candidate_points = np.column_stack([
        rng.uniform(-8, 8, n), rng.uniform(-2, 2, n), rng.uniform(8, 30, n)
    ])
    query_pixels = np.column_stack([
        rng.uniform(0, 1226, n), rng.uniform(0, 370, n)
    ])
    shared = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    closure = detector.verify(
        LoopCandidate(60, 0, 0.5), shared, query_pixels, shared,
        candidate_points, np.ones(n, bool), K, OdometryConfig(),
    )
    assert closure is None
    assert detector.stats.rejected_geometry + detector.stats.rejected_few_matches > 0


@requires_cv2
def test_geometric_verification_accepts_a_consistent_transform(rng, pinhole):
    detector, _ = _detector(rng)
    K = np.array([[pinhole["fx"], 0.0, pinhole["cx"]],
                  [0.0, pinhole["fy"], pinhole["cy"]],
                  [0.0, 0.0, 1.0]])
    n = 200
    candidate_points = np.column_stack([
        rng.uniform(-10, 10, n), rng.uniform(-2.5, 2.5, n), rng.uniform(8, 35, n)
    ])
    motion = se3_exp(np.array([0.4, 0.0, -1.5, 0.0, 0.02, 0.0]))
    query_pixels = project(transform_points(motion, candidate_points),
                           pinhole["fx"], pinhole["fy"], pinhole["cx"], pinhole["cy"])
    inside = (
        (query_pixels[:, 0] > 0) & (query_pixels[:, 0] < 1226)
        & (query_pixels[:, 1] > 0) & (query_pixels[:, 1] < 370)
    )
    descriptors = rng.integers(0, 256, (n, 32), dtype=np.uint8)

    closure = detector.verify(
        LoopCandidate(60, 0, 0.6),
        descriptors[inside], query_pixels[inside],
        descriptors[inside], candidate_points[inside],
        np.ones(int(inside.sum()), bool), K, OdometryConfig(),
    )
    assert closure is not None
    assert closure.n_inliers >= detector.config.min_inliers
    # The edge measurement is T_candidate^-1 T_query, i.e. the inverse of the
    # PnP solution.
    assert np.abs(closure.relative_pose - se3_inverse(motion)).max() < 1e-2
    assert detector.stats.accepted == 1


def test_stats_serialise():
    detector = LoopDetector(BagOfWords(np.zeros((2, 32), np.uint8), np.ones(2)))
    stats = detector.stats.as_dict()
    assert set(stats) == {
        "queries", "appearance_candidates", "rejected_consistency",
        "rejected_few_matches", "rejected_geometry", "accepted",
    }
