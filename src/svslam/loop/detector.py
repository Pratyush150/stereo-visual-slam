"""Appearance-based loop detection with geometric verification.

A loop closure is the single highest-value and single most dangerous event in a
SLAM system.  Correct, it removes the accumulated drift of an entire lap in one
step.  Wrong, it welds two unrelated places together and the map is finished --
and unlike a bad odometry step there is no averaging that will save you, because
the constraint is strong and confidently wrong.

So detection is deliberately conservative and runs three gates in series:

1. **Appearance.**  Bag-of-words similarity against every keyframe outside a
   temporal exclusion window.  The similarity threshold is set *relative to the
   score against recent neighbours*, not as an absolute number: a keyframe on a
   featureless road matches everything weakly, one in a rich scene matches
   everything strongly, and an absolute threshold is wrong in both cases.
2. **Temporal consistency.**  A candidate must be supported by consecutive
   queries.  A single-frame appearance spike is nearly always a coincidence;
   real revisits persist over several keyframes.
3. **Geometry.**  RANSAC PnP between the candidate's landmarks and the query's
   features.  The inlier count is the decisive test -- two places that look alike
   almost never produce a consistent rigid transform over dozens of features.

The rejection counts from each gate are recorded, because a loop detector that
never rejects anything is not being conservative, it is being lucky.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.features import match_descriptors
from ..frontend.odometry import OdometryConfig, estimate_pose_pnp
from ..se3 import se3_inverse
from .vocabulary import BagOfWords, train_vocabulary

__all__ = ["LoopConfig", "LoopCandidate", "LoopClosure", "LoopDetector", "build_vocabulary_from"]


@dataclass(frozen=True)
class LoopConfig:
    """Gates and thresholds for loop detection."""

    #: Keyframes closer than this in index are never considered a loop.
    temporal_exclusion: int = 30
    #: Candidate score must exceed this fraction of the best recent-neighbour
    #: score.  Deliberately permissive: measured on KITTI drive 0027 the
    #: bag-of-words score separates a true revisit from an unrelated stretch of
    #: the same residential street by only a few percent, so appearance is used
    #: to *rank* candidates and geometry is what actually decides.
    relative_score: float = 0.5
    #: Absolute floor on the bag-of-words score, as a backstop.
    min_score: float = 0.02
    #: How many consecutive queries must support a candidate region.
    consistency_required: int = 2
    #: Minimum descriptor matches before geometry is even attempted.
    min_matches: int = 25
    #: Minimum RANSAC PnP inliers to accept a loop.
    min_inliers: int = 30
    #: Minimum inlier ratio to accept a loop.
    min_inlier_ratio: float = 0.35
    #: Candidates evaluated geometrically per query, best-scoring first.
    #: Measured top-3 retrieval recall on drive 0027 was 0.71 and top-5 was 0.76,
    #: so verifying five costs little and finds noticeably more real loops.
    max_candidates: int = 5
    #: Vocabulary size for :func:`build_vocabulary_from`.  A larger vocabulary
    #: makes each keyframe's word set sparser, which is what gives the inverse
    #: document frequency anything to work with: at 400 words nearly every word
    #: appears in nearly every keyframe and the score stops discriminating.
    vocabulary_size: int = 1000


@dataclass
class LoopCandidate:
    """An appearance match that has not yet been geometrically verified."""

    query_id: int
    candidate_id: int
    score: float


@dataclass
class LoopClosure:
    """An accepted loop closure and the constraint it contributes."""

    query_id: int
    candidate_id: int
    score: float
    n_matches: int
    n_inliers: int
    #: ``T_i^-1 T_j`` for the pose-graph edge (i = candidate, j = query).
    relative_pose: np.ndarray


@dataclass
class LoopStats:
    """Counters for every gate, so the rejection behaviour can be reported."""

    queries: int = 0
    appearance_candidates: int = 0
    rejected_consistency: int = 0
    rejected_few_matches: int = 0
    rejected_geometry: int = 0
    accepted: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def build_vocabulary_from(
    descriptor_sets: list[np.ndarray],
    n_words: int = 400,
    max_descriptors: int = 40000,
    seed: int = 0,
) -> BagOfWords:
    """Train a vocabulary from a held-out set of keyframe descriptors.

    Training on the same frames the detector will later query inflates its
    apparent performance, so callers should pass a held-out slice of the
    sequence.  The descriptor pool is subsampled to ``max_descriptors`` because
    k-means cost is linear in it and a few tens of thousands is already plenty
    for a few hundred words.
    """
    pool = [d for d in descriptor_sets if d is not None and len(d)]
    if not pool:
        raise ValueError("no descriptors supplied for vocabulary training")
    stacked = np.vstack(pool)
    rng = np.random.default_rng(seed)
    if stacked.shape[0] > max_descriptors:
        stacked = stacked[rng.choice(stacked.shape[0], max_descriptors, replace=False)]
    vocab = train_vocabulary(stacked, n_words=n_words, seed=seed, n_documents=len(pool))
    # The first pass weights words by cluster size.  Now that the words exist,
    # recompute the inverse document frequency over whole keyframes, which is
    # what the score actually needs: a word appearing once in every keyframe is
    # useless even if its cluster is small.
    doc_words = [vocab.words(d) for d in pool]
    doc_freq = np.zeros(vocab.size, dtype=float)
    for words in doc_words:
        if words.size:
            doc_freq[np.unique(words)] += 1.0
    idf = np.clip(np.log(max(len(pool), 1) / np.maximum(doc_freq, 1e-9)), 0.0, None)
    return BagOfWords(centres=vocab.centres, idf=idf)


class LoopDetector:
    """Maintains a keyframe appearance database and proposes verified loops."""

    def __init__(self, vocabulary: BagOfWords, config: LoopConfig | None = None) -> None:
        self.vocabulary = vocabulary
        self.config = config or LoopConfig()
        self._ids: list[int] = []
        self._vectors: list[np.ndarray] = []
        self._index: dict[int, int] = {}
        self._previous_candidates: set[int] = set()
        self._consistency: dict[int, int] = {}
        self.stats = LoopStats()

    def add(self, keyframe_id: int, descriptors: np.ndarray) -> None:
        """Insert a keyframe's appearance into the database."""
        self._index[int(keyframe_id)] = len(self._ids)
        self._ids.append(int(keyframe_id))
        self._vectors.append(self.vocabulary.vector(descriptors))

    def query(self, keyframe_id: int, descriptors: np.ndarray) -> list[LoopCandidate]:
        """Return appearance candidates that also pass the consistency gate."""
        self.stats.queries += 1
        cfg = self.config
        if len(self._ids) <= cfg.temporal_exclusion:
            return []

        vector = self.vocabulary.vector(descriptors)
        scores = np.array([BagOfWords.similarity(vector, v) for v in self._vectors])
        ids = np.array(self._ids)

        # The query keyframe is already in the database by this point, and its
        # score against itself is exactly 1.0.  Including it in the reference
        # set makes the relative threshold 1.0 * relative_score, which nothing
        # ever clears -- the detector then silently proposes nothing, forever.
        recent = (ids > keyframe_id - cfg.temporal_exclusion) & (ids != keyframe_id)
        old = (ids <= keyframe_id - cfg.temporal_exclusion) & (ids != keyframe_id)
        if not np.any(old):
            return []

        # Normalise against how well this keyframe matches its own neighbours.
        # That is the score a "boring" match should reach, so anything a loop
        # candidate must beat is expressed relative to it.
        reference = float(np.max(scores[recent])) if np.any(recent) else 1.0
        threshold = max(cfg.relative_score * reference, cfg.min_score)

        order = np.argsort(-scores)
        candidates: list[LoopCandidate] = []
        for k in order:
            if not old[k] or scores[k] < threshold:
                continue
            candidates.append(LoopCandidate(keyframe_id, int(ids[k]), float(scores[k])))
            if len(candidates) >= cfg.max_candidates:
                break
        self.stats.appearance_candidates += len(candidates)

        # Temporal consistency: a candidate region must persist across queries.
        current = {c.candidate_id for c in candidates}
        consistent: list[LoopCandidate] = []
        new_counts: dict[int, int] = {}
        for c in candidates:
            near_previous = any(
                abs(c.candidate_id - prev) <= cfg.temporal_exclusion
                for prev in self._previous_candidates
            )
            count = (self._consistency.get(c.candidate_id, 0) + 1) if near_previous else 1
            new_counts[c.candidate_id] = count
            if count >= cfg.consistency_required:
                consistent.append(c)
            else:
                self.stats.rejected_consistency += 1
        self._previous_candidates = current
        self._consistency = new_counts
        return consistent

    def verify(
        self,
        candidate: LoopCandidate,
        query_descriptors: np.ndarray,
        query_keypoints: np.ndarray,
        candidate_descriptors: np.ndarray,
        candidate_points_cam: np.ndarray,
        candidate_point_valid: np.ndarray,
        K: np.ndarray,
        odometry_config: OdometryConfig | None = None,
    ) -> LoopClosure | None:
        """Geometrically verify one candidate with RANSAC PnP.

        ``candidate_points_cam`` holds 3D points in the *candidate keyframe's*
        camera frame, so the PnP solution is directly the relative transform
        between the two keyframes -- no world frame is involved, which means a
        loop can be verified even when the global map has drifted badly.
        """
        cfg = self.config
        matches = match_descriptors(query_descriptors, candidate_descriptors)
        if matches.shape[0] < cfg.min_matches:
            self.stats.rejected_few_matches += 1
            return None

        valid = np.asarray(candidate_point_valid, dtype=bool)[matches[:, 1]]
        matches = matches[valid]
        if matches.shape[0] < cfg.min_matches:
            self.stats.rejected_few_matches += 1
            return None

        points = np.asarray(candidate_points_cam, dtype=float)[matches[:, 1]]
        pixels = np.asarray(query_keypoints, dtype=float)[matches[:, 0]]

        estimate = estimate_pose_pnp(points, pixels, K, odometry_config)
        n_inliers = int(estimate.inliers.sum())
        ratio = n_inliers / max(matches.shape[0], 1)
        if (
            not estimate.success
            or n_inliers < cfg.min_inliers
            or ratio < cfg.min_inlier_ratio
        ):
            self.stats.rejected_geometry += 1
            return None

        # PnP returns the transform candidate-frame -> query-frame; the
        # pose-graph edge wants T_candidate^-1 T_query, which is its inverse.
        self.stats.accepted += 1
        return LoopClosure(
            query_id=candidate.query_id,
            candidate_id=candidate.candidate_id,
            score=candidate.score,
            n_matches=int(matches.shape[0]),
            n_inliers=n_inliers,
            relative_pose=se3_inverse(estimate.T_cw),
        )
