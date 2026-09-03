"""Map structures: keyframes, landmarks, observations and covisibility.

The map is deliberately small and explicit -- dictionaries and numpy arrays
rather than a database.  A KITTI drive produces a few hundred keyframes and a
few tens of thousands of landmarks, which fits in memory comfortably; the value
here is in the bookkeeping being obvious enough to debug.

Two pieces of bookkeeping matter more than they first appear:

**Covisibility.** Two keyframes are covisible if they observe enough landmarks
in common.  This graph is what decides which keyframes go into a bundle
adjustment window, and it is also the structure a loop closure edits.

**Culling.** Landmarks triangulated from a single stereo pair and never seen
again are noise, and they are the majority.  Keeping them inflates the BA
problem and biases it, because a landmark seen once is perfectly explained by
any pose.  A landmark has to earn its place by being observed from several
keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .se3 import se3_inverse

__all__ = ["Keyframe", "Landmark", "SlamMap"]


@dataclass
class Keyframe:
    """One keyframe: its pose, its features, and what it observed."""

    id: int
    frame_index: int
    T_cw: np.ndarray
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 32), np.uint8))
    #: Feature index -> landmark id.
    observations: dict[int, int] = field(default_factory=dict)
    #: Feature index -> right-image u coordinate, when stereo matched.
    stereo_u_right: dict[int, float] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def T_wc(self) -> np.ndarray:
        """Camera pose in the world frame."""
        return se3_inverse(self.T_cw)

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return self.T_wc[:3, 3]


@dataclass
class Landmark:
    """A 3D point plus the keyframes that saw it."""

    id: int
    position: np.ndarray
    descriptor: np.ndarray = field(default_factory=lambda: np.zeros(32, np.uint8))
    #: keyframe id -> feature index in that keyframe.
    observations: dict[int, int] = field(default_factory=dict)
    n_visible: int = 0
    n_found: int = 0
    bad: bool = False

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def found_ratio(self) -> float:
        """How often the landmark was matched when it should have been visible.

        A landmark that keeps being projected into view but never matched is
        almost certainly a bad triangulation, and this ratio is what catches it.
        """
        return self.n_found / max(self.n_visible, 1)


class SlamMap:
    """Container for keyframes and landmarks with covisibility bookkeeping."""

    def __init__(self) -> None:
        self.keyframes: dict[int, Keyframe] = {}
        self.landmarks: dict[int, Landmark] = {}
        self._next_keyframe_id = 0
        self._next_landmark_id = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_keyframe(
        self,
        frame_index: int,
        T_cw: np.ndarray,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        timestamp: float = 0.0,
    ) -> Keyframe:
        """Insert a new keyframe and return it."""
        kf = Keyframe(
            id=self._next_keyframe_id,
            frame_index=int(frame_index),
            T_cw=np.array(T_cw, dtype=float),
            keypoints=np.asarray(keypoints, dtype=float).reshape(-1, 2),
            descriptors=np.asarray(descriptors),
            timestamp=float(timestamp),
        )
        self.keyframes[kf.id] = kf
        self._next_keyframe_id += 1
        return kf

    def add_landmark(self, position: np.ndarray, descriptor: np.ndarray) -> Landmark:
        """Insert a new landmark and return it."""
        lm = Landmark(
            id=self._next_landmark_id,
            position=np.array(position, dtype=float).reshape(3),
            descriptor=np.asarray(descriptor),
        )
        self.landmarks[lm.id] = lm
        self._next_landmark_id += 1
        return lm

    def add_observation(
        self, keyframe_id: int, feature_index: int, landmark_id: int,
        u_right: float | None = None,
    ) -> None:
        """Link a keyframe feature to a landmark."""
        kf = self.keyframes[keyframe_id]
        lm = self.landmarks[landmark_id]
        kf.observations[int(feature_index)] = int(landmark_id)
        lm.observations[int(keyframe_id)] = int(feature_index)
        lm.n_found += 1
        lm.n_visible = max(lm.n_visible, lm.n_found)
        if u_right is not None:
            kf.stereo_u_right[int(feature_index)] = float(u_right)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def n_keyframes(self) -> int:
        return len(self.keyframes)

    @property
    def n_landmarks(self) -> int:
        return len(self.landmarks)

    def keyframe_ids(self) -> list[int]:
        return sorted(self.keyframes)

    def trajectory(self) -> np.ndarray:
        """``(K, 4, 4)`` keyframe poses ``T_wc``, in keyframe order."""
        ids = self.keyframe_ids()
        if not ids:
            return np.zeros((0, 4, 4))
        return np.array([self.keyframes[i].T_wc for i in ids])

    def landmark_positions(self) -> np.ndarray:
        """``(L, 3)`` positions of every landmark still alive."""
        ids = sorted(self.landmarks)
        if not ids:
            return np.zeros((0, 3))
        return np.array([self.landmarks[i].position for i in ids])

    def covisibility(self, min_shared: int = 15) -> dict[int, dict[int, int]]:
        """Covisibility graph as ``{keyframe: {other: shared landmark count}}``.

        Built by inverting the landmark observation lists, which is ``O(sum of
        observations)`` rather than the ``O(K^2)`` of comparing every pair of
        keyframes.
        """
        counts: dict[int, dict[int, int]] = {kf_id: {} for kf_id in self.keyframes}
        for lm in self.landmarks.values():
            if lm.bad:
                continue
            observers = sorted(lm.observations)
            for a_i, a in enumerate(observers):
                for b in observers[a_i + 1:]:
                    counts[a][b] = counts[a].get(b, 0) + 1
                    counts[b][a] = counts[b].get(a, 0) + 1
        return {
            kf: {other: c for other, c in neighbours.items() if c >= min_shared}
            for kf, neighbours in counts.items()
        }

    def local_window(self, keyframe_id: int, size: int, min_shared: int = 15) -> list[int]:
        """The ``size`` keyframes most covisible with ``keyframe_id``, plus itself.

        Falls back to temporal neighbours when the covisibility graph is thin,
        which happens at the start of a sequence and after a tracking failure.
        """
        if keyframe_id not in self.keyframes:
            return []
        neighbours = self.covisibility(min_shared).get(keyframe_id, {})
        ranked = sorted(neighbours, key=lambda k: -neighbours[k])[: max(size - 1, 0)]
        window = {keyframe_id, *ranked}
        if len(window) < size:
            ids = self.keyframe_ids()
            pos = ids.index(keyframe_id)
            for offset in range(1, size):
                for candidate in (pos - offset, pos + offset):
                    if 0 <= candidate < len(ids) and len(window) < size:
                        window.add(ids[candidate])
        return sorted(window)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cull_landmarks(
        self, min_observations: int = 3, min_found_ratio: float = 0.25
    ) -> int:
        """Remove weakly supported landmarks; return how many were removed.

        Landmarks observed by fewer than ``min_observations`` keyframes, or whose
        found ratio is too low, are deleted along with their observation links.
        """
        doomed = [
            lm_id
            for lm_id, lm in self.landmarks.items()
            if lm.n_observations < min_observations or lm.found_ratio < min_found_ratio
        ]
        for lm_id in doomed:
            lm = self.landmarks.pop(lm_id)
            for kf_id, feat in lm.observations.items():
                kf = self.keyframes.get(kf_id)
                if kf is not None:
                    kf.observations.pop(feat, None)
        return len(doomed)

    def apply_pose_correction(self, corrected: dict[int, np.ndarray]) -> None:
        """Rewrite keyframe poses and drag their landmarks along rigidly.

        After a pose-graph optimisation the keyframes move but the landmarks do
        not, and leaving them behind destroys the map.  Each landmark is
        transported by the correction of the keyframe that first observed it,
        which keeps it in the same place *relative to that keyframe* -- the
        relationship that was actually measured.
        """
        deltas: dict[int, np.ndarray] = {}
        for kf_id, T_wc_new in corrected.items():
            kf = self.keyframes.get(kf_id)
            if kf is None:
                continue
            deltas[kf_id] = np.asarray(T_wc_new, dtype=float) @ kf.T_cw
            kf.T_cw = se3_inverse(np.asarray(T_wc_new, dtype=float))

        for lm in self.landmarks.values():
            if not lm.observations:
                continue
            anchor = min(lm.observations)
            delta = deltas.get(anchor)
            if delta is None:
                continue
            lm.position = delta[:3, :3] @ lm.position + delta[:3, 3]
