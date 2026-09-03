"""The stereo SLAM pipeline: everything wired together.

Per-frame flow::

    detect (bucketed ORB)
      -> sparse stereo along epipolar lines -> metric 3D points
      -> match against the reference keyframe -> 3D-2D correspondences
      -> RANSAC PnP + motion-only Gauss-Newton -> pose
      -> keyframe? -> insert, triangulate new landmarks, local BA
                   -> loop query -> geometric verification -> pose graph

Design decisions worth stating, because they are the ones that cost time:

* **Tracking is against the reference keyframe, not the previous frame.**
  Frame-to-frame chaining accumulates error at frame rate.  Matching to a
  keyframe that is several frames back gives a wider baseline and, more
  importantly, means the error does not compound between keyframes.
* **Every frame's pose is stored relative to its reference keyframe.**  When a
  loop closure later moves the keyframes, the non-keyframe poses move with
  them.  Storing absolute poses would leave the reported trajectory unchanged
  by loop closure everywhere except at keyframes, which is both wrong and a
  very easy mistake to not notice.
* **Landmarks live in the world frame of the first keyframe**, which is the
  identity.  That matches the KITTI odometry convention and means no extra
  change of basis before evaluation.
* **A landmark is associated into a new keyframe only if it projects there.**
  Trusting the descriptor match alone puts roughly half the associations in
  wrong, which does not crash anything -- it quietly hands bundle adjustment
  contradictory observations, and bundle adjustment resolves the contradiction
  by moving keyframes metres out of place.  Measured on KITTI drive 0027 the
  difference is 7.35% translation error against 2.00%.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend.ba import BAConfig, BAProblem, bundle_adjust
from .backend.posegraph import (
    PoseGraphConfig,
    PoseGraphEdge,
    optimise_pose_graph,
)
from .frontend.features import (
    FeatureConfig,
    FeatureSet,
    OrbDetector,
    match_descriptors,
    spatial_spread,
)
from .frontend.odometry import (
    KeyframePolicy,
    OdometryConfig,
    estimate_pose_pnp,
    is_plausible_motion,
)
from .frontend.stereo import StereoConfig, match_stereo_epipolar
from .loop.detector import LoopConfig, LoopDetector, build_vocabulary_from
from .map import SlamMap
from .reprojection import project, transform_points
from .se3 import se3_inverse

__all__ = ["PipelineConfig", "PipelineResult", "StereoSlam"]


@dataclass
class PipelineConfig:
    """All tunables for a run, grouped by the stage they belong to."""

    feature: FeatureConfig = field(default_factory=FeatureConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    keyframe: KeyframePolicy = field(default_factory=KeyframePolicy)
    ba: BAConfig = field(default_factory=BAConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    posegraph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    #: Keyframes in a local bundle-adjustment window.
    local_ba_window: int = 7
    #: Run local BA at all.
    enable_ba: bool = True
    #: Run loop detection and pose-graph optimisation at all.
    enable_loop: bool = True
    #: Minimum 3D-2D correspondences before PnP is attempted.
    min_track_matches: int = 20
    #: Covisibility threshold for choosing the BA window.
    covisibility_threshold: int = 15
    #: Cull landmarks every N keyframes.
    cull_interval: int = 5
    #: Odometry-edge information weight in the pose graph.
    odometry_information: float = 100.0
    #: Loop-edge information weight in the pose graph.
    loop_information: float = 100.0


@dataclass
class PipelineResult:
    """Trajectory, map and statistics from one run."""

    poses: np.ndarray
    frame_indices: list[int]
    keyframe_frame_indices: list[int]
    slam_map: SlamMap
    stats: dict[str, float]
    timings: dict[str, float]
    loop_closures: list = field(default_factory=list)
    rejected_loops: dict[str, int] = field(default_factory=dict)
    poses_before_loop: np.ndarray | None = None
    tracked_per_frame: list[int] = field(default_factory=list)
    spread_per_keyframe: list[float] = field(default_factory=list)


class _Timer:
    """Accumulates wall time per named stage."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self._start: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name in self._start:
            self.totals[name] = self.totals.get(name, 0.0) + (
                time.perf_counter() - self._start.pop(name)
            )


class StereoSlam:
    """Stereo visual SLAM over a KITTI sequence."""

    def __init__(self, calibration, config: PipelineConfig | None = None) -> None:
        self.calibration = calibration
        self.config = config or PipelineConfig()
        self.detector = OrbDetector(self.config.feature)
        self.map = SlamMap()
        self.timer = _Timer()

        self._K = calibration.K
        self._reference_id: int | None = None
        self._reference_tracked = 1
        #: How many frame-to-frame estimates the plausibility gate threw out.
        self.n_rejected_motions = 0
        self._reference_stereo: dict[int, np.ndarray] = {}
        self._frames_since_keyframe = 0
        self._reference_tracked = 1
        self._last_T_cw = np.eye(4)
        self._last_velocity = np.eye(4)
        self._edges: list[PoseGraphEdge] = []
        self._loop_detector: LoopDetector | None = None
        #: keyframe id -> (points_cam, valid mask) for loop verification.
        self._keyframe_points: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def train_vocabulary(self, dataset, frame_indices) -> None:
        """Train the loop-closure vocabulary on a held-out set of frames.

        Held out means: not the frames the detector will be scored on.  Training
        the vocabulary on the query frames themselves makes the words fit that
        exact appearance and inflates every similarity score.
        """
        descriptor_sets = []
        for index in frame_indices:
            features = self.detector.detect(dataset.load_left(int(index)))
            if len(features):
                descriptor_sets.append(features.descriptors)
        vocabulary = build_vocabulary_from(
            descriptor_sets, n_words=self.config.loop.vocabulary_size
        )
        self._loop_detector = LoopDetector(vocabulary, self.config.loop)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        dataset,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        progress: int = 0,
    ) -> PipelineResult:
        """Process a slice of a sequence and return the trajectory and map."""
        cfg = self.config
        stop = len(dataset) if stop is None else min(stop, len(dataset))
        frame_indices: list[int] = []
        rel_to_reference: list[tuple[int, np.ndarray]] = []
        tracked_per_frame: list[int] = []
        spread_per_keyframe: list[float] = []
        keyframe_frames: list[int] = []
        loop_closures: list = []
        n_stereo_total = 0
        n_frames = 0
        poses_before_loop: np.ndarray | None = None

        for index in range(start, stop, max(step, 1)):
            self.timer.start("io")
            left, right = dataset.load_stereo(index)
            self.timer.stop("io")

            self.timer.start("features")
            features = self.detector.detect(left)
            self.timer.stop("features")

            self.timer.start("stereo")
            stereo = match_stereo_epipolar(
                left, right, features.points, cfg.stereo,
                fx=self.calibration.fx, fy=self.calibration.fy,
                cx=self.calibration.cx, cy=self.calibration.cy,
                baseline=self.calibration.baseline,
            )
            self.timer.stop("stereo")
            n_stereo_total += len(stereo)
            n_frames += 1

            timestamp = float(dataset.timestamps[index]) if len(dataset.timestamps) > index else 0.0

            if self._reference_id is None:
                self._bootstrap(features, stereo, index, timestamp)
                frame_indices.append(index)
                rel_to_reference.append((self._reference_id, np.eye(4)))
                tracked_per_frame.append(len(stereo))
                keyframe_frames.append(index)
                spread_per_keyframe.append(
                    spatial_spread(features.points, left.shape,
                                   cfg.feature.grid_rows, cfg.feature.grid_cols)["normalised_entropy"]
                )
                continue

            self.timer.start("tracking")
            T_cw, n_inliers = self._track(features)
            self.timer.stop("tracking")

            if self._frames_since_keyframe == 0:
                # The first successful track after a keyframe defines what
                # "well tracked" means for it.  Comparing instead against the
                # keyframe's total landmark count makes every frame a keyframe,
                # because only a fraction of a keyframe's landmarks are ever
                # visible in any one later frame.
                self._reference_tracked = max(n_inliers, 1)
            self._last_velocity = T_cw @ se3_inverse(self._last_T_cw)
            self._last_T_cw = T_cw
            self._frames_since_keyframe += 1
            tracked_per_frame.append(n_inliers)

            reference = self.map.keyframes[self._reference_id]
            relative = T_cw @ reference.T_wc
            frame_indices.append(index)
            rel_to_reference.append((self._reference_id, relative))

            if self.config.keyframe.should_insert(
                self._frames_since_keyframe, n_inliers, self._reference_tracked, relative
            ):
                self.timer.start("keyframe")
                new_id = self._insert_keyframe(features, stereo, index, timestamp, T_cw)
                self.timer.stop("keyframe")
                keyframe_frames.append(index)
                spread_per_keyframe.append(
                    spatial_spread(features.points, left.shape,
                                   cfg.feature.grid_rows, cfg.feature.grid_cols)["normalised_entropy"]
                )
                rel_to_reference[-1] = (new_id, np.eye(4))

                if cfg.enable_ba:
                    self.timer.start("bundle_adjustment")
                    self._local_bundle_adjustment(new_id)
                    self.timer.stop("bundle_adjustment")

                if cfg.enable_loop and self._loop_detector is not None:
                    self.timer.start("loop")
                    closures = self._detect_loops(new_id, features)
                    self.timer.stop("loop")
                    if closures:
                        if poses_before_loop is None:
                            poses_before_loop = self._assemble(frame_indices, rel_to_reference)
                        loop_closures.extend(closures)
                        self.timer.start("pose_graph")
                        self._close_loops(closures)
                        self.timer.stop("pose_graph")

                if new_id % max(cfg.cull_interval, 1) == 0:
                    self.map.cull_landmarks(min_observations=2)

                self._last_T_cw = self.map.keyframes[new_id].T_cw

            if progress and (index - start) % progress == 0:
                print(
                    f"  frame {index:5d}  keyframes {self.map.n_keyframes:4d}"
                    f"  landmarks {self.map.n_landmarks:6d}  tracked {n_inliers:4d}",
                    flush=True,
                )

        poses = self._assemble(frame_indices, rel_to_reference)
        stats = {
            "frames": float(n_frames),
            "keyframes": float(self.map.n_keyframes),
            "landmarks": float(self.map.n_landmarks),
            "mean_stereo_points": n_stereo_total / max(n_frames, 1),
            "mean_tracked_features": float(np.mean(tracked_per_frame)) if tracked_per_frame else 0.0,
            "mean_feature_spread": float(np.mean(spread_per_keyframe)) if spread_per_keyframe else 0.0,
            "loop_closures_accepted": float(len(loop_closures)),
            "rejected_implausible_motions": float(self.n_rejected_motions),
        }
        rejected = (
            self._loop_detector.stats.as_dict() if self._loop_detector is not None else {}
        )
        return PipelineResult(
            poses=poses,
            frame_indices=frame_indices,
            keyframe_frame_indices=keyframe_frames,
            slam_map=self.map,
            stats=stats,
            timings=dict(self.timer.totals),
            loop_closures=loop_closures,
            rejected_loops=rejected,
            poses_before_loop=poses_before_loop,
            tracked_per_frame=tracked_per_frame,
            spread_per_keyframe=spread_per_keyframe,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assemble(self, frame_indices, rel_to_reference) -> np.ndarray:
        """Rebuild every frame pose from its reference keyframe's current pose."""
        poses = np.zeros((len(frame_indices), 4, 4))
        for k, (kf_id, relative) in enumerate(rel_to_reference):
            reference = self.map.keyframes.get(kf_id)
            if reference is None:
                poses[k] = np.eye(4)
                continue
            poses[k] = reference.T_wc @ se3_inverse(relative)
        return poses

    def _bootstrap(self, features: FeatureSet, stereo, index: int, timestamp: float) -> None:
        """Create the first keyframe; its camera frame defines the world."""
        kf = self.map.add_keyframe(index, np.eye(4), features.points, features.descriptors, timestamp)
        self._reference_id = kf.id
        self._store_keyframe_points(kf.id, features, stereo)
        for k, feature_index in enumerate(stereo.index):
            landmark = self.map.add_landmark(
                stereo.points_cam[k], features.descriptors[feature_index]
            )
            u_right = float(stereo.uv_left[k, 0] - stereo.disparity[k])
            self.map.add_observation(kf.id, int(feature_index), landmark.id, u_right)
        if self._loop_detector is not None:
            self._loop_detector.add(kf.id, features.descriptors)
        self._last_T_cw = np.eye(4)
        self._frames_since_keyframe = 0

    def _store_keyframe_points(self, kf_id: int, features: FeatureSet, stereo) -> None:
        points = np.zeros((len(features), 3))
        valid = np.zeros(len(features), dtype=bool)
        points[stereo.index] = stereo.points_cam
        valid[stereo.index] = True
        self._keyframe_points[kf_id] = (points, valid)

    def _track(self, features: FeatureSet) -> tuple[np.ndarray, int]:
        """Match against the reference keyframe and solve for the pose."""
        reference = self.map.keyframes[self._reference_id]
        matches = match_descriptors(
            features.descriptors,
            reference.descriptors,
            self.config.feature.ratio_test,
            self.config.feature.max_hamming,
        )
        points, pixels = [], []
        for query_idx, ref_idx in matches:
            landmark_id = reference.observations.get(int(ref_idx))
            if landmark_id is None:
                continue
            landmark = self.map.landmarks.get(landmark_id)
            if landmark is None:
                continue
            points.append(landmark.position)
            pixels.append(features.points[int(query_idx)])

        # Constant-velocity prediction is the fallback and the PnP seed.
        predicted = self._last_velocity @ self._last_T_cw
        if len(points) < self.config.min_track_matches:
            return predicted, len(points)

        estimate = estimate_pose_pnp(
            np.array(points), np.array(pixels), self._K, self.config.odometry,
            T_cw_guess=predicted,
        )
        if not estimate.success:
            return predicted, int(estimate.inliers.sum())

        # Plausibility gate.  A pose that implies an impossible jump between two
        # consecutive frames is discarded in favour of the constant-velocity
        # prediction, and the low inlier count returned here forces a keyframe,
        # which re-seeds the local map from this frame's own stereo pair.
        relative = self._last_T_cw @ se3_inverse(estimate.T_cw)
        if not is_plausible_motion(
            relative,
            self.config.odometry.max_translation_per_frame,
            self.config.odometry.max_rotation_per_frame,
        ):
            self.n_rejected_motions += 1
            return predicted, 0
        return estimate.T_cw, int(estimate.inliers.sum())

    def _insert_keyframe(
        self, features: FeatureSet, stereo, index: int, timestamp: float, T_cw: np.ndarray
    ) -> int:
        """Insert a keyframe, link existing landmarks and triangulate new ones."""
        previous = self.map.keyframes[self._reference_id]
        kf = self.map.add_keyframe(index, T_cw, features.points, features.descriptors, timestamp)
        self._store_keyframe_points(kf.id, features, stereo)

        # Carry over landmarks that matched into this frame -- but only the
        # matches that are geometrically consistent with the pose just
        # estimated.  Linking every descriptor match instead is the single most
        # damaging mistake available here: roughly half of them are wrong, each
        # wrong one becomes a permanent observation with a reprojection error of
        # tens of pixels, and the local bundle adjustment then starts from a
        # 40-pixel RMSE and drags keyframes metres out of place trying to
        # satisfy the contradiction.  RANSAC already knows which matches are
        # good; the same threshold is reused here.
        matches = match_descriptors(
            features.descriptors, previous.descriptors,
            self.config.feature.ratio_test, self.config.feature.max_hamming,
        )
        linked: set[int] = set()
        stereo_lookup = {int(i): k for k, i in enumerate(stereo.index)}
        threshold = self.config.odometry.final_inlier_threshold
        candidates = []
        for query_idx, ref_idx in matches:
            landmark_id = previous.observations.get(int(ref_idx))
            if landmark_id is None or landmark_id not in self.map.landmarks:
                continue
            candidates.append((int(query_idx), int(landmark_id)))

        if candidates:
            positions = np.array([self.map.landmarks[l].position for _, l in candidates])
            predicted = project(
                transform_points(T_cw, positions),
                self.calibration.fx, self.calibration.fy,
                self.calibration.cx, self.calibration.cy,
            )
            observed = features.points[[q for q, _ in candidates]]
            consistent = np.linalg.norm(predicted - observed, axis=1) < threshold
        else:
            consistent = np.zeros(0, dtype=bool)

        for (query_idx, landmark_id), keep in zip(candidates, consistent):
            if not keep:
                continue
            k = stereo_lookup.get(query_idx)
            u_right = (
                float(stereo.uv_left[k, 0] - stereo.disparity[k]) if k is not None else None
            )
            self.map.add_observation(kf.id, query_idx, landmark_id, u_right)
            linked.add(query_idx)

        # Triangulate the rest from this frame's own stereo pair.
        T_wc = kf.T_wc
        for k, feature_index in enumerate(stereo.index):
            if int(feature_index) in linked:
                continue
            position = T_wc[:3, :3] @ stereo.points_cam[k] + T_wc[:3, 3]
            landmark = self.map.add_landmark(position, features.descriptors[feature_index])
            u_right = float(stereo.uv_left[k, 0] - stereo.disparity[k])
            self.map.add_observation(kf.id, int(feature_index), landmark.id, u_right)

        relative = kf.T_cw @ previous.T_wc
        self._edges.append(
            PoseGraphEdge(
                previous.id, kf.id, se3_inverse(relative),
                np.eye(6) * self.config.odometry_information,
            )
        )
        if self._loop_detector is not None:
            self._loop_detector.add(kf.id, features.descriptors)

        self._reference_id = kf.id
        self._frames_since_keyframe = 0
        return kf.id

    def _local_bundle_adjustment(self, keyframe_id: int) -> None:
        """Refine the covisible window around the newest keyframe."""
        window = self.map.local_window(
            keyframe_id, self.config.local_ba_window, self.config.covisibility_threshold
        )
        if len(window) < 2:
            return
        slot = {kf_id: k for k, kf_id in enumerate(window)}
        poses = np.array([self.map.keyframes[k].T_cw for k in window])

        # Gather the observations, then keep only landmarks that at least two
        # keyframes in the window actually see.  A landmark observed once has
        # three stereo residuals and three unknowns of its own: it is exactly
        # determined by that single observation and constrains no camera pose at
        # all.  Keeping them inflates the problem several-fold and makes the
        # reported reprojection RMSE meaningless.
        entries: list[tuple[int, int, np.ndarray]] = []
        seen: dict[int, int] = {}
        for kf_id in window:
            kf = self.map.keyframes[kf_id]
            for feature_index, landmark_id in kf.observations.items():
                landmark = self.map.landmarks.get(landmark_id)
                if landmark is None or feature_index not in kf.stereo_u_right:
                    continue
                uv = kf.keypoints[feature_index]
                entries.append((
                    slot[kf_id], landmark_id,
                    np.array([uv[0], uv[1], kf.stereo_u_right[feature_index]]),
                ))
                seen[landmark_id] = seen.get(landmark_id, 0) + 1

        landmark_ids: list[int] = []
        landmark_slot: dict[int, int] = {}
        cam_idx: list[int] = []
        pt_idx: list[int] = []
        observations: list[np.ndarray] = []
        for camera, landmark_id, observation in entries:
            if seen[landmark_id] < 2:
                continue
            if landmark_id not in landmark_slot:
                landmark_slot[landmark_id] = len(landmark_ids)
                landmark_ids.append(landmark_id)
            cam_idx.append(camera)
            pt_idx.append(landmark_slot[landmark_id])
            observations.append(observation)

        if len(observations) < 20 or not landmark_ids:
            return

        problem = BAProblem(
            poses_cw=poses,
            points=np.array([self.map.landmarks[i].position for i in landmark_ids]),
            camera_index=np.array(cam_idx),
            point_index=np.array(pt_idx),
            observations=np.array(observations),
            fx=self.calibration.fx, fy=self.calibration.fy,
            cx=self.calibration.cx, cy=self.calibration.cy,
            baseline=self.calibration.baseline,
            # Fix the oldest keyframe in the window: without a fixed camera the
            # window would drift bodily away from the rest of the map.
            fixed_cameras=(0,),
        )
        result = bundle_adjust(problem, self.config.ba)
        if not np.all(np.isfinite(result.poses_cw)) or not np.all(np.isfinite(result.points)):
            return
        for kf_id, k in slot.items():
            self.map.keyframes[kf_id].T_cw = result.poses_cw[k]
        for landmark_id, k in landmark_slot.items():
            self.map.landmarks[landmark_id].position = result.points[k]

    def _detect_loops(self, keyframe_id: int, features: FeatureSet) -> list:
        """Query the appearance database and geometrically verify candidates."""
        assert self._loop_detector is not None
        candidates = self._loop_detector.query(keyframe_id, features.descriptors)
        closures = []
        for candidate in candidates:
            other = self.map.keyframes.get(candidate.candidate_id)
            if other is None:
                continue
            points, valid = self._keyframe_points.get(
                candidate.candidate_id, (None, None)
            )
            if points is None or not np.any(valid):
                continue
            closure = self._loop_detector.verify(
                candidate,
                features.descriptors,
                features.points,
                other.descriptors,
                points,
                valid,
                self._K,
                self.config.odometry,
            )
            if closure is not None:
                closures.append(closure)
        return closures

    def _close_loops(self, closures: list) -> None:
        """Add loop edges and re-optimise the pose graph, then move the map."""
        for closure in closures:
            self._edges.append(
                PoseGraphEdge(
                    closure.candidate_id, closure.query_id, closure.relative_pose,
                    np.eye(6) * self.config.loop_information, is_loop=True,
                )
            )
        ids = self.map.keyframe_ids()
        slot = {kf_id: k for k, kf_id in enumerate(ids)}
        poses = np.array([self.map.keyframes[k].T_wc for k in ids])
        edges = [
            PoseGraphEdge(slot[e.i], slot[e.j], e.measurement, e.information, e.is_loop)
            for e in self._edges
            if e.i in slot and e.j in slot
        ]
        if not edges:
            return
        result = optimise_pose_graph(poses, edges, self.config.posegraph, fixed=(0,))
        self.map.apply_pose_correction(
            {kf_id: result.poses[slot[kf_id]] for kf_id in ids}
        )
        self._last_T_cw = self.map.keyframes[self._reference_id].T_cw
