"""End-to-end pipeline behaviour on a synthetic stereo sequence.

A real KITTI drive is several gigabytes, so the pipeline is exercised here on a
rendered scene instead: a textured plane and some boxes seen by a stereo rig
moving along a known trajectory.  It is not photorealistic, but it has real
disparity, real parallax and a known answer, which is what the pipeline needs to
be tested against without a download.
"""

from __future__ import annotations

import numpy as np
import pytest

from svslam.dataset.kitti import StereoCalibration
from svslam.frontend.odometry import KeyframePolicy
from svslam.pipeline import PipelineConfig, StereoSlam
from svslam.se3 import se3_exp, se3_inverse

from conftest import requires_cv2

FX = FY = 400.0
CX, CY = 320.0, 180.0
WIDTH, HEIGHT = 512, 288
BASELINE = 0.5


def _calibration() -> StereoCalibration:
    P_left = np.array([[FX, 0.0, CX, 0.0], [0.0, FY, CY, 0.0], [0.0, 0.0, 1.0, 0.0]])
    P_right = P_left.copy()
    P_right[0, 3] = -FX * BASELINE
    return StereoCalibration(P_left=P_left, P_right=P_right, image_size=(WIDTH, HEIGHT))


class SyntheticStereoSequence:
    """A rendered stereo corridor with an exactly known ground truth.

    The scene is four textured planes -- a ground plane, two side walls and an
    end wall -- and each frame is produced by casting a ray through every pixel
    and sampling the texture where it hits.  That is slower than splatting
    points, and it is worth it: the images have correct perspective, correct
    parallax between frames and correct stereo disparity, so a pipeline that
    tracks them is doing real work rather than matching a blob pattern.

    Camera convention matches KITTI: x right, y down, z forward.
    """

    #: (point on plane, unit normal, two in-plane axes for texture coordinates)
    PLANES = (
        (np.array([0.0, 1.65, 0.0]), np.array([0.0, -1.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([-7.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([7.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 0.0, 42.0]), np.array([0.0, 0.0, -1.0]),
         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    )

    def __init__(self, n_frames: int = 16, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.calibration = _calibration()
        self.timestamps = np.arange(n_frames) * 0.1
        self._n = n_frames
        self._texture = self._fractal_texture(rng)
        self._texels_per_metre = 24.0
        self._poses = np.array([
            se3_exp(np.array([0.0, 0.0, 0.45 * i, 0.0, 0.004 * i, 0.0]))
            for i in range(n_frames)
        ])
        self._rays = self._pixel_rays()

    def __len__(self) -> int:
        return self._n

    @staticmethod
    def _fractal_texture(rng, size: int = 1024) -> np.ndarray:
        """Multi-octave noise, because white noise is unmatchable.

        A flat random texture looks rich but every 31x31 patch of it resembles
        every other one, so the ratio test and the mutual-best cross-check throw
        almost all matches away.  Summing octaves gives structure at several
        scales at once, which is what makes a patch locally distinctive -- and it
        is what real surfaces look like.
        """
        image = np.zeros((size, size), dtype=float)
        amplitude, total = 1.0, 0.0
        for octave in (4, 8, 16, 32, 64, 128, 256):
            grid = rng.normal(size=(octave, octave))
            repeat = size // octave
            image += amplitude * np.repeat(
                np.repeat(grid, repeat, axis=0), repeat, axis=1
            )[:size, :size]
            total += amplitude
            amplitude *= 0.7
        image /= total
        image = (image - image.min()) / (image.max() - image.min())
        return (20.0 + 215.0 * image).astype(np.uint8)

    def _pixel_rays(self) -> np.ndarray:
        """Unit direction, in camera coordinates, for the centre of every pixel."""
        u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
        rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
        return rays.reshape(-1, 3)

    def _render(self, T_wc: np.ndarray, baseline_offset: float) -> np.ndarray:
        centre = T_wc[:3, 3] + T_wc[:3, :3] @ np.array([baseline_offset, 0.0, 0.0])
        directions = self._rays @ T_wc[:3, :3].T

        best_t = np.full(directions.shape[0], np.inf)
        colour = np.zeros(directions.shape[0], dtype=np.uint8)
        for origin, normal, axis_u, axis_v in self.PLANES:
            denominator = directions @ normal
            hit = np.abs(denominator) > 1e-6
            t = np.full(directions.shape[0], np.inf)
            t[hit] = ((origin - centre) @ normal) / denominator[hit]
            valid = hit & (t > 0.5) & (t < best_t)
            if not np.any(valid):
                continue
            points = centre + directions[valid] * t[valid, None]
            tu = (points @ axis_u) * self._texels_per_metre
            tv = (points @ axis_v) * self._texels_per_metre
            iu = np.mod(np.round(tu).astype(np.int64), self._texture.shape[1])
            iv = np.mod(np.round(tv).astype(np.int64), self._texture.shape[0])
            colour[valid] = self._texture[iv, iu]
            best_t[valid] = t[valid]
        return colour.reshape(HEIGHT, WIDTH)

    def load_stereo(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        T_wc = self._poses[index]
        return self._render(T_wc, 0.0), self._render(T_wc, BASELINE)

    def load_left(self, index: int) -> np.ndarray:
        return self._render(self._poses[index], 0.0)

    def ground_truth(self) -> np.ndarray:
        return self._poses


@pytest.fixture(scope="module")
def synthetic():
    return SyntheticStereoSequence()


def _config() -> PipelineConfig:
    from dataclasses import replace

    config = PipelineConfig()
    config.enable_loop = False
    config.local_ba_window = 4
    config.feature = replace(config.feature, max_features=900, grid_rows=4,
                             grid_cols=8, max_per_cell=32)
    config.keyframe = KeyframePolicy(min_tracked=40, max_tracked_ratio=0.5,
                                     translation_threshold=1.5, rotation_threshold=0.2,
                                     max_frames=5, min_frames=2)
    return config


@requires_cv2
def test_pipeline_tracks_a_synthetic_sequence(synthetic):
    slam = StereoSlam(synthetic.calibration, _config())
    result = slam.run(synthetic, 0, len(synthetic))

    assert len(result.frame_indices) == len(synthetic)
    assert result.poses.shape == (len(synthetic), 4, 4)
    assert np.all(np.isfinite(result.poses))
    assert result.stats["keyframes"] >= 2
    assert result.stats["landmarks"] > 200
    assert np.allclose(result.poses[0], np.eye(4), atol=1e-9)


@requires_cv2
def test_pipeline_recovers_metric_scale(synthetic):
    """Stereo observes scale directly; the path length must come out right."""
    slam = StereoSlam(synthetic.calibration, _config())
    result = slam.run(synthetic, 0, len(synthetic))
    truth = synthetic.ground_truth()

    estimated = np.linalg.norm(np.diff(result.poses[:, :3, 3], axis=0), axis=1).sum()
    actual = np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum()
    assert actual > 5.0  # the trajectory is long enough for the claim to mean something
    assert abs(estimated - actual) / actual < 0.05


@requires_cv2
def test_pipeline_trajectory_is_close_to_the_truth(synthetic):
    from svslam.evaluation.kitti_metrics import absolute_trajectory_error

    slam = StereoSlam(synthetic.calibration, _config())
    result = slam.run(synthetic, 0, len(synthetic))
    ate = absolute_trajectory_error(result.poses, synthetic.ground_truth(), align=True)
    assert ate["rmse"] < 0.3


@requires_cv2
def test_pipeline_records_timings_and_stats(synthetic):
    slam = StereoSlam(synthetic.calibration, _config())
    result = slam.run(synthetic, 0, 12)
    assert set(result.timings) >= {"features", "stereo"}
    assert all(v >= 0.0 for v in result.timings.values())
    assert result.stats["mean_tracked_features"] > 0.0
    assert 0.0 <= result.stats["mean_feature_spread"] <= 1.0
    assert len(result.tracked_per_frame) == 12


@requires_cv2
def test_keyframe_poses_are_a_subset_of_the_trajectory(synthetic):
    slam = StereoSlam(synthetic.calibration, _config())
    result = slam.run(synthetic, 0, len(synthetic))
    assert result.keyframe_frame_indices[0] == 0
    assert set(result.keyframe_frame_indices) <= set(result.frame_indices)
    assert len(result.keyframe_frame_indices) == result.slam_map.n_keyframes


@requires_cv2
def test_disabling_bundle_adjustment_still_produces_a_trajectory(synthetic):
    config = _config()
    config.enable_ba = False
    slam = StereoSlam(synthetic.calibration, config)
    result = slam.run(synthetic, 0, 15)
    assert np.all(np.isfinite(result.poses))
    assert "bundle_adjustment" not in result.timings


def test_keyframe_policy_fires_on_each_of_its_conditions():
    policy = KeyframePolicy(min_tracked=50, max_tracked_ratio=0.5,
                            translation_threshold=2.0, rotation_threshold=0.1,
                            max_frames=10, min_frames=2)
    identity = np.eye(4)
    # Too soon after the last keyframe.
    assert not policy.should_insert(1, 10, 100, identity)
    # Too few tracked features.
    assert policy.should_insert(3, 10, 100, identity)
    # Lost too large a fraction of the reference's features.
    assert policy.should_insert(3, 40, 100, identity)
    # Travelled far enough.
    far = np.eye(4)
    far[2, 3] = 3.0
    assert policy.should_insert(3, 200, 100, far)
    # Rotated far enough.
    turned = se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0]))
    assert policy.should_insert(3, 200, 100, turned)
    # Nothing happened, but the frame budget ran out anyway.
    assert policy.should_insert(10, 200, 100, identity)
    # Nothing happened and there is budget left.
    assert not policy.should_insert(3, 200, 100, identity)
