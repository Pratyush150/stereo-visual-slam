"""Dataset readers.

The layout-parsing tests run everywhere against fixture directories.  The tests
that need the real download are skipped, not failed, when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from svslam.dataset.kitti import (
    KittiOdometryDataset,
    KittiRawDataset,
    open_sequence,
    read_oxts_records,
)

from conftest import requires_cv2, requires_kitti

P0 = "7.070912e+02 0 6.018873e+02 0 0 7.070912e+02 1.831104e+02 0 0 0 1 0"
P1 = "7.070912e+02 0 6.018873e+02 -3.798145e+02 0 7.070912e+02 1.831104e+02 0 0 0 1 0"


def _make_odometry_sequence(tmp_path, n=4):
    import cv2

    sequence = tmp_path / "sequences" / "07"
    (sequence / "image_0").mkdir(parents=True)
    (sequence / "image_1").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        image = rng.integers(0, 255, (30, 60), dtype=np.uint8)
        cv2.imwrite(str(sequence / "image_0" / f"{i:06d}.png"), image)
        cv2.imwrite(str(sequence / "image_1" / f"{i:06d}.png"), image)
    (sequence / "calib.txt").write_text(
        f"P0: {P0}\nP1: {P1}\nP2: {P0}\nP3: {P1}\n"
    )
    (sequence / "times.txt").write_text("\n".join(f"{i * 0.1:.6e}" for i in range(n)) + "\n")

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    rows = []
    for i in range(n):
        T = np.eye(4)
        T[2, 3] = i * 1.5
        rows.append(" ".join(f"{v:.9e}" for v in T[:3, :4].reshape(-1)))
    (poses_dir / "07.txt").write_text("\n".join(rows) + "\n")
    return sequence


def _make_raw_drive(tmp_path, n=5):
    import cv2

    day = tmp_path / "2011_09_30"
    drive = day / "2011_09_30_drive_0027_sync"
    for camera in ("image_00", "image_01"):
        (drive / camera / "data").mkdir(parents=True)
    (drive / "oxts" / "data").mkdir(parents=True)

    rng = np.random.default_rng(1)
    stamps = []
    for i in range(n):
        image = rng.integers(0, 255, (30, 60), dtype=np.uint8)
        cv2.imwrite(str(drive / "image_00" / "data" / f"{i:010d}.png"), image)
        cv2.imwrite(str(drive / "image_01" / "data" / f"{i:010d}.png"), image)
        (drive / "oxts" / "data" / f"{i:010d}.txt").write_text(
            f"49.01 {8.42 + i * 1e-4:.10f} 112.8 0 0 0 " + "0 " * 24 + "\n"
        )
        stamps.append(f"2011-09-30 12:00:{i * 0.1:09.6f}")
    (drive / "image_00" / "timestamps.txt").write_text("\n".join(stamps) + "\n")

    def flat(a):
        return " ".join(f"{v:.12e}" for v in np.asarray(a).reshape(-1))

    (day / "calib_cam_to_cam.txt").write_text(
        f"S_rect_00: {flat([60.0, 30.0])}\nR_rect_00: {flat(np.eye(3))}\n"
        f"P_rect_00: {P0}\nS_rect_01: {flat([60.0, 30.0])}\nP_rect_01: {P1}\n"
    )
    (day / "calib_velo_to_cam.txt").write_text(
        f"R: {flat(np.eye(3))}\nT: {flat([0.0, -0.08, -0.27])}\n"
    )
    (day / "calib_imu_to_velo.txt").write_text(
        f"R: {flat(np.eye(3))}\nT: {flat([-0.81, 0.32, -0.80])}\n"
    )
    return drive


@requires_cv2
def test_odometry_layout(tmp_path):
    sequence = _make_odometry_sequence(tmp_path)
    dataset = KittiOdometryDataset(sequence)
    assert len(dataset) == 4
    assert dataset.calibration.image_size == (60, 30)
    assert np.isclose(dataset.calibration.baseline, 3.798145e02 / 7.070912e02)
    left, right = dataset.load_stereo(0)
    assert left.shape == (30, 60) and right.shape == (30, 60)
    gt = dataset.ground_truth()
    assert gt is not None and gt.shape == (4, 4, 4)
    assert np.allclose(gt[:, 2, 3], [0.0, 1.5, 3.0, 4.5])
    assert np.allclose(dataset.timestamps, [0.0, 0.1, 0.2, 0.3])


@requires_cv2
def test_raw_layout(tmp_path):
    drive = _make_raw_drive(tmp_path)
    dataset = KittiRawDataset(drive)
    assert len(dataset) == 5
    assert dataset.calibration.T_cam_imu is not None
    gt = dataset.ground_truth()
    assert gt is not None and gt.shape == (5, 4, 4)
    assert np.allclose(gt[0], np.eye(4), atol=1e-9)
    steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
    assert np.allclose(steps, steps[0], rtol=1e-6)
    assert dataset.timestamps[-1] > dataset.timestamps[0]


@requires_cv2
def test_open_sequence_detects_the_layout(tmp_path):
    sequence = _make_odometry_sequence(tmp_path)
    assert isinstance(open_sequence(sequence), KittiOdometryDataset)
    drive = _make_raw_drive(tmp_path / "raw")
    assert isinstance(open_sequence(drive), KittiRawDataset)
    with pytest.raises(FileNotFoundError):
        open_sequence(tmp_path / "not-a-sequence")


@requires_cv2
def test_frames_iterator(tmp_path):
    sequence = _make_odometry_sequence(tmp_path)
    dataset = KittiOdometryDataset(sequence)
    frames = list(dataset.frames(1, 4, 2))
    assert [f[0] for f in frames] == [1, 3]
    assert all(f[1].shape == (30, 60) for f in frames)


def test_oxts_reader_handles_a_flat_directory(tmp_path):
    (tmp_path / "0000000000.txt").write_text("49.0 8.4 100.0 0.1 0.2 0.3 " + "0 " * 24)
    (tmp_path / "0000000001.txt").write_text("49.1 8.5 101.0 0.0 0.0 0.0 " + "0 " * 24)
    records = read_oxts_records(tmp_path)
    assert len(records) == 2
    assert np.isclose(records[0].lat, 49.0)
    assert np.isclose(records[0].yaw, 0.3)
    assert np.isclose(records[1].alt, 101.0)


def test_oxts_reader_on_an_empty_directory(tmp_path):
    assert read_oxts_records(tmp_path) == []


# ---------------------------------------------------------------------------
# The real download.
# ---------------------------------------------------------------------------


@requires_kitti
def test_real_sequence_loads(kitti_sequence):
    assert len(kitti_sequence) > 100
    left, right = kitti_sequence.load_stereo(0)
    assert left.shape == right.shape
    assert left.ndim == 2
    assert 0.5 < kitti_sequence.calibration.baseline < 0.6
    assert 300.0 < kitti_sequence.calibration.fx < 1200.0


@requires_kitti
def test_real_ground_truth_is_metric_and_smooth(kitti_sequence):
    gt = kitti_sequence.ground_truth()
    if gt is None:
        pytest.skip("this sequence ships no ground truth")
    assert np.allclose(gt[0], np.eye(4), atol=1e-9)
    steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
    # A car at 10 Hz: every frame-to-frame step is a plausible distance.
    assert steps.max() < 4.0
    assert steps.mean() > 0.05
    for i in range(gt.shape[0]):
        assert np.allclose(gt[i, :3, :3] @ gt[i, :3, :3].T, np.eye(3), atol=1e-9)


@requires_kitti
def test_projection_choice_changes_path_length_only_slightly(kitti_sequence):
    """A real measurement of how much the projection choice actually matters."""
    from svslam.dataset.kitti import KittiRawDataset as Raw
    from svslam.dataset.kitti import oxts_to_poses, read_oxts_records

    if not isinstance(kitti_sequence, Raw):
        pytest.skip("raw layout only")
    records = read_oxts_records(kitti_sequence.root / "oxts")
    mercator = oxts_to_poses(records, projection="mercator",
                             T_cam_imu=kitti_sequence.calibration.T_cam_imu)
    enu = oxts_to_poses(records, projection="enu",
                        T_cam_imu=kitti_sequence.calibration.T_cam_imu)
    length_m = np.linalg.norm(np.diff(mercator[:, :3, 3], axis=0), axis=1).sum()
    length_e = np.linalg.norm(np.diff(enu[:, :3, 3], axis=0), axis=1).sum()
    assert abs(length_m - length_e) / length_e < 0.01
