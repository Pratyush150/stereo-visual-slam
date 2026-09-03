"""KITTI calibration parsing, for both dataset layouts."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.dataset.kitti import (
    StereoCalibration,
    load_odometry_calibration,
    load_raw_calibration,
    parse_calib_file,
)

# The real 2011_09_30 rectified greyscale projection matrices.
P0 = np.array([
    [7.070912e02, 0.0, 6.018873e02, 0.0],
    [0.0, 7.070912e02, 1.831104e02, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])
P1 = np.array([
    [7.070912e02, 0.0, 6.018873e02, -3.798145e02],
    [0.0, 7.070912e02, 1.831104e02, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])


def _write_raw_calibration(directory, R_rect=None, R_velo=None, T_velo=None):
    def flat(a):
        return " ".join(f"{v:.12e}" for v in np.asarray(a).reshape(-1))

    R_rect = np.eye(3) if R_rect is None else R_rect
    lines = [
        "calib_time: 09-Jan-2012 13:59:33",
        "corner_dist: 9.950000e-02",
        f"S_rect_00: {flat([1226.0, 370.0])}",
        f"R_rect_00: {flat(R_rect)}",
        f"P_rect_00: {flat(P0)}",
        f"S_rect_01: {flat([1226.0, 370.0])}",
        f"P_rect_01: {flat(P1)}",
        f"S_rect_02: {flat([1226.0, 370.0])}",
        f"P_rect_02: {flat(P0)}",
        f"S_rect_03: {flat([1226.0, 370.0])}",
        f"P_rect_03: {flat(P1)}",
    ]
    (directory / "calib_cam_to_cam.txt").write_text("\n".join(lines) + "\n")

    R_velo = np.eye(3) if R_velo is None else R_velo
    T_velo = np.array([0.0, -0.08, -0.27]) if T_velo is None else T_velo
    (directory / "calib_velo_to_cam.txt").write_text(
        f"R: {flat(R_velo)}\nT: {flat(T_velo)}\n"
    )
    (directory / "calib_imu_to_velo.txt").write_text(
        f"R: {flat(np.eye(3))}\nT: {flat([-0.81, 0.32, -0.80])}\n"
    )


def test_parse_calib_file_skips_non_numeric_entries(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "calib_time: 09-Jan-2012 13:59:33\n"
        "corner_dist: 9.950000e-02\n"
        "K: 1 0 2 0 1 3 0 0 1\n"
        "\n"
        "junk-line-without-a-colon\n"
    )
    parsed = parse_calib_file(path)
    assert "calib_time" not in parsed
    assert np.isclose(parsed["corner_dist"], 0.0995)
    assert parsed["K"].shape == (9,)


def test_raw_calibration_round_trips(tmp_path):
    _write_raw_calibration(tmp_path)
    calibration = load_raw_calibration(tmp_path)

    assert np.allclose(calibration.P_left, P0)
    assert np.allclose(calibration.P_right, P1)
    assert calibration.image_size == (1226, 370)
    assert np.isclose(calibration.fx, 707.0912)
    assert np.isclose(calibration.fy, 707.0912)
    assert np.isclose(calibration.cx, 601.8873)
    assert np.isclose(calibration.cy, 183.1104)
    assert np.allclose(calibration.K, P0[:3, :3])


def test_baseline_is_recovered_from_the_projection_matrices(tmp_path):
    """P_right[0, 3] == -fx * b, so the baseline never has to be hard-coded."""
    _write_raw_calibration(tmp_path)
    calibration = load_raw_calibration(tmp_path)
    assert np.isclose(calibration.baseline, 3.798145e02 / 7.070912e02)
    assert 0.53 < calibration.baseline < 0.545


def test_imu_to_camera_chain_is_composed(tmp_path):
    """T_cam_imu = R_rect . T_velo_cam . T_imu_velo, and it must be rigid."""
    _write_raw_calibration(tmp_path)
    calibration = load_raw_calibration(tmp_path)
    T = calibration.T_cam_imu
    assert T is not None and T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    # With identity rotations the translations simply add.
    assert np.allclose(T[:3, 3], np.array([-0.81, 0.24, -1.07]), atol=1e-9)


def test_missing_extrinsics_leave_the_chain_unset(tmp_path):
    _write_raw_calibration(tmp_path)
    (tmp_path / "calib_velo_to_cam.txt").unlink()
    calibration = load_raw_calibration(tmp_path)
    assert calibration.T_cam_imu is None
    assert calibration.T_cam_velo is None
    # Intrinsics still work.
    assert np.isclose(calibration.fx, 707.0912)


def test_colour_pair_selection(tmp_path):
    _write_raw_calibration(tmp_path)
    grey = load_raw_calibration(tmp_path, colour=False)
    colour = load_raw_calibration(tmp_path, colour=True)
    assert np.allclose(grey.P_left, colour.P_left)  # same in this fixture
    assert grey.image_size == colour.image_size


def test_odometry_calibration(tmp_path):
    def flat(a):
        return " ".join(f"{v:.12e}" for v in np.asarray(a).reshape(-1))

    Tr = np.hstack([np.eye(3), np.array([[0.0], [-0.08], [-0.27]])])
    (tmp_path / "calib.txt").write_text(
        f"P0: {flat(P0)}\nP1: {flat(P1)}\nP2: {flat(P0)}\nP3: {flat(P1)}\n"
        f"Tr: {flat(Tr)}\n"
    )
    calibration = load_odometry_calibration(tmp_path / "calib.txt")
    assert np.allclose(calibration.P_left, P0)
    assert np.isclose(calibration.baseline, 3.798145e02 / 7.070912e02)
    assert calibration.T_cam_velo is not None
    assert np.allclose(calibration.T_cam_velo[:3, 3], [0.0, -0.08, -0.27])


def test_calibration_to_dict_is_serialisable(tmp_path):
    _write_raw_calibration(tmp_path)
    data = load_raw_calibration(tmp_path).to_dict()
    assert data["image_size"] == [1226, 370]
    assert isinstance(data["P_left"], list)
    assert np.isclose(data["baseline"], 3.798145e02 / 7.070912e02)


def test_calibration_is_constructible_directly():
    calibration = StereoCalibration(P_left=P0, P_right=P1, image_size=(1226, 370))
    assert np.isclose(calibration.baseline, 3.798145e02 / 7.070912e02)
    assert calibration.T_cam_imu is None
