"""Geodetic conversions and the OXTS ground-truth trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from svslam.dataset.geodesy import (
    EARTH_RADIUS_M,
    EnuOrigin,
    ecef_to_enu,
    geodetic_to_ecef,
    geodetic_to_enu,
    mercator_scale,
    mercator_xy,
)
from svslam.dataset.kitti import OxtsRecord, oxts_to_poses
from svslam.se3 import se3_inverse

# Roughly where the KITTI recordings were made.
LAT0, LON0, ALT0 = 49.011212, 8.4228850, 112.83


def test_mercator_scale_is_cos_latitude():
    assert np.isclose(mercator_scale(0.0), 1.0)
    assert np.isclose(mercator_scale(60.0), 0.5)
    assert np.isclose(mercator_scale(LAT0), np.cos(np.deg2rad(LAT0)))
    # At KITTI's latitude, skipping the scale is a ~52% error, not a rounding one.
    assert 0.65 < mercator_scale(LAT0) < 0.66


def test_mercator_origin_and_east_direction():
    scale = mercator_scale(LAT0)
    origin = mercator_xy(LAT0, LON0, scale)
    east = mercator_xy(LAT0, LON0 + 0.001, scale)
    north = mercator_xy(LAT0 + 0.001, LON0, scale)
    assert east[0] > origin[0]
    assert np.isclose(east[1], origin[1])
    assert north[1] > origin[1]
    assert np.isclose(north[0], origin[0])


def test_mercator_metres_per_degree_of_longitude():
    """One degree of longitude is R cos(lat) pi/180 metres on the ground."""
    scale = mercator_scale(LAT0)
    a = mercator_xy(LAT0, LON0, scale)
    b = mercator_xy(LAT0, LON0 + 1.0, scale)
    expected = EARTH_RADIUS_M * np.cos(np.deg2rad(LAT0)) * np.deg2rad(1.0)
    assert np.isclose(b[0] - a[0], expected, rtol=1e-12)


def test_ecef_round_trip_through_enu():
    origin = EnuOrigin(LAT0, LON0, ALT0)
    assert np.allclose(ecef_to_enu(origin.ecef, origin), 0.0, atol=1e-6)
    assert np.allclose(origin.rotation @ origin.rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(origin.rotation), 1.0)


def test_enu_axes_point_the_right_way():
    origin = EnuOrigin(LAT0, LON0, ALT0)
    east = geodetic_to_enu(LAT0, LON0 + 0.001, ALT0, origin)
    north = geodetic_to_enu(LAT0 + 0.001, LON0, ALT0, origin)
    up = geodetic_to_enu(LAT0, LON0, ALT0 + 10.0, origin)
    assert east[0] > 0 and abs(east[1]) < 1e-3 and abs(east[2]) < 1.0
    assert north[1] > 0 and abs(north[0]) < 1e-6
    assert np.isclose(up[2], 10.0, atol=1e-6)


def test_geodetic_to_ecef_matches_known_geometry():
    """At the equator and prime meridian, ECEF is (a + h, 0, 0)."""
    ecef = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.isclose(ecef[0], 6378137.0)
    assert np.allclose(ecef[1:], 0.0, atol=1e-6)
    pole = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert np.isclose(pole[2], 6356752.314245, atol=1e-3)


def _records(n=40, step_deg=1e-4):
    return [
        OxtsRecord(LAT0 + i * step_deg * 0.0, LON0 + i * step_deg, ALT0, 0.0, 0.0, 0.0)
        for i in range(n)
    ]


def test_oxts_poses_start_at_the_identity():
    poses = oxts_to_poses(_records())
    assert np.allclose(poses[0], np.eye(4), atol=1e-9)


def test_oxts_poses_have_constant_spacing_for_constant_steps():
    poses = oxts_to_poses(_records())
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    assert np.allclose(steps, steps[0], rtol=1e-9)
    assert steps[0] > 0.0


def test_mercator_and_enu_agree_to_within_a_fraction_of_a_percent():
    """Two different projections of the same points; the disagreement is small.

    They are not identical -- Mercator is spherical and ENU is ellipsoidal -- so
    the point of the test is that the choice is a fraction of a percent of path
    length, not that it does not matter.
    """
    records = _records(200, 2e-4)
    mercator = oxts_to_poses(records, projection="mercator")
    enu = oxts_to_poses(records, projection="enu")
    length_m = np.linalg.norm(np.diff(mercator[:, :3, 3], axis=0), axis=1).sum()
    length_e = np.linalg.norm(np.diff(enu[:, :3, 3], axis=0), axis=1).sum()
    assert abs(length_m - length_e) / length_e < 0.005


def test_unknown_projection_is_rejected():
    with pytest.raises(ValueError):
        oxts_to_poses(_records(), projection="utm")


def test_empty_records_give_an_empty_trajectory():
    assert oxts_to_poses([]).shape == (0, 4, 4)


def test_camera_change_of_basis_preserves_relative_motion():
    """Changing frame must not change how far the vehicle travelled."""
    records = _records(50, 2e-4)
    imu_poses = oxts_to_poses(records)
    T_cam_imu = np.array([
        [0.0, -1.0, 0.0, -0.32],
        [0.0, 0.0, -1.0, 0.73],
        [1.0, 0.0, 0.0, -1.14],
        [0.0, 0.0, 0.0, 1.0],
    ])
    cam_poses = oxts_to_poses(records, T_cam_imu=T_cam_imu)
    assert np.allclose(cam_poses[0], np.eye(4), atol=1e-9)
    for i in range(1, len(records)):
        a = se3_inverse(imu_poses[i - 1]) @ imu_poses[i]
        b = se3_inverse(cam_poses[i - 1]) @ cam_poses[i]
        assert np.isclose(np.linalg.norm(a[:3, 3]), np.linalg.norm(b[:3, 3]), atol=1e-9)


def test_yaw_becomes_a_rotation_about_the_camera_y_axis():
    """The vehicle turning left must show up as yaw in the camera frame too."""
    records = [
        OxtsRecord(LAT0, LON0, ALT0, 0.0, 0.0, 0.0),
        OxtsRecord(LAT0, LON0, ALT0, 0.0, 0.0, 0.5),
    ]
    T_cam_imu = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    poses = oxts_to_poses(records, T_cam_imu=T_cam_imu)
    from svslam.se3 import so3_log

    axis = so3_log(poses[1][:3, :3])
    assert np.isclose(np.linalg.norm(axis), 0.5, atol=1e-9)
    # Camera y is down, so a left turn is a negative rotation about it.
    assert abs(axis[1]) > 0.99 * np.linalg.norm(axis)
