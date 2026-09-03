"""The plausibility gate that stops one bad frame ruining a whole sequence."""

from __future__ import annotations

import numpy as np

from svslam.frontend.odometry import OdometryConfig, is_plausible_motion
from svslam.se3 import se3_exp


def _config() -> OdometryConfig:
    return OdometryConfig()


def test_a_normal_driving_step_is_accepted():
    config = _config()
    # 1.2 m and 2 degrees in one 100 ms frame: about 43 km/h through a bend.
    motion = se3_exp(np.array([0.0, 0.0, 1.2, 0.0, np.deg2rad(2.0), 0.0]))
    assert is_plausible_motion(
        motion, config.max_translation_per_frame, config.max_rotation_per_frame
    )


def test_standing_still_is_accepted():
    config = _config()
    assert is_plausible_motion(
        np.eye(4), config.max_translation_per_frame, config.max_rotation_per_frame
    )


def test_an_impossible_jump_is_rejected():
    """The exact failure this gate exists for: a 12 m step in one frame."""
    config = _config()
    motion = se3_exp(np.array([3.0, 1.0, 12.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(motion[:3, 3]) > config.max_translation_per_frame
    assert not is_plausible_motion(
        motion, config.max_translation_per_frame, config.max_rotation_per_frame
    )


def test_an_impossible_rotation_is_rejected():
    config = _config()
    motion = se3_exp(np.array([0.1, 0.0, 0.5, 0.0, np.deg2rad(40.0), 0.0]))
    assert not is_plausible_motion(
        motion, config.max_translation_per_frame, config.max_rotation_per_frame
    )


def test_non_finite_poses_are_rejected():
    broken = np.eye(4)
    broken[0, 3] = np.nan
    assert not is_plausible_motion(broken, 4.0, 0.35)
    broken[0, 3] = np.inf
    assert not is_plausible_motion(broken, 4.0, 0.35)


def test_the_gate_is_loose_enough_not_to_fire_on_motorway_speed():
    """It must catch the impossible, not the merely fast."""
    config = _config()
    # 2.5 m per 100 ms frame is 90 km/h, a perfectly normal KITTI speed.
    fast = se3_exp(np.array([0.0, 0.0, 2.5, 0.0, 0.0, 0.0]))
    assert is_plausible_motion(
        fast, config.max_translation_per_frame, config.max_rotation_per_frame
    )


def test_thresholds_are_configurable():
    motion = se3_exp(np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]))
    assert is_plausible_motion(motion, 4.0, 0.35)
    assert not is_plausible_motion(motion, 1.0, 0.35)
