"""Map bookkeeping: observations, covisibility, culling, pose correction."""

from __future__ import annotations

import numpy as np

from svslam.map import SlamMap
from svslam.se3 import se3_exp, se3_inverse


def _map_with_shared_landmarks(n_keyframes=4, n_landmarks=30, shared=20):
    """Keyframes 0..n-1, the first ``shared`` landmarks seen by all of them."""
    slam_map = SlamMap()
    for i in range(n_keyframes):
        T_cw = se3_inverse(se3_exp(np.array([0.0, 0.0, float(i), 0.0, 0.0, 0.0])))
        slam_map.add_keyframe(
            frame_index=i * 5, T_cw=T_cw,
            keypoints=np.zeros((n_landmarks, 2)),
            descriptors=np.zeros((n_landmarks, 32), np.uint8),
        )
    for j in range(n_landmarks):
        landmark = slam_map.add_landmark(np.array([float(j), 0.0, 10.0]),
                                         np.zeros(32, np.uint8))
        observers = range(n_keyframes) if j < shared else [0]
        for kf_id in observers:
            slam_map.add_observation(kf_id, j, landmark.id, u_right=float(j) - 20.0)
    return slam_map


def test_keyframe_and_landmark_counts():
    slam_map = _map_with_shared_landmarks()
    assert slam_map.n_keyframes == 4
    assert slam_map.n_landmarks == 30
    assert slam_map.keyframe_ids() == [0, 1, 2, 3]


def test_observations_are_linked_both_ways():
    slam_map = _map_with_shared_landmarks()
    kf = slam_map.keyframes[1]
    assert kf.observations[5] == 5
    assert slam_map.landmarks[5].observations[1] == 5
    assert slam_map.landmarks[5].n_observations == 4
    assert kf.stereo_u_right[5] == -15.0


def test_pose_accessors_are_consistent():
    slam_map = _map_with_shared_landmarks()
    kf = slam_map.keyframes[2]
    assert np.allclose(kf.T_wc @ kf.T_cw, np.eye(4), atol=1e-12)
    assert np.allclose(kf.centre, kf.T_wc[:3, 3])
    assert np.isclose(kf.centre[2], 2.0)


def test_trajectory_is_in_keyframe_order():
    slam_map = _map_with_shared_landmarks()
    trajectory = slam_map.trajectory()
    assert trajectory.shape == (4, 4, 4)
    assert np.allclose(trajectory[:, 2, 3], [0.0, 1.0, 2.0, 3.0])


def test_covisibility_counts_shared_landmarks():
    slam_map = _map_with_shared_landmarks(shared=20)
    graph = slam_map.covisibility(min_shared=15)
    assert graph[0][1] == 20
    assert graph[3][0] == 20
    # A higher threshold prunes the graph.
    assert slam_map.covisibility(min_shared=25) == {i: {} for i in range(4)}


def test_local_window_prefers_covisible_keyframes():
    slam_map = _map_with_shared_landmarks(n_keyframes=6)
    window = slam_map.local_window(3, size=3, min_shared=15)
    assert 3 in window
    assert len(window) == 3


def test_local_window_falls_back_to_temporal_neighbours():
    """At the start of a sequence the covisibility graph is empty."""
    slam_map = _map_with_shared_landmarks(n_keyframes=5, shared=0)
    window = slam_map.local_window(2, size=3, min_shared=15)
    assert window == [1, 2, 3]


def test_culling_removes_singly_observed_landmarks():
    slam_map = _map_with_shared_landmarks(n_keyframes=4, n_landmarks=30, shared=20)
    removed = slam_map.cull_landmarks(min_observations=2)
    assert removed == 10
    assert slam_map.n_landmarks == 20
    # The observation links are removed too, not just the landmark.
    assert all(lm_id < 20 for lm_id in slam_map.keyframes[0].observations.values())


def test_culling_uses_the_found_ratio():
    slam_map = _map_with_shared_landmarks()
    landmark = slam_map.landmarks[0]
    landmark.n_visible = 100
    landmark.n_found = 1
    assert landmark.found_ratio < 0.05
    slam_map.cull_landmarks(min_observations=1, min_found_ratio=0.25)
    assert 0 not in slam_map.landmarks


def test_pose_correction_moves_landmarks_with_their_anchor():
    """Moving keyframes without moving landmarks silently destroys the map."""
    slam_map = _map_with_shared_landmarks()
    before = {i: lm.position.copy() for i, lm in slam_map.landmarks.items()}
    delta = se3_exp(np.array([1.0, -2.0, 0.5, 0.0, 0.1, 0.0]))
    corrected = {kf_id: delta @ kf.T_wc for kf_id, kf in slam_map.keyframes.items()}

    slam_map.apply_pose_correction(corrected)

    for lm_id, lm in slam_map.landmarks.items():
        expected = delta[:3, :3] @ before[lm_id] + delta[:3, 3]
        assert np.allclose(lm.position, expected, atol=1e-9)
    # The keyframe-relative geometry is unchanged, which is the invariant.
    kf = slam_map.keyframes[0]
    local = kf.T_cw[:3, :3] @ slam_map.landmarks[0].position + kf.T_cw[:3, 3]
    assert np.allclose(local, [0.0, 0.0, 10.0], atol=1e-9)


def test_landmark_positions_array():
    slam_map = _map_with_shared_landmarks()
    positions = slam_map.landmark_positions()
    assert positions.shape == (30, 3)
    assert np.allclose(positions[:, 2], 10.0)


def test_empty_map_is_safe():
    slam_map = SlamMap()
    assert slam_map.trajectory().shape == (0, 4, 4)
    assert slam_map.landmark_positions().shape == (0, 3)
    assert slam_map.local_window(0, 5) == []
    assert slam_map.cull_landmarks() == 0
