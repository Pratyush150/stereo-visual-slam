"""Stereo visual SLAM evaluated on KITTI.

A from-scratch stereo visual odometry and SLAM pipeline: bucketed feature
extraction, sparse stereo, RANSAC PnP with motion-only Gauss-Newton refinement,
windowed bundle adjustment with the Schur complement, bag-of-words loop closure
with geometric verification, and SE(3) pose-graph optimisation -- scored with
the official KITTI odometry metrics.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
