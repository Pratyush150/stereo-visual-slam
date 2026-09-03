"""Shared fixtures.

Tests that need the KITTI download are skipped, never failed, when it is
absent.  Point ``SVSLAM_KITTI`` at a raw drive or odometry sequence directory to
enable them::

    SVSLAM_KITTI=/data/kitti/2011_09_30/2011_09_30_drive_0027_sync pytest -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KITTI_ENV = "SVSLAM_KITTI"


def kitti_path() -> Path | None:
    """The configured KITTI sequence directory, if it exists."""
    raw = os.environ.get(KITTI_ENV)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


requires_kitti = pytest.mark.skipif(
    kitti_path() is None,
    reason=f"set {KITTI_ENV} to a KITTI sequence directory to run this test",
)

try:  # pragma: no cover
    import cv2 as _cv2

    HAVE_CV2 = True
except ImportError:  # pragma: no cover
    _cv2 = None
    HAVE_CV2 = False

requires_cv2 = pytest.mark.skipif(not HAVE_CV2, reason="OpenCV is not installed")


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test is deterministic."""
    return np.random.default_rng(20260903)


@pytest.fixture
def pinhole() -> dict[str, float]:
    """KITTI's rectified greyscale intrinsics, rounded."""
    return {"fx": 707.0912, "fy": 707.0912, "cx": 601.8873, "cy": 183.1104,
            "baseline": 0.5371506532679237}


@pytest.fixture
def kitti_sequence():
    """An open KITTI sequence, or a skip."""
    path = kitti_path()
    if path is None:
        pytest.skip(f"set {KITTI_ENV} to run this test")
    from svslam.dataset.kitti import open_sequence

    return open_sequence(path)
