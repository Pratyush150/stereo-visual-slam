"""Readers for the two KITTI layouts.

Two different downloads of KITTI are in common use and they are not
interchangeable:

``raw``
    ``2011_09_30/2011_09_30_drive_0027_sync/`` with ``image_00``..``image_03``,
    ``oxts/`` and ``velodyne_points/``, plus per-day ``calib_*.txt`` files.
    Ground truth has to be *derived* from the OXTS GPS/IMU records.

``odometry``
    ``sequences/07/`` with ``image_0``, ``image_1``, ``calib.txt``,
    ``times.txt`` and a ready-made ``poses/07.txt``.

Both are supported behind one interface so the pipeline does not care which one
you downloaded.  Images are loaded lazily -- a raw drive is several gigabytes
and does not belong in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..se3 import normalise_rotation, se3_inverse
from .geodesy import EnuOrigin, geodetic_to_enu, mercator_scale, mercator_xy

try:  # pragma: no cover - exercised only when OpenCV is present
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

__all__ = [
    "StereoCalibration",
    "parse_calib_file",
    "load_raw_calibration",
    "load_odometry_calibration",
    "OxtsRecord",
    "read_oxts_records",
    "oxts_to_poses",
    "KittiRawDataset",
    "KittiOdometryDataset",
    "open_sequence",
]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StereoCalibration:
    """Rectified stereo intrinsics and the extrinsics needed for ground truth.

    Attributes
    ----------
    P_left, P_right:
        3x4 rectified projection matrices.  In a rectified KITTI pair the
        right-camera matrix carries the baseline in its last column as
        ``-fx * b``.
    T_cam_imu:
        4x4 transform taking a point in IMU coordinates into *rectified left
        camera* coordinates, or ``None`` for the odometry layout where the OXTS
        chain is not shipped.
    """

    P_left: np.ndarray
    P_right: np.ndarray
    image_size: tuple[int, int]
    T_cam_imu: np.ndarray | None = None
    R_rect: np.ndarray | None = None
    T_cam_velo: np.ndarray | None = None

    @property
    def K(self) -> np.ndarray:
        """3x3 rectified intrinsic matrix of the left camera."""
        return np.array(self.P_left[:3, :3], dtype=float)

    @property
    def fx(self) -> float:
        return float(self.P_left[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P_left[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P_left[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P_left[1, 2])

    @property
    def baseline(self) -> float:
        """Stereo baseline in metres, recovered from the projection matrices.

        ``P_right[0, 3] = -fx * baseline`` for a rectified pair, so the baseline
        never has to be hard-coded per sequence.
        """
        return float(-self.P_right[0, 3] / self.P_left[0, 0])

    def to_dict(self) -> dict[str, object]:
        """Plain-Python view, used by the CLI tools and the round-trip test."""
        return {
            "P_left": self.P_left.tolist(),
            "P_right": self.P_right.tolist(),
            "image_size": list(self.image_size),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline": self.baseline,
        }


def parse_calib_file(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Parse a ``key: v1 v2 ...`` KITTI calibration file into float arrays.

    Non-numeric values (``calib_time``) are skipped rather than raising, since
    every KITTI calibration file carries at least one.
    """
    out: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            try:
                out[key.strip()] = np.array(
                    [float(tok) for tok in value.split()], dtype=float
                )
            except ValueError:
                continue
    return out


def _rt_to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 rigid transform, re-orthonormalising the rotation.

    KITTI stores its extrinsic rotations to six decimal places, so the matrices
    on disk are not exactly orthonormal.  Composing three of them and then
    inverting one with a transpose leaves an error around 1e-7 -- small, but it
    means the ground-truth trajectory does not start exactly at the identity,
    which is confusing to debug.  Projecting each rotation back onto SO(3) as it
    is read costs nothing and removes the whole class of problem.
    """
    T = np.eye(4)
    T[:3, :3] = normalise_rotation(np.asarray(R, dtype=float).reshape(3, 3))
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def load_raw_calibration(
    calib_dir: str | os.PathLike[str], *, colour: bool = False
) -> StereoCalibration:
    """Build a :class:`StereoCalibration` from a raw-drive calibration folder.

    ``calib_dir`` holds ``calib_cam_to_cam.txt``, ``calib_velo_to_cam.txt`` and
    ``calib_imu_to_velo.txt``.  With ``colour=False`` the greyscale pair
    (``image_00``/``image_01``) is used; those are the cameras the KITTI
    odometry benchmark is defined on.

    The IMU-to-camera chain is composed as::

        T_cam_imu = R_rect_00 . T_velo_cam . T_imu_velo

    which is what turns an OXTS pose into a camera pose.
    """
    calib_dir = Path(calib_dir)
    cam = parse_calib_file(calib_dir / "calib_cam_to_cam.txt")
    left_idx, right_idx = ("02", "03") if colour else ("00", "01")

    P_left = cam[f"P_rect_{left_idx}"].reshape(3, 4)
    P_right = cam[f"P_rect_{right_idx}"].reshape(3, 4)
    size = cam[f"S_rect_{left_idx}"]
    image_size = (int(round(size[0])), int(round(size[1])))

    R_rect = np.eye(4)
    R_rect[:3, :3] = normalise_rotation(cam["R_rect_00"].reshape(3, 3))

    T_cam_imu = None
    T_cam_velo = None
    velo_path = calib_dir / "calib_velo_to_cam.txt"
    imu_path = calib_dir / "calib_imu_to_velo.txt"
    if velo_path.exists():
        velo = parse_calib_file(velo_path)
        T_velo_cam = _rt_to_homogeneous(velo["R"], velo["T"])
        T_cam_velo = R_rect @ T_velo_cam
        if imu_path.exists():
            imu = parse_calib_file(imu_path)
            T_imu_velo = _rt_to_homogeneous(imu["R"], imu["T"])
            T_cam_imu = T_cam_velo @ T_imu_velo

    return StereoCalibration(
        P_left=P_left,
        P_right=P_right,
        image_size=image_size,
        T_cam_imu=T_cam_imu,
        R_rect=R_rect,
        T_cam_velo=T_cam_velo,
    )


def load_odometry_calibration(
    calib_path: str | os.PathLike[str], image_size: tuple[int, int] = (1226, 370)
) -> StereoCalibration:
    """Build a :class:`StereoCalibration` from an odometry ``calib.txt``."""
    raw = parse_calib_file(calib_path)
    return StereoCalibration(
        P_left=raw["P0"].reshape(3, 4),
        P_right=raw["P1"].reshape(3, 4),
        image_size=image_size,
        T_cam_velo=(
            _rt_to_homogeneous(raw["Tr"].reshape(3, 4)[:, :3], raw["Tr"].reshape(3, 4)[:, 3])
            if "Tr" in raw
            else None
        ),
    )


# --------------------------------------------------------------------------
# OXTS ground truth
# --------------------------------------------------------------------------

# Field order of oxts/dataformat.txt; only the first six are used here.
OXTS_FIELDS = ("lat", "lon", "alt", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class OxtsRecord:
    """One OXTS GPS/IMU sample (the subset the ground truth needs)."""

    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float


def read_oxts_records(oxts_dir: str | os.PathLike[str]) -> list[OxtsRecord]:
    """Read every ``oxts/data/*.txt`` record, in frame order."""
    data_dir = Path(oxts_dir)
    if (data_dir / "data").is_dir():
        data_dir = data_dir / "data"
    records: list[OxtsRecord] = []
    for path in sorted(data_dir.glob("*.txt")):
        values = np.fromstring(path.read_text(), sep=" ")
        if values.size < 6:
            continue
        records.append(OxtsRecord(*(float(v) for v in values[:6])))
    return records


def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """OXTS roll/pitch/yaw to a rotation matrix, ``R = Rz(yaw) Ry(pitch) Rx(roll)``.

    This is the convention the KITTI raw development kit uses; the resulting
    frame is x-forward, y-left, z-up in a local east/north/up world.
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def oxts_to_poses(
    records: Sequence[OxtsRecord],
    *,
    projection: str = "mercator",
    T_cam_imu: np.ndarray | None = None,
) -> np.ndarray:
    """Convert OXTS records to a ground-truth trajectory in a local metric frame.

    Steps, in order:

    1. Project latitude/longitude to local metres.  ``projection="mercator"``
       reproduces the official KITTI development kit (spherical Mercator with
       the scale anchored at the first frame's latitude).  ``projection="enu"``
       uses a proper WGS-84 ellipsoidal east/north/up conversion instead.
       Altitude is taken as the vertical axis in both cases.
    2. Build each IMU pose as ``[Rz(yaw) Ry(pitch) Rx(roll) | t]``.
    3. Left-multiply by the inverse of the first pose, so the trajectory starts
       at the identity -- otherwise every pose carries a multi-thousand-kilometre
       offset and single precision anywhere downstream would be destroyed.
    4. If ``T_cam_imu`` is given, change basis into the rectified left camera::

           T_cam(i) = T_cam_imu . T_imu(0)^-1 . T_imu(i) . T_cam_imu^-1

       This is what makes the result directly comparable to the KITTI odometry
       benchmark, whose poses are those of the left camera in the first left
       camera frame.

    Returns
    -------
    ``(N, 4, 4)`` array of poses mapping camera (or IMU) points into the world.
    """
    if not records:
        return np.zeros((0, 4, 4))

    lat = np.array([r.lat for r in records], dtype=float)
    lon = np.array([r.lon for r in records], dtype=float)
    alt = np.array([r.alt for r in records], dtype=float)

    if projection == "mercator":
        scale = mercator_scale(lat[0])
        xy = mercator_xy(lat, lon, scale)
        translations = np.column_stack([xy[:, 0], xy[:, 1], alt])
    elif projection == "enu":
        origin = EnuOrigin(lat[0], lon[0], alt[0])
        translations = geodetic_to_enu(lat, lon, alt, origin)
    else:
        raise ValueError(f"unknown projection {projection!r}")

    poses = np.zeros((len(records), 4, 4))
    for i, rec in enumerate(records):
        poses[i] = _rt_to_homogeneous(
            _euler_to_rotation(rec.roll, rec.pitch, rec.yaw), translations[i]
        )

    first_inv = se3_inverse(poses[0])
    poses = np.einsum("ij,njk->nik", first_inv, poses)

    if T_cam_imu is not None:
        T = np.asarray(T_cam_imu, dtype=float)
        T_inv = se3_inverse(T)
        poses = np.einsum("ij,njk,kl->nil", T, poses, T_inv)

    for i in range(poses.shape[0]):
        poses[i, :3, :3] = normalise_rotation(poses[i, :3, :3])
    return poses


# --------------------------------------------------------------------------
# Sequence readers
# --------------------------------------------------------------------------


def _load_grey(path: Path) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required to load KITTI images")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image {path}")
    return image


@dataclass
class _SequenceBase:
    """Shared behaviour of the two sequence readers."""

    root: Path
    calibration: StereoCalibration = field(init=False)
    left_paths: list[Path] = field(init=False, default_factory=list)
    right_paths: list[Path] = field(init=False, default_factory=list)
    timestamps: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0))
    _gt: np.ndarray | None = field(init=False, default=None)

    def __len__(self) -> int:
        return len(self.left_paths)

    def load_stereo(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(left, right)`` greyscale images for one frame."""
        return _load_grey(self.left_paths[index]), _load_grey(self.right_paths[index])

    def load_left(self, index: int) -> np.ndarray:
        return _load_grey(self.left_paths[index])

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[
        tuple[int, np.ndarray, np.ndarray]
    ]:
        """Iterate ``(index, left, right)`` over a slice of the sequence."""
        stop = len(self) if stop is None else min(stop, len(self))
        for i in range(start, stop, step):
            left, right = self.load_stereo(i)
            yield i, left, right

    def ground_truth(self) -> np.ndarray | None:
        """``(N, 4, 4)`` camera poses in the first camera frame, or ``None``."""
        return self._gt


class KittiRawDataset(_SequenceBase):
    """Reader for a KITTI *raw* synced drive directory.

    Parameters
    ----------
    drive_dir:
        e.g. ``.../2011_09_30/2011_09_30_drive_0027_sync``.
    calib_dir:
        Folder holding the ``calib_*.txt`` files.  Defaults to the parent of
        ``drive_dir``, which is where the official archives put them.
    colour:
        Use ``image_02``/``image_03`` instead of the greyscale pair.
    projection:
        Passed to :func:`oxts_to_poses`.
    """

    def __init__(
        self,
        drive_dir: str | os.PathLike[str],
        calib_dir: str | os.PathLike[str] | None = None,
        *,
        colour: bool = False,
        projection: str = "mercator",
    ) -> None:
        self.root = Path(drive_dir)
        calib_root = Path(calib_dir) if calib_dir is not None else self.root.parent
        self.calibration = load_raw_calibration(calib_root, colour=colour)

        left_dir, right_dir = ("image_02", "image_03") if colour else ("image_00", "image_01")
        self.left_paths = sorted((self.root / left_dir / "data").glob("*.png"))
        self.right_paths = sorted((self.root / right_dir / "data").glob("*.png"))
        n = min(len(self.left_paths), len(self.right_paths))
        self.left_paths = self.left_paths[:n]
        self.right_paths = self.right_paths[:n]

        self.timestamps = self._read_timestamps(self.root / left_dir / "timestamps.txt", n)

        oxts_dir = self.root / "oxts"
        self._gt = None
        if oxts_dir.is_dir():
            records = read_oxts_records(oxts_dir)[:n]
            if records:
                self._gt = oxts_to_poses(
                    records,
                    projection=projection,
                    T_cam_imu=self.calibration.T_cam_imu,
                )

    @staticmethod
    def _read_timestamps(path: Path, n: int) -> np.ndarray:
        """Read ``timestamps.txt`` as seconds since the first frame."""
        if not path.exists():
            return np.arange(n, dtype=float) * 0.1
        seconds: list[float] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # "2011-09-30 12:34:56.123456789" -- only the time of day matters
            # for relative timing within a drive.
            hms = line.split(" ")[-1]
            h, m, s = hms.split(":")
            seconds.append(int(h) * 3600 + int(m) * 60 + float(s))
        if not seconds:
            return np.arange(n, dtype=float) * 0.1
        arr = np.array(seconds[:n], dtype=float)
        return arr - arr[0]


class KittiOdometryDataset(_SequenceBase):
    """Reader for the KITTI *odometry* benchmark folder layout."""

    def __init__(
        self,
        sequence_dir: str | os.PathLike[str],
        poses_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(sequence_dir)
        self.left_paths = sorted((self.root / "image_0").glob("*.png"))
        self.right_paths = sorted((self.root / "image_1").glob("*.png"))
        n = min(len(self.left_paths), len(self.right_paths))
        self.left_paths = self.left_paths[:n]
        self.right_paths = self.right_paths[:n]

        image_size = (1226, 370)
        if n and cv2 is not None:
            probe = _load_grey(self.left_paths[0])
            image_size = (int(probe.shape[1]), int(probe.shape[0]))
        self.calibration = load_odometry_calibration(self.root / "calib.txt", image_size)

        times_path = self.root / "times.txt"
        self.timestamps = (
            np.fromstring(times_path.read_text(), sep="\n")[:n]
            if times_path.exists()
            else np.arange(n, dtype=float) * 0.1
        )

        if poses_file is None:
            guess = self.root.parent.parent / "poses" / f"{self.root.name}.txt"
            poses_file = guess if guess.exists() else None
        self._gt = None
        if poses_file is not None and Path(poses_file).exists():
            flat = np.loadtxt(poses_file).reshape(-1, 3, 4)[:n]
            poses = np.tile(np.eye(4), (flat.shape[0], 1, 1))
            poses[:, :3, :4] = flat
            self._gt = poses


def open_sequence(path: str | os.PathLike[str], **kwargs: object) -> _SequenceBase:
    """Open either layout, deciding from what is actually on disk."""
    path = Path(path)
    if (path / "image_0").is_dir() and (path / "calib.txt").exists():
        return KittiOdometryDataset(path)
    if (path / "image_00").is_dir():
        return KittiRawDataset(path, **kwargs)  # type: ignore[arg-type]
    raise FileNotFoundError(
        f"{path} looks like neither a KITTI odometry sequence nor a raw drive"
    )
