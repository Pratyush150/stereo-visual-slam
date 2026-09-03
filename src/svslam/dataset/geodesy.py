"""Geodetic conversions for KITTI ground truth.

KITTI raw drives ship ground truth as OXTS records: WGS-84 latitude, longitude
and altitude plus roll/pitch/yaw from a coupled GPS/IMU unit.  To compare a
visual odometry trajectory against it you need a *local metric* frame, and the
choice of projection is not cosmetic -- a naive conversion introduces a
systematic scale error that shows up directly in the KITTI translation
percentage.

The KITTI development kit uses a spherical Mercator projection anchored at the
first pose's latitude.  We reproduce it exactly (so our numbers are comparable
with anything else evaluated on KITTI) and additionally provide a proper WGS-84
ellipsoidal ENU conversion, because the two disagree and it is worth being able
to show by how much.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EARTH_RADIUS_M",
    "WGS84_A",
    "WGS84_F",
    "mercator_scale",
    "mercator_xy",
    "geodetic_to_ecef",
    "ecef_to_enu",
    "geodetic_to_enu",
    "EnuOrigin",
]

# Value used by the official KITTI raw-data development kit.
EARTH_RADIUS_M = 6378137.0

# WGS-84 ellipsoid.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def mercator_scale(lat_deg: float) -> float:
    """Mercator scale factor ``cos(lat)`` at the projection anchor latitude.

    Mercator stretches distances by ``1/cos(lat)``; anchoring the scale at the
    first pose and dividing it back out keeps metres metric near the origin.
    Skipping this step is a classic source of a few percent of pure scale error
    at KITTI's latitude of about 49 degrees, where ``cos(lat) ~ 0.656``.
    """
    return float(np.cos(np.deg2rad(lat_deg)))


def mercator_xy(lat_deg: np.ndarray, lon_deg: np.ndarray, scale: float) -> np.ndarray:
    """Spherical Mercator projection, KITTI development-kit convention.

    Parameters
    ----------
    lat_deg, lon_deg:
        Latitude and longitude in degrees (scalars or arrays).
    scale:
        Anchor scale from :func:`mercator_scale`.

    Returns
    -------
    ``(..., 2)`` array of ``[x_east, y_north]`` in metres.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    x = scale * np.deg2rad(lon) * EARTH_RADIUS_M
    y = scale * EARTH_RADIUS_M * np.log(np.tan(np.deg2rad(45.0 + lat / 2.0)))
    return np.stack([x, y], axis=-1)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """WGS-84 geodetic coordinates to Earth-centred, Earth-fixed metres."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + alt) * sin_lat
    return np.stack([x, y, z], axis=-1)


@dataclass(frozen=True)
class EnuOrigin:
    """Anchor for a local east/north/up frame."""

    lat_deg: float
    lon_deg: float
    alt_m: float

    @property
    def ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)

    @property
    def rotation(self) -> np.ndarray:
        """Rotation taking an ECEF displacement into east/north/up."""
        lat = np.deg2rad(self.lat_deg)
        lon = np.deg2rad(self.lon_deg)
        sl, cl = np.sin(lat), np.cos(lat)
        so, co = np.sin(lon), np.cos(lon)
        return np.array(
            [
                [-so, co, 0.0],
                [-sl * co, -sl * so, cl],
                [cl * co, cl * so, sl],
            ]
        )


def ecef_to_enu(ecef: np.ndarray, origin: EnuOrigin) -> np.ndarray:
    """Rotate an ECEF position into the local ENU frame of ``origin``."""
    delta = np.asarray(ecef, dtype=float) - origin.ecef
    return delta @ origin.rotation.T


def geodetic_to_enu(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray, origin: EnuOrigin
) -> np.ndarray:
    """WGS-84 geodetic coordinates straight to local ENU metres."""
    return ecef_to_enu(geodetic_to_ecef(lat_deg, lon_deg, alt_m), origin)
