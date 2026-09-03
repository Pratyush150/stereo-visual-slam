#!/usr/bin/env python3
"""Download and verify the KITTI data this repository is evaluated on.

The dataset is deliberately **not** committed.  A raw drive is several gigabytes
of someone else's copyrighted imagery; what belongs in the repository is the
exact recipe for obtaining it.

Default target is raw drive ``2011_09_30_drive_0027``, which is the raw-data
equivalent of odometry sequence 07 and contains a genuine revisit -- the vehicle
returns to within about ten metres of its start, so loop closure has something
real to find.

Usage::

    python3 tools/fetch_kitti.py --output /data/kitti
    python3 tools/fetch_kitti.py --output /data/kitti --drive 2011_09_26_drive_0095
    python3 tools/fetch_kitti.py --output /data/kitti --list
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


@dataclass(frozen=True)
class Drive:
    """One raw drive plus the calibration archive it needs."""

    name: str
    date: str
    #: Compressed size in bytes, as served, or ``None`` when not yet recorded.
    sync_bytes: int | None
    note: str

    @property
    def sync_url(self) -> str:
        return f"{BASE_URL}/{self.name}/{self.name}_sync.zip"

    @property
    def calib_url(self) -> str:
        return f"{BASE_URL}/{self.date}_calib.zip"


#: Sizes here were measured by downloading the archives, not copied from a page.
DRIVES: dict[str, Drive] = {
    "2011_09_30_drive_0027": Drive(
        name="2011_09_30_drive_0027",
        date="2011_09_30",
        sync_bytes=4_424_930_450,
        note="raw equivalent of odometry sequence 07; 1106 frames, ~695 m, contains a loop",
    ),
    "2011_09_26_drive_0095": Drive(
        name="2011_09_26_drive_0095",
        date="2011_09_26",
        sync_bytes=1_097_371_915,
        note="smaller alternative (~1.0 GB); short urban drive, no loop",
    ),
}

#: Calibration archive sizes, likewise measured.
CALIB_BYTES: dict[str, int] = {"2011_09_30": 4073, "2011_09_26": 4068}


def _human(n: int | None) -> str:
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def download(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    """Download ``url`` to ``destination``, resuming nothing but verifying size.

    A partial download is the single most common failure here -- a 4 GB transfer
    that stops at 3.1 GB leaves a file that looks plausible and unzips to a
    truncated drive.  The size check plus the CRC test in :func:`verify_zip`
    catches that before any of it reaches the pipeline.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        expected_bytes is None or destination.stat().st_size == expected_bytes
    ):
        print(f"  already present: {destination.name} ({_human(destination.stat().st_size)})")
        return destination

    print(f"  downloading {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(temporary, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            copied += len(block)
            if total:
                pct = 100.0 * copied / total
                print(f"\r    {_human(copied)} / {_human(total)}  ({pct:5.1f}%)",
                      end="", flush=True)
    print()
    temporary.replace(destination)

    size = destination.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(
            f"{destination.name}: expected {expected_bytes} bytes, got {size}. "
            "The download is incomplete or the archive changed upstream."
        )
    return destination


def verify_zip(path: Path) -> None:
    """CRC-test every member, the same check ``unzip -t`` performs."""
    print(f"  verifying {path.name} ...", end="", flush=True)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"{path.name}: corrupt member {bad}")
    print(" ok")


def sha256(path: Path, limit: int | None = None) -> str:
    """SHA-256 of a file, optionally of only its first ``limit`` bytes."""
    digest = hashlib.sha256()
    read = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            if limit is not None and read + len(block) > limit:
                block = block[: limit - read]
            digest.update(block)
            read += len(block)
            if limit is not None and read >= limit:
                break
    return digest.hexdigest()


def extract(path: Path, output: Path, members: tuple[str, ...] = ()) -> None:
    """Extract an archive, optionally only members containing one of ``members``.

    The velodyne scans and colour cameras are most of a raw drive's bulk and this
    pipeline uses neither, so by default only the greyscale stereo pair and the
    OXTS records come out.
    """
    print(f"  extracting {path.name} -> {output}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if members:
            names = [n for n in names if any(m in n for m in members)]
        for name in names:
            archive.extract(name, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/kitti", help="where to download and extract")
    parser.add_argument("--drive", default="2011_09_30_drive_0027", choices=sorted(DRIVES))
    parser.add_argument("--list", action="store_true", help="list known drives and exit")
    parser.add_argument("--keep-archives", action="store_true",
                        help="do not delete the zip files after extraction")
    parser.add_argument("--all-sensors", action="store_true",
                        help="extract velodyne and colour cameras too (much larger)")
    parser.add_argument("--verify-only", action="store_true",
                        help="only CRC-check archives already present")
    args = parser.parse_args(argv)

    if args.list:
        for drive in DRIVES.values():
            print(f"{drive.name}  ({_human(drive.sync_bytes)})")
            print(f"    {drive.note}")
            print(f"    sync : {drive.sync_url}")
            print(f"    calib: {drive.calib_url}")
        return 0

    drive = DRIVES[args.drive]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    calib_zip = output / f"{drive.date}_calib.zip"
    sync_zip = output / f"{drive.name}_sync.zip"

    if args.verify_only:
        for path in (calib_zip, sync_zip):
            if path.exists():
                verify_zip(path)
            else:
                print(f"  missing: {path}")
        return 0

    print(f"drive: {drive.name}  -- {drive.note}")
    download(drive.calib_url, calib_zip, CALIB_BYTES.get(drive.date))
    verify_zip(calib_zip)
    download(drive.sync_url, sync_zip, drive.sync_bytes)
    verify_zip(sync_zip)

    members = () if args.all_sensors else ("image_00/", "image_01/", "oxts/")
    extract(calib_zip, output)
    extract(sync_zip, output, members)

    # The readers expect the calibration files beside the drive directory.
    calib_dir = output / drive.date
    for name in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt", "calib_imu_to_velo.txt"):
        source = calib_dir / name
        if not source.exists():
            print(f"  warning: {name} not found after extraction")

    if not args.keep_archives:
        for path in (sync_zip,):
            print(f"  removing {path.name}")
            path.unlink(missing_ok=True)

    drive_dir = calib_dir / f"{drive.name}_sync"
    print()
    print("done. run the pipeline with:")
    print(f"  python3 benchmarks/run.py --sequence {drive_dir}")
    if not drive_dir.exists():
        print(f"  (expected {drive_dir} to exist -- check the extraction above)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
