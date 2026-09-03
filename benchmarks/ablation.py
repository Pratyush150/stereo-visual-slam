#!/usr/bin/env python3
"""Measure what each part of the pipeline is worth.

Runs the same slice of a sequence several times, each with one component
disabled, and reports the KITTI metrics for each. The numbers in the README's
ablation table come from here.

An ablation is the only honest way to justify a design decision. "Bucketing
spreads features better" is an assertion; "turning bucketing off changes
translation error from X to Y on this sequence" is a measurement, and it is
also how you find out that something you were sure mattered does not.

Usage::

    python3 benchmarks/ablation.py --sequence /data/2011_09_30/..._sync --frames 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svslam.dataset.kitti import open_sequence  # noqa: E402
from svslam.evaluation.kitti_metrics import evaluate  # noqa: E402
from svslam.pipeline import PipelineConfig, StereoSlam  # noqa: E402


def _full(config: PipelineConfig) -> None:
    """The baseline: everything on."""


def _no_bundle_adjustment(config: PipelineConfig) -> None:
    config.enable_ba = False


def _no_bucketing(config: PipelineConfig) -> None:
    config.feature = replace(config.feature, use_bucketing=False)


def _no_depth_gate(config: PipelineConfig) -> None:
    """Accept every stereo match, however uncertain its depth."""
    config.stereo = replace(
        config.stereo, max_relative_depth_sigma=1.0, min_disparity=1.0, max_depth=250.0
    )


def _no_motion_gate(config: PipelineConfig) -> None:
    """Believe every pose the solver returns, however impossible."""
    config.odometry = replace(
        config.odometry,
        max_translation_per_frame=1e9,
        max_rotation_per_frame=1e9,
    )


def _no_association_gate(config: PipelineConfig) -> None:
    """Link every descriptor match to a landmark, without a geometric check."""
    config.odometry = replace(config.odometry, final_inlier_threshold=1e9)


ABLATIONS = (
    ("full pipeline", _full),
    ("no bundle adjustment", _no_bundle_adjustment),
    ("no feature bucketing", _no_bucketing),
    ("no depth-uncertainty gate", _no_depth_gate),
    ("no motion plausibility gate", _no_motion_gate),
    ("no landmark association gate", _no_association_gate),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--calib", default=None)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the table as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kwargs = {"calib_dir": args.calib} if args.calib else {}
    dataset = open_sequence(args.sequence, **kwargs)
    stop = min(args.start + args.frames, len(dataset))
    ground_truth = dataset.ground_truth()
    if ground_truth is None:
        print("this sequence has no ground truth to score against", file=sys.stderr)
        return 2

    rows = []
    for name, mutate in ABLATIONS:
        config = PipelineConfig()
        # Loop closure needs a revisit, which a short slice does not contain, so
        # it is off for every row: the comparison is of the odometry front and
        # back end, held equal in every other respect.
        config.enable_loop = False
        mutate(config)

        started = time.perf_counter()
        result = StereoSlam(dataset.calibration, config).run(dataset, args.start, stop)
        runtime = time.perf_counter() - started

        report = evaluate(result.poses, ground_truth[result.frame_indices])
        rows.append({
            "ablation": name,
            "frames": len(result.frame_indices),
            "translation_percent": round(report.translation_percent, 3),
            "rotation_deg_per_m": round(report.rotation_deg_per_m, 5),
            "ate_rmse_m": round(report.ate["rmse"], 3),
            "keyframes": int(result.stats["keyframes"]),
            "mean_tracked": round(result.stats["mean_tracked_features"], 1),
            "feature_spread": round(result.stats["mean_feature_spread"], 3),
            "seconds_per_frame": round(runtime / max(len(result.frame_indices), 1), 3),
        })
        print(f"  {name:<32} {rows[-1]['translation_percent']:>7.3f} %"
              f"  {rows[-1]['rotation_deg_per_m']:>8.5f} deg/m"
              f"  ATE {rows[-1]['ate_rmse_m']:>7.3f} m"
              f"  {rows[-1]['seconds_per_frame']:>6.3f} s/frame", flush=True)

    header = (
        "| variant | translation (%) | rotation (deg/m) | ATE RMSE (m) | "
        "keyframes | mean tracked | feature spread | s/frame |"
    )
    print()
    print(f"{args.start}..{stop} of {args.sequence}")
    print(header)
    print("|" + "---|" * 7 + "---|")
    for row in rows:
        print(f"| {row['ablation']} | {row['translation_percent']:.2f} | "
              f"{row['rotation_deg_per_m']:.4f} | {row['ate_rmse_m']:.2f} | "
              f"{row['keyframes']} | {row['mean_tracked']:.0f} | "
              f"{row['feature_spread']:.3f} | {row['seconds_per_frame']:.2f} |")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
