#!/usr/bin/env python3
"""End-to-end benchmark: run the pipeline on a KITTI sequence and score it.

Produces the numbers quoted in the README and the figures in
``benchmarks/output/``.  Everything it prints is measured on the machine it runs
on; nothing is cached or hard-coded.

Usage::

    python3 benchmarks/run.py --sequence /data/2011_09_30/2011_09_30_drive_0027_sync
    python3 benchmarks/run.py --sequence /data/sequences/07 --frames 500
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svslam.dataset.kitti import open_sequence  # noqa: E402
from svslam.evaluation.kitti_metrics import evaluate  # noqa: E402
from svslam.config import load_config  # noqa: E402
from svslam.pipeline import PipelineConfig, StereoSlam  # noqa: E402


def machine_description() -> str:
    """A short, factual description of the machine the benchmark ran on."""
    cores = "unknown"
    model = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            text = handle.read()
        cores = str(text.count("processor\t:"))
        for line in text.splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {cores} logical cores, Python {platform.python_version()}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="raw drive directory or odometry sequence directory")
    parser.add_argument("--calib", default=None, help="calibration directory (raw layout only)")
    parser.add_argument("--config", default=None,
                        help="YAML configuration file, e.g. config/kitti_raw.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="0 means the whole sequence")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--no-loop", action="store_true", help="disable loop closure")
    parser.add_argument("--no-ba", action="store_true", help="disable local bundle adjustment")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--vocab-start", type=int, default=200)
    parser.add_argument("--vocab-stop", type=int, default=800)
    parser.add_argument("--vocab-step", type=int, default=10)
    parser.add_argument("--progress", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    kwargs = {"calib_dir": args.calib} if args.calib else {}
    dataset = open_sequence(args.sequence, **kwargs)
    stop = len(dataset) if args.frames <= 0 else min(args.start + args.frames, len(dataset))
    print(f"sequence      : {args.sequence}")
    print(f"frames        : {args.start}..{stop} of {len(dataset)}")
    print(f"baseline      : {dataset.calibration.baseline:.4f} m")
    print(f"focal length  : {dataset.calibration.fx:.2f} px")
    print(f"machine       : {machine_description()}")

    config = load_config(args.config) if args.config else PipelineConfig()
    if args.no_loop:
        config.enable_loop = False
    if args.no_ba:
        config.enable_ba = False
    slam = StereoSlam(dataset.calibration, config)

    if config.enable_loop:
        # Held-out slice: the vocabulary is trained on a strided subset from the
        # middle of the drive, which excludes the revisited stretch where the
        # loop closure actually fires.
        vocab_frames = list(range(args.vocab_start, min(args.vocab_stop, len(dataset)), args.vocab_step))
        if len(vocab_frames) >= 10:
            print(f"vocabulary    : training on {len(vocab_frames)} held-out frames "
                  f"({args.vocab_start}..{args.vocab_stop} step {args.vocab_step})")
            t0 = time.perf_counter()
            slam.train_vocabulary(dataset, vocab_frames)
            print(f"                trained in {time.perf_counter() - t0:.1f} s")
        else:
            print("vocabulary    : sequence too short, loop closure disabled")
            config.enable_loop = False

    print("running pipeline ...", flush=True)
    t0 = time.perf_counter()
    result = slam.run(dataset, args.start, stop, args.step, progress=args.progress)
    runtime = time.perf_counter() - t0

    ground_truth = dataset.ground_truth()
    report = None
    report_before = None
    if ground_truth is not None:
        gt = ground_truth[result.frame_indices]
        report = evaluate(result.poses, gt)
        if result.poses_before_loop is not None:
            report_before = evaluate(result.poses_before_loop, gt)

    n_frames = len(result.frame_indices)
    summary = {
        "sequence": str(args.sequence),
        "machine": machine_description(),
        "frames": n_frames,
        "keyframes": int(result.stats["keyframes"]),
        "landmarks": int(result.stats["landmarks"]),
        "mean_stereo_points_per_frame": round(result.stats["mean_stereo_points"], 1),
        "mean_tracked_features": round(result.stats["mean_tracked_features"], 1),
        "mean_feature_spread_entropy": round(result.stats["mean_feature_spread"], 3),
        "loop_closures_accepted": int(result.stats["loop_closures_accepted"]),
        "loop_gate_counters": result.rejected_loops,
        "rejected_implausible_motions": int(result.stats["rejected_implausible_motions"]),
        "runtime_s": round(runtime, 1),
        "seconds_per_frame": round(runtime / max(n_frames, 1), 3),
        "stage_seconds": {k: round(v, 2) for k, v in result.timings.items()},
        "stage_ms_per_frame": {
            k: round(1000.0 * v / max(n_frames, 1), 1) for k, v in result.timings.items()
        },
    }
    if report is not None:
        summary["metrics"] = report.as_dict()
        summary["estimated_path_length_m"] = round(
            float(np.linalg.norm(np.diff(result.poses[:, :3, 3], axis=0), axis=1).sum()), 2
        )
    if report_before is not None:
        summary["metrics_before_loop_closure"] = report_before.as_dict()

    # Loop-closure precision against ground truth, when ground truth exists.
    if ground_truth is not None and result.loop_closures:
        gt_all = ground_truth
        kf_frames = {
            kf.id: kf.frame_index for kf in result.slam_map.keyframes.values()
        }
        correct = 0
        distances = []
        for closure in result.loop_closures:
            a = kf_frames.get(closure.query_id)
            b = kf_frames.get(closure.candidate_id)
            if a is None or b is None:
                continue
            d = float(np.linalg.norm(gt_all[a][:3, 3] - gt_all[b][:3, 3]))
            distances.append(round(d, 2))
            if d < 15.0:
                correct += 1
        summary["loop_closure_gt_distances_m"] = distances
        summary["loop_closure_precision"] = round(correct / max(len(distances), 1), 3)

    with open(output / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savetxt(
        output / "trajectory.txt",
        result.poses[:, :3, :4].reshape(-1, 12),
        fmt="%.9e",
    )
    if ground_truth is not None:
        np.savetxt(
            output / "ground_truth.txt",
            ground_truth[result.frame_indices][:, :3, :4].reshape(-1, 12),
            fmt="%.9e",
        )

    print()
    print("=" * 68)
    print(f"frames processed        : {n_frames}")
    print(f"keyframes               : {summary['keyframes']}")
    print(f"landmarks               : {summary['landmarks']}")
    print(f"mean stereo points/frame: {summary['mean_stereo_points_per_frame']}")
    print(f"mean tracked features   : {summary['mean_tracked_features']}")
    print(f"loop closures accepted  : {summary['loop_closures_accepted']}")
    print(f"loop gate counters      : {summary['loop_gate_counters']}")
    print(f"implausible motions cut : {summary['rejected_implausible_motions']}")
    if report is not None:
        print(f"translation error       : {report.translation_percent:.3f} %")
        print(f"rotation error          : {report.rotation_deg_per_m:.5f} deg/m")
        print(f"ATE RMSE                : {report.ate['rmse']:.3f} m")
        print(f"RPE (10 frames) trans   : {report.rpe['translation_rmse']:.4f} m")
        print(f"ground-truth path       : {report.path_length:.1f} m")
        print(f"estimated path          : {summary['estimated_path_length_m']} m")
        print()
        print(report.format_table())
    print()
    print(f"runtime                 : {runtime:.1f} s  ({summary['seconds_per_frame']} s/frame)")
    for stage, ms in summary["stage_ms_per_frame"].items():
        print(f"  {stage:<20}: {ms:>8.1f} ms/frame")
    print("=" * 68)

    if not args.no_figures:
        _figures(dataset, result, report, report_before, output, config)
    print(f"\nwrote {output}")
    return 0


def _figures(dataset, result, report, report_before, output: Path, config) -> None:
    """Render every figure from the run that just happened."""
    from svslam import viz
    from svslam.backend.ba import BAProblem
    from svslam.frontend.features import OrbDetector
    from svslam.frontend.odometry import estimate_pose_pnp
    from svslam.frontend.stereo import match_stereo_epipolar

    ground_truth = dataset.ground_truth()
    gt = ground_truth[result.frame_indices] if ground_truth is not None else None

    subtitle = ""
    if report is not None:
        subtitle = (
            f"{len(result.frame_indices)} frames, {report.path_length:.0f} m  -  "
            f"{report.translation_percent:.2f}% translation, "
            f"{report.rotation_deg_per_m:.4f} deg/m, ATE {report.ate['rmse']:.2f} m"
        )
    viz.plot_trajectory(result.poses, gt, output / "trajectory.png", subtitle=subtitle)

    if report is not None:
        viz.plot_kitti_errors(report, output / "kitti_error_vs_length.png")

    if result.poses_before_loop is not None and gt is not None:
        viz.plot_loop_comparison(
            result.poses_before_loop, result.poses, gt,
            output / "loop_closure.png",
            ate_before=report_before.ate["rmse"] if report_before else None,
            ate_after=report.ate["rmse"] if report else None,
        )

    if result.slam_map.n_keyframes > 2:
        viz.plot_covisibility(result.slam_map, output / "covisibility.png",
                              config.covisibility_threshold, result.loop_closures)

    # A real frame-to-frame track, with the rejections RANSAC actually made.
    # Matching a frame against itself would show a hundred percent inliers and
    # prove nothing; the correspondences here are between two different frames.
    from svslam.frontend.features import match_descriptors

    frames = result.frame_indices
    first = frames[min(len(frames) - 1, 40)]
    second = frames[min(len(frames) - 1, 42)]
    if second > first:
        left_a, right_a = dataset.load_stereo(first)
        left_b = dataset.load_left(second)
        detector = OrbDetector(config.feature)
        features_a = detector.detect(left_a)
        features_b = detector.detect(left_b)
        stereo = match_stereo_epipolar(
            left_a, right_a, features_a.points, config.stereo,
            fx=dataset.calibration.fx, fy=dataset.calibration.fy,
            cx=dataset.calibration.cx, cy=dataset.calibration.cy,
            baseline=dataset.calibration.baseline,
        )
        depth_of = {int(i): k for k, i in enumerate(stereo.index)}
        matches = match_descriptors(
            features_b.descriptors, features_a.descriptors,
            config.feature.ratio_test, config.feature.max_hamming,
        )
        usable = np.array([int(j) in depth_of for _, j in matches], dtype=bool)
        matches = matches[usable] if matches.size else matches
        if matches.shape[0] > 30:
            rows = np.array([depth_of[int(j)] for _, j in matches])
            points_cam = stereo.points_cam[rows]
            pixels_now = features_b.points[matches[:, 0]]
            pixels_then = stereo.uv_left[rows]
            estimate = estimate_pose_pnp(
                points_cam, pixels_now, dataset.calibration.K, config.odometry
            )
            viz.plot_tracked_features(
                left_b, pixels_now, estimate.inliers,
                output / "tracked_features.png",
                title=f"Frame {first} tracked into frame {second}",
                subtitle=(
                    f"{matches.shape[0]} stereo-triangulated correspondences, "
                    f"{int(estimate.inliers.sum())} kept by RANSAC PnP, "
                    f"{int((~estimate.inliers).sum())} rejected"
                ),
                previous_points=pixels_then,
            )

    # BA sparsity, taken from a real window of the map that was just built.
    slam_map = result.slam_map
    ids = slam_map.keyframe_ids()
    if len(ids) > config.local_ba_window:
        window = slam_map.local_window(ids[len(ids) // 2], config.local_ba_window,
                                       config.covisibility_threshold)
        slot = {k: i for i, k in enumerate(window)}
        lm_slot: dict[int, int] = {}
        ci, pi, obs = [], [], []
        for kf_id in window:
            kf = slam_map.keyframes[kf_id]
            for feature_index, lm_id in kf.observations.items():
                if lm_id not in slam_map.landmarks or feature_index not in kf.stereo_u_right:
                    continue
                lm_slot.setdefault(lm_id, len(lm_slot))
                uv = kf.keypoints[feature_index]
                ci.append(slot[kf_id])
                pi.append(lm_slot[lm_id])
                obs.append([uv[0], uv[1], kf.stereo_u_right[feature_index]])
        if obs:
            problem = BAProblem(
                np.array([slam_map.keyframes[k].T_cw for k in window]),
                np.array([slam_map.landmarks[i].position for i in lm_slot]),
                np.array(ci), np.array(pi), np.array(obs),
                dataset.calibration.fx, dataset.calibration.fy,
                dataset.calibration.cx, dataset.calibration.cy,
                baseline=dataset.calibration.baseline,
            )
            viz.plot_ba_sparsity(problem, output / "ba_sparsity.png")


if __name__ == "__main__":
    raise SystemExit(main())
