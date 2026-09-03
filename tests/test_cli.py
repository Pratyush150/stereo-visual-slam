"""The command-line tools: argument parsing and the pose file format.

The tools are executable scripts without a ``.py`` suffix, so they are loaded
here by path rather than imported.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_loader(
        name, importlib.machinery.SourceFileLoader(name, str(path))
    )
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses resolve their own module through
    # sys.modules while the class body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_tool():
    return _load("svslam_run_tool", ROOT / "tools" / "svslam-run")


@pytest.fixture(scope="module")
def eval_tool():
    return _load("svslam_eval_tool", ROOT / "tools" / "svslam-eval")


@pytest.fixture(scope="module")
def fetch_tool():
    return _load("fetch_kitti_tool", ROOT / "tools" / "fetch_kitti.py")


def test_run_tool_defaults(run_tool):
    args = run_tool.parse_args(["--sequence", "/tmp/x"])
    assert args.sequence == "/tmp/x"
    assert args.frames == 0
    assert args.step == 1
    assert not args.no_loop
    assert not args.no_bucketing


def test_run_tool_flags(run_tool):
    args = run_tool.parse_args([
        "--sequence", "/tmp/x", "--frames", "50", "--no-loop", "--no-ba",
        "--window", "9", "--max-features", "800", "--no-bucketing",
    ])
    assert args.frames == 50
    assert args.no_loop and args.no_ba and args.no_bucketing
    assert args.window == 9
    assert args.max_features == 800


def test_eval_tool_requires_a_ground_truth_source(eval_tool):
    with pytest.raises(SystemExit):
        eval_tool.parse_args(["--estimate", "a.txt"])
    args = eval_tool.parse_args(["--estimate", "a.txt", "--ground-truth", "b.txt"])
    assert args.ground_truth == "b.txt"
    assert args.rpe_delta == 10


def test_pose_file_round_trip(eval_tool, tmp_path):
    """The KITTI pose format is twelve numbers per line, row-major 3x4."""
    from svslam.se3 import se3_exp

    poses = np.array([se3_exp(np.array([0.0, 0.0, float(i), 0.0, 0.01 * i, 0.0]))
                      for i in range(7)])
    path = tmp_path / "poses.txt"
    np.savetxt(path, poses[:, :3, :4].reshape(-1, 12), fmt="%.12e")

    loaded = eval_tool.load_poses(path)
    assert loaded.shape == (7, 4, 4)
    assert np.allclose(loaded, poses, atol=1e-11)
    assert np.allclose(loaded[:, 3, :], [0.0, 0.0, 0.0, 1.0])


def test_pose_file_with_wrong_width_is_rejected(eval_tool, tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        eval_tool.load_poses(path)


def test_eval_tool_end_to_end(eval_tool, tmp_path, capsys):
    """Score a trajectory with a known 2% scale error through the CLI entry point."""
    truth = np.tile(np.eye(4), (1000, 1, 1))
    truth[:, 2, 3] = np.arange(1000, dtype=float)
    estimate = truth.copy()
    estimate[:, 2, 3] *= 1.02

    gt_path = tmp_path / "gt.txt"
    est_path = tmp_path / "est.txt"
    json_path = tmp_path / "report.json"
    np.savetxt(gt_path, truth[:, :3, :4].reshape(-1, 12), fmt="%.12e")
    np.savetxt(est_path, estimate[:, :3, :4].reshape(-1, 12), fmt="%.12e")

    code = eval_tool.main([
        "--estimate", str(est_path), "--ground-truth", str(gt_path),
        "--json", str(json_path),
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "translation error  : 2.000 %" in output
    assert json_path.exists()


def test_fetch_tool_knows_its_urls(fetch_tool):
    drive = fetch_tool.DRIVES["2011_09_30_drive_0027"]
    assert drive.sync_url.endswith("2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip")
    assert drive.calib_url.endswith("2011_09_30_calib.zip")
    assert drive.sync_bytes == 4_424_930_450
    assert fetch_tool.CALIB_BYTES["2011_09_30"] == 4073


def test_fetch_tool_lists_without_network(fetch_tool, capsys):
    assert fetch_tool.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "2011_09_30_drive_0027" in output
    assert "avg-kitti" in output


def test_fetch_tool_verify_only_reports_missing_files(fetch_tool, tmp_path, capsys):
    assert fetch_tool.main(["--verify-only", "--output", str(tmp_path)]) == 0
    assert "missing" in capsys.readouterr().out


def test_tools_are_executable():
    for name in ("svslam-run", "svslam-eval", "fetch_kitti.py"):
        path = ROOT / "tools" / name
        assert path.exists()
        assert path.stat().st_mode & 0o111, f"{name} is not executable"


def test_benchmark_runner_parses_arguments():
    module = _load("benchmark_runner", ROOT / "benchmarks" / "run.py")
    args = module.parse_args(["--sequence", "/tmp/x", "--frames", "10", "--no-figures"])
    assert args.frames == 10
    assert args.no_figures
    assert "cores" in module.machine_description()
