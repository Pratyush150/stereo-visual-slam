"""Loading the YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest

from svslam.config import _minimal_parse, have_yaml, load_config, load_yaml, parse_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize("name", ["kitti_raw.yaml", "kitti_odometry.yaml"])
def test_shipped_configs_load(name):
    config = load_config(CONFIG_DIR / name)
    assert config.feature.max_features > 0
    assert config.stereo.max_disparity > config.stereo.min_disparity
    assert config.local_ba_window >= 2
    assert config.loop.vocabulary_size >= 100


@pytest.mark.parametrize("name", ["kitti_raw.yaml", "kitti_odometry.yaml"])
def test_shipped_configs_match_the_dataclass_defaults(name):
    """The files document the defaults; drift between them is a bug."""
    from svslam.pipeline import PipelineConfig

    loaded = load_config(CONFIG_DIR / name)
    default = PipelineConfig()
    assert loaded.feature == default.feature
    assert loaded.stereo == default.stereo
    assert loaded.odometry == default.odometry
    assert loaded.keyframe == default.keyframe
    assert loaded.loop == default.loop
    assert loaded.local_ba_window == default.local_ba_window
    assert loaded.covisibility_threshold == default.covisibility_threshold
    assert loaded.enable_ba == default.enable_ba
    assert loaded.enable_loop == default.enable_loop


def test_unknown_key_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="stereo.not_a_real_key"):
        parse_config({"stereo": {"not_a_real_key": 1}})


def test_overrides_are_applied():
    config = parse_config({
        "features": {"max_features": 250, "use_bucketing": False},
        "bundle_adjustment": {"enabled": False, "window": 11},
        "loop_closure": {"enabled": False, "min_inliers": 44},
        "pose_graph": {"kernel": "huber", "loop_information": 25.0},
    })
    assert config.feature.max_features == 250
    assert not config.feature.use_bucketing
    assert not config.enable_ba
    assert config.local_ba_window == 11
    assert not config.enable_loop
    assert config.loop.min_inliers == 44
    assert config.posegraph.kernel == "huber"
    assert config.loop_information == 25.0


def test_empty_config_gives_the_defaults():
    from svslam.pipeline import PipelineConfig

    assert parse_config({}).feature == PipelineConfig().feature


def test_minimal_parser_handles_the_shipped_files():
    """The fallback parser must read the same files PyYAML does."""
    for name in ("kitti_raw.yaml", "kitti_odometry.yaml"):
        text = (CONFIG_DIR / name).read_text(encoding="utf-8")
        parsed = _minimal_parse(text)
        assert parsed["features"]["max_features"] == 1500
        assert parsed["stereo"]["patch_radius"] == 5
        assert parsed["bundle_adjustment"]["enabled"] is True
        assert parsed["pose_graph"]["kernel"] == "dcs"
        assert parse_config(parsed).feature.max_features == 1500


def test_minimal_parser_coerces_scalars():
    parsed = _minimal_parse(
        "section:\n"
        "  an_int: 12\n"
        "  a_float: 1.0e-4\n"
        "  a_true: true\n"
        "  a_false: false\n"
        "  a_string: dcs   # with a comment\n"
        "  a_null: null\n"
    )
    section = parsed["section"]
    assert section["an_int"] == 12 and isinstance(section["an_int"], int)
    assert section["a_float"] == 1e-4
    assert section["a_true"] is True and section["a_false"] is False
    assert section["a_string"] == "dcs"
    assert section["a_null"] is None


def test_load_yaml_and_have_yaml_agree():
    data = load_yaml(CONFIG_DIR / "kitti_raw.yaml")
    assert isinstance(have_yaml(), bool)
    assert data["dataset"]["layout"] == "raw"
    assert data["features"]["grid_cols"] == 12
