"""Load a :class:`~svslam.pipeline.PipelineConfig` from a YAML file.

The YAML files under ``config/`` are the single readable place where every
threshold in the pipeline is written down.  They are not a second source of
truth: each key maps onto a field of one of the frozen dataclasses in
:mod:`svslam`, and any key that does not is an error rather than a silent
no-op -- a typo in a config file that quietly changes nothing is worse than a
crash.

PyYAML is optional.  Without it, a small parser handles the subset these files
use (two levels of ``key: value`` mappings with scalar values), which keeps the
package's hard dependency list at numpy and OpenCV.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .backend.ba import BAConfig
from .backend.posegraph import PoseGraphConfig
from .frontend.features import FeatureConfig
from .frontend.odometry import KeyframePolicy, OdometryConfig
from .frontend.stereo import StereoConfig
from .loop.detector import LoopConfig
from .pipeline import PipelineConfig

try:  # pragma: no cover - exercised only when PyYAML is installed
    import yaml

    _HAVE_YAML = True
except ImportError:  # pragma: no cover
    yaml = None
    _HAVE_YAML = False

__all__ = ["load_yaml", "parse_config", "load_config", "have_yaml"]


def have_yaml() -> bool:
    """True if PyYAML is available; otherwise the built-in parser is used."""
    return _HAVE_YAML


def _coerce(token: str) -> Any:
    token = token.strip()
    if token in {"true", "True", "yes"}:
        return True
    if token in {"false", "False", "no"}:
        return False
    if token in {"null", "~", ""}:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token.strip("'\"")


def _minimal_parse(text: str) -> dict[str, Any]:
    """Parse the two-level ``key: value`` subset these config files use."""
    root: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1].isspace()
        key, _, value = line.strip().partition(":")
        key = key.strip()
        if not indented:
            if value.strip():
                root[key] = _coerce(value)
                section = None
            else:
                section = {}
                root[key] = section
        elif section is not None:
            section[key] = _coerce(value)
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into nested dictionaries."""
    text = Path(path).read_text(encoding="utf-8")
    if _HAVE_YAML:
        return yaml.safe_load(text) or {}
    return _minimal_parse(text)


def _apply(instance, values: dict[str, Any], section: str, renames: dict[str, str] | None = None):
    """Return a copy of a frozen dataclass with ``values`` applied.

    Unknown keys raise, so a misspelt threshold is caught at load time rather
    than being silently ignored for the length of a benchmark run.
    """
    renames = renames or {}
    known = {f.name for f in fields(instance)}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        name = renames.get(key, key)
        if name is None:
            continue
        if name not in known:
            raise ValueError(f"unknown key '{section}.{key}' in configuration")
        updates[name] = value
    return replace(instance, **updates)


def parse_config(data: dict[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from parsed YAML."""
    config = PipelineConfig()

    if "features" in data:
        config.feature = _apply(FeatureConfig(), data["features"], "features")
    if "stereo" in data:
        config.stereo = _apply(StereoConfig(), data["stereo"], "stereo")
    if "odometry" in data:
        config.odometry = _apply(OdometryConfig(), data["odometry"], "odometry")
    if "keyframes" in data:
        config.keyframe = _apply(KeyframePolicy(), data["keyframes"], "keyframes")

    ba = dict(data.get("bundle_adjustment", {}))
    config.enable_ba = bool(ba.pop("enabled", config.enable_ba))
    config.local_ba_window = int(ba.pop("window", config.local_ba_window))
    config.covisibility_threshold = int(
        ba.pop("covisibility_threshold", config.covisibility_threshold)
    )
    if ba:
        config.ba = _apply(BAConfig(), ba, "bundle_adjustment")

    loop = dict(data.get("loop_closure", {}))
    config.enable_loop = bool(loop.pop("enabled", config.enable_loop))
    if loop:
        config.loop = _apply(LoopConfig(), loop, "loop_closure")

    graph = dict(data.get("pose_graph", {}))
    config.odometry_information = float(
        graph.pop("odometry_information", config.odometry_information)
    )
    config.loop_information = float(graph.pop("loop_information", config.loop_information))
    if graph:
        config.posegraph = _apply(PoseGraphConfig(), graph, "pose_graph")

    return config


def load_config(path: str | Path) -> PipelineConfig:
    """Read a YAML file and build the pipeline configuration it describes."""
    return parse_config(load_yaml(path))
