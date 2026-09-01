"""Configuration loading and validation without Hydra magic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_paths(value: Any, base: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("campaign configuration must be a mapping")
    project_root = (path.parent / config["paths"]["project_root"]).resolve()
    config["paths"]["project_root"] = str(project_root)
    for key in ("data_root", "current_labels", "queue_csv", "manifest_dir", "run_dir"):
        config["paths"][key] = str((project_root / config["paths"][key]).resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    phases = config["training"]["phases"]
    covered: list[int] = []
    for phase in phases:
        start, end = map(int, phase["epochs"])
        if start > end:
            raise ValueError(f"invalid phase range: {phase['epochs']}")
        total = sum(float(v) for v in phase["sources"].values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"source fractions must sum to 1 in {phase['name']}")
        covered.extend(range(start, end + 1))
    expected = list(range(1, int(config["training"]["epochs"]) + 1))
    if covered != expected:
        raise ValueError("training phases must cover every epoch exactly once")
    if config["targets"].get("taper_axis", False):
        raise ValueError("axis tapering was explicitly rejected")
    first = phases[0]
    if list(map(int, first["epochs"])) != [1, 5]:
        raise ValueError("the long 90% synthetic opening phase must cover epochs 1-5")
    if int(first["pairs_per_epoch"]) != 6000:
        raise ValueError("epochs 1-5 must remain large 6000-pair epochs")
    if abs(float(first["sources"].get("procedural", 0.0)) - 0.90) > 1e-12:
        raise ValueError("epochs 1-5 must use exactly 90% procedural data")
    if config["input"]["causal_channels_t"] != [-2, -1, 0]:
        raise ValueError("X_t causal channels changed")
    if config["input"]["causal_channels_tp1"] != [-1, 0, 1]:
        raise ValueError("X_t+1 causal channels changed")
    if int(config["input"]["sam_input_size"]) != 1008:
        raise ValueError("the pinned standard SAM3 builder uses 1008x1008 inputs")
    procedural = config.get("procedural", {})
    if abs(float(procedural.get("merge_probability", 0.0)) - 0.05) > 1e-12:
        raise ValueError("procedural merge probability must remain exactly 5%")
    if float(procedural.get("small_branch_fraction", 0.0)) <= 0.0:
        raise ValueError("some procedural branches must use a small opening angle")
    if not config["validation"].get("no_test_split", False):
        raise ValueError("this campaign intentionally has train/validation only")


def phase_for_epoch(config: dict[str, Any], epoch: int) -> dict[str, Any]:
    for phase in config["training"]["phases"]:
        start, end = phase["epochs"]
        if int(start) <= int(epoch) <= int(end):
            return phase
    raise KeyError(f"epoch {epoch} is outside the configured campaign")


def required_environment(config: dict[str, Any]) -> tuple[Path, Path]:
    paths = config["paths"]
    repo_name = paths["sam3_repo_env"]
    checkpoint_name = paths["sam3_checkpoint_env"]
    try:
        repo = Path(os.environ[repo_name]).expanduser().resolve()
        checkpoint = Path(os.environ[checkpoint_name]).expanduser().resolve()
    except KeyError as error:
        raise RuntimeError(f"required environment variable is unset: {error.args[0]}") from error
    return repo, checkpoint
