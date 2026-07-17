sed: --: No such file or directory
"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "timestamp_column": "timestamp",
    "calibration_fraction": 0.20,
    "cadence": {"gap_factor": 1.5},
    "spike": {"window": 31, "min_periods": 15, "threshold": 7.5},
    "flatline": {"min_points": 12, "tolerance_factor": 0.05},
    "shift": {"window": 18, "threshold": 4.0, "persistence": 3},
    "noise": {"window": 18, "threshold": 3.0, "persistence": 3},
    "quality_weights": {
        "completeness": 0.25,
        "timeliness": 0.20,
        "uniqueness": 0.15,
        "validity": 0.20,
        "stability": 0.20,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load a YAML configuration over the built-in defaults."""

    supplied: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("configuration root must be a mapping")
        supplied = loaded
    config = _merge(DEFAULT_CONFIG, supplied)
    if overrides:
        config = _merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    fraction = float(config["calibration_fraction"])
    if not 0.05 <= fraction <= 0.50:
        raise ValueError("calibration_fraction must be between 0.05 and 0.50")

    positive_fields = [
        ("cadence.gap_factor", config["cadence"]["gap_factor"]),
        ("spike.window", config["spike"]["window"]),
        ("spike.min_periods", config["spike"]["min_periods"]),
        ("spike.threshold", config["spike"]["threshold"]),
        ("flatline.min_points", config["flatline"]["min_points"]),
        ("flatline.tolerance_factor", config["flatline"]["tolerance_factor"]),
        ("shift.window", config["shift"]["window"]),
        ("shift.threshold", config["shift"]["threshold"]),
        ("shift.persistence", config["shift"]["persistence"]),
        ("noise.window", config["noise"]["window"]),
        ("noise.threshold", config["noise"]["threshold"]),
        ("noise.persistence", config["noise"]["persistence"]),
    ]
    for name, value in positive_fields:
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive")

    weights = config["quality_weights"]
    expected = {"completeness", "timeliness", "uniqueness", "validity", "stability"}
    if set(weights) != expected:
        raise ValueError("quality_weights must define all five quality dimensions")
    if any(float(weight) < 0 for weight in weights.values()):
        raise ValueError("quality weights must not be negative")
    if abs(sum(float(weight) for weight in weights.values()) - 1.0) > 1e-9:
        raise ValueError("quality weights must sum to 1.0")
