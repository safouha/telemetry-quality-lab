sed: --: No such file or directory
"""Deterministic synthetic telemetry and labeled defect injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tqlab.models import TruthEvent, truth_frame

SCENARIOS: dict[str, set[str]] = {
    "clean": set(),
    "spikes": {"spike"},
    "outage": {"missing_value", "timestamp_gap", "duplicate_timestamp"},
    "flatline": {"flatline"},
    "shift": {"level_shift"},
    "noise": {"noise_burst"},
    "mixed": {
        "missing_value",
        "timestamp_gap",
        "duplicate_timestamp",
        "spike",
        "flatline",
        "level_shift",
        "noise_burst",
    },
}


@dataclass(slots=True)
class SyntheticDataset:
    frame: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict[str, Any]


def _bounded_width(size: int, divisor: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, size // divisor))


def generate_dataset(
    *,
    scenario: str = "mixed",
    duration: str = "72h",
    frequency: str = "5min",
    seed: int = 42,
    start: str | pd.Timestamp = "2026-01-01T00:00:00Z",
) -> SyntheticDataset:
    """Generate a deterministic multivariate telemetry dataset with event labels.

    The first 20 percent of the timeline is always free of injected defects. All
    generated values, labels, and metadata depend only on the supplied arguments.
    """

    if scenario not in SCENARIOS:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown scenario {scenario!r}; choose from {choices}")

    duration_delta = pd.Timedelta(duration)
    cadence = pd.Timedelta(frequency)
    if duration_delta <= pd.Timedelta(0) or cadence <= pd.Timedelta(0):
        raise ValueError("duration and frequency must be positive")
    periods = int(duration_delta / cadence)
    if periods < 120:
        raise ValueError("duration and frequency must produce at least 120 observations")

    start_at = pd.Timestamp(start)
    if start_at.tzinfo is None:
        start_at = start_at.tz_localize("UTC")
    else:
        start_at = start_at.tz_convert("UTC")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_at, periods=periods, freq=cadence)
    step = np.arange(periods, dtype=float)
    steps_per_day = max(float(pd.Timedelta("1d") / cadence), 1.0)

    daily = np.sin(2.0 * np.pi * step / steps_per_day)
    short_cycle = np.sin(2.0 * np.pi * step / max(steps_per_day / 4.0, 8.0))
    load = 0.62 + 0.17 * daily + 0.07 * short_cycle + rng.normal(0.0, 0.018, periods)

    temperature = 23.5 + 8.4 * load + 0.7 * daily + rng.normal(0.0, 0.22, periods)
    pressure = 4.8 + 1.7 * load - 0.018 * (temperature - 28.0) + rng.normal(0.0, 0.045, periods)
    vibration = 0.30 + 0.58 * load + np.abs(rng.normal(0.0, 0.035, periods))

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
        }
    )
    enabled = SCENARIOS[scenario]
    events: list[TruthEvent] = []

    def add_event(
        defect_type: str,
        signal: str,
        start_index: int,
        end_index: int | None = None,
        severity: str = "medium",
    ) -> None:
        finish = start_index if end_index is None else end_index
        events.append(
            TruthEvent(
                event_id=f"E{len(events) + 1:03d}",
                defect_type=defect_type,
                signal=signal,
                start=timestamps[start_index],
                end=timestamps[finish],
                severity=severity,
            )
        )

    # Locations are intentionally separated, making event-level evaluation unambiguous.
    missing_start = int(periods * 0.38)
    missing_width = _bounded_width(periods, 55, 5, 16)
    gap_start = int(periods * 0.47)
    gap_width = _bounded_width(periods, 80, 3, 10)
    duplicate_index = int(periods * 0.55)
    flatline_start = int(periods * 0.61)
    flatline_width = _bounded_width(periods, 24, 12, 30)
    shift_start = int(periods * 0.72)
    shift_width = _bounded_width(periods, 22, 10, 28)
    noise_start = int(periods * 0.84)
    noise_width = _bounded_width(periods, 28, 10, 24)

    if "spike" in enabled:
        for index, sign in ((int(periods * 0.32), 1.0), (int(periods * 0.93), -1.0)):
            local_scale = float(frame["pressure"].iloc[: max(index, 20)].std())
            frame.loc[index, "pressure"] += sign * max(8.0 * local_scale, 1.1)
            add_event("spike", "pressure", index, severity="high")

    if "missing_value" in enabled:
        end = missing_start + missing_width - 1
        frame.loc[missing_start:end, "temperature"] = np.nan
        add_event("missing_value", "temperature", missing_start, end, "high")

    gap_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if "timestamp_gap" in enabled:
        end = gap_start + gap_width - 1
        gap_bounds = (timestamps[gap_start], timestamps[end])
        add_event("timestamp_gap", "__timestamp__", gap_start, end, "high")

    duplicate_at: pd.Timestamp | None = None
    if "duplicate_timestamp" in enabled:
        duplicate_at = timestamps[duplicate_index]
        add_event("duplicate_timestamp", "__timestamp__", duplicate_index, severity="medium")

    if "flatline" in enabled:
        end = flatline_start + flatline_width - 1
        frame.loc[flatline_start:end, "vibration"] = frame.loc[flatline_start - 1, "vibration"]
        add_event("flatline", "vibration", flatline_start, end, "high")

    if "level_shift" in enabled:
        end = shift_start + shift_width - 1
        shift_size = max(float(frame["temperature"].iloc[:shift_start].std()) * 2.8, 2.4)
        frame.loc[shift_start:end, "temperature"] += shift_size
        add_event("level_shift", "temperature", shift_start, end, "high")

    if "noise_burst" in enabled:
        end = noise_start + noise_width - 1
        baseline_scale = float(frame["pressure"].iloc[:noise_start].std())
        burst = rng.normal(0.0, max(4.5 * baseline_scale, 0.65), noise_width)
        frame.loc[noise_start:end, "pressure"] += burst
        add_event("noise_burst", "pressure", noise_start, end, "high")

    if gap_bounds is not None:
        gap_mask = frame["timestamp"].between(gap_bounds[0], gap_bounds[1])
        frame = frame.loc[~gap_mask].copy()

    if duplicate_at is not None:
        duplicate_row = frame.loc[frame["timestamp"] == duplicate_at].iloc[[0]].copy()
        frame = pd.concat([frame, duplicate_row], ignore_index=True)

    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame[["temperature", "pressure", "vibration"]] = frame[
        ["temperature", "pressure", "vibration"]
    ].round(6)

    metadata = {
        "scenario": scenario,
        "seed": seed,
        "start": start_at.isoformat(),
        "duration": str(duration_delta),
        "frequency": str(cadence),
        "rows": len(frame),
        "signals": ["temperature", "pressure", "vibration"],
        "generator": "tqlab",
    }
    return SyntheticDataset(frame=frame, truth=truth_frame(events), metadata=metadata)
