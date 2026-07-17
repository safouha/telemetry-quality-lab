from __future__ import annotations

import numpy as np
import pandas as pd

from tqlab.detectors import (
    detect_flatlines,
    detect_level_shifts,
    detect_missing,
    detect_noise_bursts,
    detect_spikes,
    detect_timestamp_issues,
    robust_mad,
    true_runs,
)


def test_robust_mad_and_runs_cover_edge_cases() -> None:
    assert robust_mad([]) == 0.0
    assert robust_mad([1, 1, 2, np.nan]) == 0.0
    assert true_runs(np.array([], dtype=bool)) == []
    assert true_runs([False, True, True, False, True]) == [(1, 2), (4, 4)]


def test_timestamp_checks_find_invalid_order_duplicate_and_gap() -> None:
    timestamps = pd.Series(
        [
            "2026-01-01T00:00:00Z",
            "invalid",
            "2026-01-01T00:10:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:30:00Z",
        ]
    )
    findings, cadence = detect_timestamp_issues(timestamps, gap_factor=1.5)
    defect_types = {finding.defect_type for finding in findings}
    assert cadence == pd.Timedelta("5min")
    assert {
        "invalid_timestamp",
        "out_of_order_timestamp",
        "duplicate_timestamp",
        "timestamp_gap",
    } <= defect_types


def test_timestamp_checks_handle_no_valid_or_repeated_time() -> None:
    findings, cadence = detect_timestamp_issues(pd.Series(["bad", None]), gap_factor=1.5)
    assert findings == []
    assert cadence is None
    findings, cadence = detect_timestamp_issues(
        pd.Series(["2026-01-01", "2026-01-01"]), gap_factor=1.5
    )
    assert any(item.defect_type == "duplicate_timestamp" for item in findings)
    assert cadence is None


def test_value_detectors_identify_constructed_failures() -> None:
    rng = np.random.default_rng(4)
    index = pd.date_range("2026-01-01", periods=120, freq="1min", tz="UTC")

    spike_values = rng.normal(0.0, 0.1, len(index))
    spike_values[70] = 5.0
    spikes = detect_spikes(
        pd.Series(spike_values, index=index),
        "signal",
        window=25,
        min_periods=12,
        threshold=4.0,
    )
    assert any(finding.start == index[70] for finding in spikes)

    missing_values = pd.Series(rng.normal(size=len(index)), index=index)
    missing_values.iloc[20:25] = np.nan
    missing = detect_missing(missing_values, "signal")
    assert len(missing) == 1
    assert missing[0].details["missing_points"] == 5

    flat_values = pd.Series(rng.normal(0.0, 0.1, len(index)), index=index)
    flat_values.iloc[45:70] = flat_values.iloc[44]
    flatlines = detect_flatlines(
        flat_values,
        "signal",
        min_points=10,
        tolerance_factor=0.05,
        calibration_fraction=0.2,
    )
    assert any(finding.start <= index[45] <= finding.end for finding in flatlines)

    shifted = pd.Series(rng.normal(0.0, 0.1, len(index)), index=index)
    shifted.iloc[65:100] += 3.0
    shifts = detect_level_shifts(
        shifted,
        "signal",
        window=10,
        threshold=4.0,
        persistence=2,
        calibration_fraction=0.2,
    )
    assert any(finding.end >= index[65] for finding in shifts)

    noisy = pd.Series(rng.normal(0.0, 0.05, len(index)), index=index)
    noisy.iloc[65:100] += rng.normal(0.0, 1.5, 35)
    bursts = detect_noise_bursts(
        noisy,
        "signal",
        window=10,
        threshold=3.0,
        persistence=2,
        calibration_fraction=0.2,
    )
    assert any(finding.end >= index[65] for finding in bursts)


def test_detectors_return_empty_for_constant_series() -> None:
    index = pd.date_range("2026-01-01", periods=50, freq="1min", tz="UTC")
    constant = pd.Series(np.ones(50), index=index)
    spikes = detect_spikes(constant, "signal", window=10, min_periods=5, threshold=4.0)
    assert spikes == []
