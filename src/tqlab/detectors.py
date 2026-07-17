"""Causal, explainable data-quality detectors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from tqlab.models import Finding

TIMESTAMP_SIGNAL = "__timestamp__"


def robust_mad(values: Iterable[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def true_runs(mask: pd.Series | np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def detect_timestamp_issues(
    timestamps: pd.Series,
    *,
    gap_factor: float,
) -> tuple[list[Finding], pd.Timedelta | None]:
    """Inspect timestamps in arrival order, then detect gaps on the sorted timeline."""

    parsed = pd.to_datetime(timestamps, utc=True, errors="coerce")
    findings: list[Finding] = []
    valid = parsed.dropna()
    if valid.empty:
        return findings, None

    anchor = pd.Timestamp(valid.iloc[0])
    for position in np.flatnonzero(parsed.isna().to_numpy()):
        findings.append(
            Finding(
                detector="timestamp_parser",
                defect_type="invalid_timestamp",
                signal=TIMESTAMP_SIGNAL,
                start=anchor,
                end=anchor,
                detected_at=anchor,
                score=1.0,
                threshold=0.0,
                details={"row_position": int(position)},
            )
        )

    valid_values = valid.reset_index(drop=True)
    negative = valid_values.diff() < pd.Timedelta(0)
    for position in np.flatnonzero(negative.to_numpy()):
        current = pd.Timestamp(valid_values.iloc[position])
        previous = pd.Timestamp(valid_values.iloc[position - 1])
        findings.append(
            Finding(
                detector="arrival_order",
                defect_type="out_of_order_timestamp",
                signal=TIMESTAMP_SIGNAL,
                start=current,
                end=previous,
                detected_at=current,
                score=float((previous - current).total_seconds()),
                threshold=0.0,
                details={"previous_timestamp": previous.isoformat()},
            )
        )

    duplicate_values = valid_values[valid_values.duplicated(keep="first")]
    for duplicate in duplicate_values:
        at = pd.Timestamp(duplicate)
        findings.append(
            Finding(
                detector="timestamp_uniqueness",
                defect_type="duplicate_timestamp",
                signal=TIMESTAMP_SIGNAL,
                start=at,
                end=at,
                detected_at=at,
                score=1.0,
                threshold=0.0,
                details={"duplicate_count": int((valid_values == at).sum())},
            )
        )

    unique = pd.Series(valid_values.drop_duplicates().sort_values().to_numpy())
    deltas = unique.diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return findings, None
    expected = pd.Timedelta(positive.median())

    for position in range(1, len(unique)):
        previous = pd.Timestamp(unique.iloc[position - 1])
        current = pd.Timestamp(unique.iloc[position])
        delta = current - previous
        if delta > expected * gap_factor:
            missing_start = previous + expected
            missing_end = current - expected
            estimated = max(round(delta / expected) - 1, 1)
            findings.append(
                Finding(
                    detector="cadence_gap",
                    defect_type="timestamp_gap",
                    signal=TIMESTAMP_SIGNAL,
                    start=missing_start,
                    end=max(missing_start, missing_end),
                    detected_at=current,
                    score=float(delta / expected),
                    threshold=float(gap_factor),
                    details={
                        "expected_seconds": expected.total_seconds(),
                        "observed_seconds": delta.total_seconds(),
                        "estimated_missing_rows": estimated,
                    },
                )
            )
    return findings, expected


def detect_missing(series: pd.Series, signal: str) -> list[Finding]:
    findings: list[Finding] = []
    for start, end in true_runs(series.isna()):
        findings.append(
            Finding(
                detector="completeness",
                defect_type="missing_value",
                signal=signal,
                start=series.index[start],
                end=series.index[end],
                detected_at=series.index[start],
                score=float(end - start + 1),
                threshold=0.0,
                details={"missing_points": end - start + 1},
            )
        )
    return findings


def detect_spikes(
    series: pd.Series,
    signal: str,
    *,
    window: int,
    min_periods: int,
    threshold: float,
) -> list[Finding]:
    """Trailing Hampel detector; the current point is excluded from its baseline."""

    history = series.shift(1)
    median = history.rolling(window, min_periods=min_periods).median()
    mad = history.rolling(window, min_periods=min_periods).apply(robust_mad, raw=True)
    fallback = history.rolling(window, min_periods=min_periods).std(ddof=0)
    scale = (1.4826 * mad).where(mad > 1e-12, fallback).clip(lower=1e-12)
    scores = (series - median).abs() / scale
    flagged = (scores > threshold) & series.notna()

    findings: list[Finding] = []
    for position in np.flatnonzero(flagged.fillna(False).to_numpy()):
        at = series.index[position]
        findings.append(
            Finding(
                detector="trailing_hampel",
                defect_type="spike",
                signal=signal,
                start=at,
                end=at,
                detected_at=at,
                score=float(scores.iloc[position]),
                threshold=float(threshold),
                details={
                    "baseline_median": round(float(median.iloc[position]), 8),
                    "robust_scale": round(float(scale.iloc[position]), 8),
                    "window": int(window),
                },
            )
        )
    return findings


def detect_flatlines(
    series: pd.Series,
    signal: str,
    *,
    min_points: int,
    tolerance_factor: float,
    calibration_fraction: float,
) -> list[Finding]:
    calibration_size = min(
        len(series), max(min_points * 2, int(len(series) * calibration_fraction))
    )
    calibration_diffs = series.iloc[:calibration_size].diff().abs().dropna()
    typical_change = max(1.4826 * robust_mad(calibration_diffs), 1e-9)
    tolerance = max(typical_change * tolerance_factor, 1e-9)
    unchanged = (series.diff().abs() <= tolerance) & series.notna() & series.shift(1).notna()

    findings: list[Finding] = []
    for run_start, run_end in true_runs(unchanged):
        point_count = run_end - run_start + 2
        if point_count < min_points:
            continue
        start = max(run_start - 1, 0)
        detected = min(start + min_points - 1, run_end)
        findings.append(
            Finding(
                detector="run_length_flatline",
                defect_type="flatline",
                signal=signal,
                start=series.index[start],
                end=series.index[run_end],
                detected_at=series.index[detected],
                score=float(point_count),
                threshold=float(min_points),
                details={"points": point_count, "tolerance": tolerance},
            )
        )
    return findings


def detect_level_shifts(
    series: pd.Series,
    signal: str,
    *,
    window: int,
    threshold: float,
    persistence: int,
    calibration_fraction: float,
) -> list[Finding]:
    calibration_size = min(len(series), max(window * 2, int(len(series) * calibration_fraction)))
    calibration = series.iloc[:calibration_size].dropna()
    calibration_scale = max(1.4826 * robust_mad(calibration), 1e-9)
    recent_median = series.rolling(window, min_periods=window).median()
    prior_median = series.shift(window).rolling(window, min_periods=window).median()
    scores = (recent_median - prior_median).abs() / calibration_scale
    flagged = (scores > threshold).fillna(False)

    findings: list[Finding] = []
    for start, end in true_runs(flagged):
        if end - start + 1 < persistence:
            continue
        detected = start + persistence - 1
        findings.append(
            Finding(
                detector="two_window_shift",
                defect_type="level_shift",
                signal=signal,
                start=series.index[start],
                end=series.index[end],
                detected_at=series.index[detected],
                score=float(scores.iloc[start : end + 1].max()),
                threshold=float(threshold),
                details={
                    "window": int(window),
                    "persistence": int(persistence),
                    "calibration_scale": calibration_scale,
                },
            )
        )
    return findings


def detect_noise_bursts(
    series: pd.Series,
    signal: str,
    *,
    window: int,
    threshold: float,
    persistence: int,
    calibration_fraction: float,
) -> list[Finding]:
    differences = series.diff()
    calibration_size = min(len(series), max(window * 2, int(len(series) * calibration_fraction)))
    baseline = max(1.4826 * robust_mad(differences.iloc[1:calibration_size]), 1e-9)
    local = differences.rolling(window, min_periods=window).apply(robust_mad, raw=True) * 1.4826
    ratios = local / baseline
    flagged = (ratios > threshold).fillna(False)

    findings: list[Finding] = []
    for start, end in true_runs(flagged):
        if end - start + 1 < persistence:
            continue
        detected = start + persistence - 1
        findings.append(
            Finding(
                detector="rolling_noise_ratio",
                defect_type="noise_burst",
                signal=signal,
                start=series.index[start],
                end=series.index[end],
                detected_at=series.index[detected],
                score=float(ratios.iloc[start : end + 1].max()),
                threshold=float(threshold),
                details={
                    "window": int(window),
                    "persistence": int(persistence),
                    "baseline_difference_scale": baseline,
                },
            )
        )
    return findings
