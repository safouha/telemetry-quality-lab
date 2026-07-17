"""End-to-end scanning pipeline and transparent quality scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from tqlab.config import load_config
from tqlab.detectors import (
    TIMESTAMP_SIGNAL,
    detect_flatlines,
    detect_level_shifts,
    detect_missing,
    detect_noise_bursts,
    detect_spikes,
    detect_timestamp_issues,
)
from tqlab.models import Finding, findings_frame


@dataclass(slots=True)
class ScanResult:
    frame: pd.DataFrame
    findings: list[Finding]
    findings_table: pd.DataFrame
    signal_summary: pd.DataFrame
    cadence: pd.Timedelta | None
    timestamp_column: str
    config: dict[str, Any]


def _numeric_signals(frame: pd.DataFrame, timestamp_column: str) -> dict[str, pd.Series]:
    signals: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column == timestamp_column:
            continue
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        nonempty = int(original.notna().sum())
        minimum = min(3, nonempty)
        if nonempty > 0 and int(converted.notna().sum()) >= minimum:
            signals[str(column)] = converted.astype(float)
    return signals


def _interval_count(
    index: pd.DatetimeIndex, findings: list[Finding], defect_types: set[str]
) -> int:
    affected = np.zeros(len(index), dtype=bool)
    for finding in findings:
        if finding.defect_type in defect_types:
            affected |= (index >= finding.start) & (index <= finding.end)
    return int(affected.sum())


def _consolidate_findings(
    findings: list[Finding], cadence: pd.Timedelta | None, config: dict[str, Any]
) -> list[Finding]:
    """Prefer persistent explanations and join nearby shift boundaries."""

    if cadence is None:
        return findings
    padding_points = max(
        int(config["flatline"]["min_points"]),
        int(config["shift"]["window"]),
        int(config["noise"]["window"]),
    )
    padding = cadence * padding_points
    noise_findings = [item for item in findings if item.defect_type == "noise_burst"]
    classified: list[Finding] = []
    for finding in findings:
        if finding.defect_type == "level_shift":
            explained_by_noise = any(
                alternate.signal == finding.signal
                and alternate.start - padding <= finding.start <= alternate.end + padding
                for alternate in noise_findings
            )
            if not explained_by_noise:
                classified.append(finding)
            continue
        classified.append(finding)

    persistent_types = {"flatline", "level_shift", "noise_burst"}
    persistent = [item for item in classified if item.defect_type in persistent_types]
    consolidated: list[Finding] = []
    for finding in classified:
        if finding.defect_type == "spike":
            explained = any(
                alternate.signal == finding.signal
                and alternate.start - padding <= finding.start <= alternate.end + padding
                for alternate in persistent
            )
            if not explained:
                consolidated.append(finding)
            continue
        consolidated.append(finding)

    shift_window = int(config["shift"]["window"])
    boundary_offset = cadence * (shift_window // 2)
    merge_gap = cadence * shift_window
    merged: list[Finding] = []
    by_signal: dict[str, list[Finding]] = {}
    for finding in consolidated:
        if finding.defect_type == "level_shift":
            by_signal.setdefault(finding.signal, []).append(finding)
        else:
            merged.append(finding)
    for signal_findings in by_signal.values():
        ordered = sorted(signal_findings, key=lambda item: item.start)
        position = 0
        while position < len(ordered):
            current = ordered[position]
            if position + 1 < len(ordered):
                following = ordered[position + 1]
                if following.start - current.end <= merge_gap:
                    estimated_start = current.start - boundary_offset
                    estimated_end = max(estimated_start, following.start - boundary_offset)
                    merged.append(
                        Finding(
                            detector="paired_window_shift",
                            defect_type="level_shift",
                            signal=current.signal,
                            start=estimated_start,
                            end=estimated_end,
                            detected_at=current.detected_at,
                            score=max(current.score, following.score),
                            threshold=current.threshold,
                            details={
                                "window": shift_window,
                                "merged_change_regions": 2,
                                "boundary_offset_points": shift_window // 2,
                            },
                        )
                    )
                    position += 2
                    continue
            merged.append(current)
            position += 1
    return merged


def _quality_summary(
    prepared: pd.DataFrame,
    signals: list[str],
    findings: list[Finding],
    config: dict[str, Any],
    raw_rows: int,
) -> pd.DataFrame:
    weights = {name: float(value) for name, value in config["quality_weights"].items()}
    index = pd.DatetimeIndex(prepared[config["timestamp_column"]])
    timestamp_findings = [item for item in findings if item.signal == TIMESTAMP_SIGNAL]
    duplicate_count = sum(item.defect_type == "duplicate_timestamp" for item in timestamp_findings)
    out_of_order_count = sum(
        item.defect_type == "out_of_order_timestamp" for item in timestamp_findings
    )
    estimated_missing = sum(
        int(item.details.get("estimated_missing_rows", 0))
        for item in timestamp_findings
        if item.defect_type == "timestamp_gap"
    )

    uniqueness = 100.0 * (1.0 - min(duplicate_count / max(raw_rows, 1), 1.0))
    timing_burden = estimated_missing + out_of_order_count
    timeliness = 100.0 * (1.0 - min(timing_burden / max(raw_rows + estimated_missing, 1), 1.0))

    rows: list[dict[str, Any]] = []
    for signal in signals:
        series = prepared[signal]
        signal_findings = [item for item in findings if item.signal == signal]
        missing = int(series.isna().sum())
        completeness = 100.0 * (1.0 - missing / max(len(series), 1))
        spike_points = _interval_count(
            index,
            signal_findings,
            {"spike"},
        )
        unstable_points = _interval_count(
            index,
            signal_findings,
            {"flatline", "level_shift", "noise_burst"},
        )
        # Small isolated validity errors deserve visibility without hiding the raw counts.
        validity = 100.0 * (1.0 - min(5.0 * spike_points / max(len(series), 1), 1.0))
        stability = 100.0 * (1.0 - min(unstable_points / max(len(series), 1), 1.0))
        dimensions = {
            "completeness": completeness,
            "timeliness": timeliness,
            "uniqueness": uniqueness,
            "validity": validity,
            "stability": stability,
        }
        overall = sum(dimensions[name] * weights[name] for name in dimensions)
        rows.append(
            {
                "signal": signal,
                "observations": len(series),
                "missing_points": missing,
                "finding_count": len(signal_findings),
                **{name: round(value, 2) for name, value in dimensions.items()},
                "quality_score": round(overall, 2),
            }
        )

    if rows:
        numeric_dimensions = [
            "completeness",
            "timeliness",
            "uniqueness",
            "validity",
            "stability",
            "quality_score",
        ]
        aggregate = {
            "signal": "__dataset__",
            "observations": raw_rows,
            "missing_points": sum(int(row["missing_points"]) for row in rows),
            "finding_count": len(findings),
   sed: --: No such file or directory
     }
        for name in numeric_dimensions:
            aggregate[name] = round(float(np.mean([float(row[name]) for row in rows])), 2)
        rows.insert(0, aggregate)
    return pd.DataFrame(rows)


def scan_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    config: dict[str, Any] | None = None,
) -> ScanResult:
    """Scan a telemetry frame with causal detectors and return structured findings."""

    effective = load_config(overrides=config) if config is not None else load_config()
    column = timestamp_column or str(effective["timestamp_column"])
    effective["timestamp_column"] = column
    if column not in frame.columns:
        raise ValueError(f"timestamp column {column!r} was not found")

    timestamp_findings, cadence = detect_timestamp_issues(
        frame[column], gap_factor=float(effective["cadence"]["gap_factor"])
    )
    parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
    signals = _numeric_signals(frame, column)
    if not signals:
        raise ValueError("no numeric signal columns were found")

    prepared = pd.DataFrame({column: parsed})
    for name, values in signals.items():
        prepared[name] = values
    prepared = prepared.dropna(subset=[column]).sort_values(column, kind="stable")
    prepared = prepared.groupby(column, sort=True, as_index=False).first()

    findings = list(timestamp_findings)
    calibration_fraction = float(effective["calibration_fraction"])
    indexed = prepared.set_index(column)
    for signal in signals:
        series = indexed[signal]
        findings.extend(detect_missing(series, signal))
        findings.extend(detect_spikes(series, signal, **effective["spike"]))
        findings.extend(
            detect_flatlines(
                series,
                signal,
                calibration_fraction=calibration_fraction,
                **effective["flatline"],
            )
        )
        findings.extend(
            detect_level_shifts(
                series,
                signal,
                calibration_fraction=calibration_fraction,
                **effective["shift"],
            )
        )
        findings.extend(
            detect_noise_bursts(
                series,
                signal,
                calibration_fraction=calibration_fraction,
                **effective["noise"],
            )
        )

    findings = _consolidate_findings(findings, cadence, effective)
    findings.sort(key=lambda item: (item.start, item.signal, item.defect_type, item.detector))
    findings = [replace(item, finding_id=f"F{index:04d}") for index, item in enumerate(findings, 1)]
    summary = _quality_summary(
        prepared,
        list(signals),
        findings,
        effective,
        raw_rows=len(frame),
    )
    return ScanResult(
        frame=prepared,
        findings=findings,
        findings_table=findings_frame(findings),
        signal_summary=summary,
        cadence=cadence,
        timestamp_column=column,
        config=effective,
    )
