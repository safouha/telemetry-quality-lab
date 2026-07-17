sed: --: No such file or directory
"""Event-level and point-level evaluation against labeled synthetic defects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from tqlab.models import Finding, findings_frame

BENCHMARK_TYPES = (
    "missing_value",
    "timestamp_gap",
    "duplicate_timestamp",
    "spike",
    "flatline",
    "level_shift",
    "noise_burst",
)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _normalize(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for column in ("start", "end"):
        normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
    if "detected_at" in normalized:
        normalized["detected_at"] = pd.to_datetime(
            normalized["detected_at"], utc=True, errors="coerce"
        )
    return normalized.dropna(subset=["start", "end"])


def _event_pairs(
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for truth_position, truth_row in truth.reset_index(drop=True).iterrows():
        truth_start = pd.Timestamp(truth_row["start"])
        truth_end = pd.Timestamp(truth_row["end"])
        for predicted_position, predicted_row in predicted.reset_index(drop=True).iterrows():
            predicted_start = pd.Timestamp(predicted_row["start"])
            predicted_end = pd.Timestamp(predicted_row["end"])
            if predicted_end < truth_start - tolerance or predicted_start > truth_end + tolerance:
                continue
            intersection_start = max(truth_start, predicted_start)
            intersection_end = min(truth_end, predicted_end)
            overlap = max((intersection_end - intersection_start).total_seconds(), 0.0)
            boundary_distance = abs((predicted_start - truth_start).total_seconds())
            score = overlap + 1.0 / (1.0 + boundary_distance)
            candidates.append((score, truth_position, predicted_position))

    matched_truth: set[int] = set()
    matched_predicted: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, truth_position, predicted_position in sorted(candidates, reverse=True):
        if truth_position in matched_truth or predicted_position in matched_predicted:
            continue
        matched_truth.add(truth_position)
        matched_predicted.add(predicted_position)
        pairs.append((truth_position, predicted_position))
    return pairs


def _mask(events: pd.DataFrame, timeline: pd.DatetimeIndex) -> np.ndarray:
    result = np.zeros(len(timeline), dtype=bool)
    for _, event in events.iterrows():
        result |= (timeline >= event["start"]) & (timeline <= event["end"])
    return result


def _metrics_row(
    defect_type: str,
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    cadence: pd.Timedelta,
) -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    delays: list[float] = []
    for signal in sorted(set(truth.get("signal", [])) | set(predicted.get("signal", []))):
        signal_truth = truth.loc[truth["signal"] == signal].reset_index(drop=True)
        signal_predicted = predicted.loc[predicted["signal"] == signal].reset_index(drop=True)
        signal_pairs = _event_pairs(signal_truth, signal_predicted, cadence)
        pairs.extend(signal_pairs)
        for truth_position, predicted_position in signal_pairs:
            truth_start = pd.Timestamp(signal_truth.iloc[truth_position]["start"])
            detected_at = pd.Timestamp(signal_predicted.iloc[predicted_position]["detected_at"])
            delays.append(max((detected_at - truth_start).total_seconds(), 0.0))

    event_tp = len(pairs)
    event_fp = len(predicted) - event_tp
    event_fn = len(truth) - event_tp
    event_precision = event_tp / len(predicted) if len(predicted) else 1.0
    event_recall = event_tp / len(truth) if len(truth) else 1.0

    truth_mask = np.zeros(len(timeline), dtype=bool)
    predicted_mask = np.zeros(len(timeline), dtype=bool)
    for signal in sorted(set(truth.get("signal", [])) | set(predicted.get("signal", []))):
        truth_mask |= _mask(truth.loc[truth["signal"] == signal], timeline)
        predicted_mask |= _mask(predicted.loc[predicted["signal"] == signal], timeline)
    point_tp = int(np.sum(truth_mask & predicted_mask))
    point_fp = int(np.sum(~truth_mask & predicted_mask))
    point_fn = int(np.sum(truth_mask & ~predicted_mask))
    point_tn = int(np.sum(~truth_mask & ~predicted_mask))
    point_precision = point_tp / (point_tp + point_fp) if point_tp + point_fp else 1.0
    point_recall = point_tp / (point_tp + point_fn) if point_tp + point_fn else 1.0

    return {
        "defect_type": defect_type,
        "truth_events": len(truth),
        "predicted_events": len(predicted),
        "matched_events": event_tp,
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
        "event_precision": round(event_precision, 6),
        "event_recall": round(event_recall, 6),
        "event_f1": round(_f1(event_precision, event_recall), 6),
        "point_tp": point_tp,
        "point_fp": point_fp,
        "point_fn": point_fn,
        "point_tn": point_tn,
        "point_precision": round(point_precision, 6),
        "point_recall": round(point_recall, 6),
        "point_f1": round(_f1(point_precision, point_recall), 6),
        "false_positive_rate": round(point_fp / max(point_fp + point_tn, 1), 6),
        "mean_detection_delay_seconds": round(float(np.mean(delays)), 3) if delays else np.nan,
    }


def evaluate_findings(
    truth: pd.DataFrame,
    findings: list[Finding] | pd.DataFrame,
    *,
    cadence: pd.Timedelta,
    timeline: Iterable[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Evaluate findings on both event boundaries and a regular point grid."""

    if cadence <= pd.Timedelta(0):
        raise ValueError("cadence must be positive")
    truth_table = _normalize(truth)
    predicted_table = _normalize(
        findings_frame(findings) if isinstance(findings, list) else findings
    )
    observed = (
        pd.DatetimeIndex(pd.to_datetime(list(timeline), utc=True)).drop_duplicates().sort_values()
    )
    if observed.empty:
        raise ValueError("timeline must contain at least one timestamp")
    regular = pd.date_range(observed.min(), observed.max(), freq=cadence)

    rows: list[dict[str, Any]] = []
    for defect_type in BENCHMARK_TYPES:
        rows.append(
            _metrics_row(
                defect_type,
                truth_table.loc[truth_table["defect_type"] == defect_type],
                predicted_table.loc[predicted_table["defect_type"] == defect_type],
                regular,
                cadence,
            )
        )

    count_fields = [
        "truth_events",
        "predicted_events",
        "matched_events",
        "event_tp",
        "event_fp",
        "event_fn",
        "point_tp",
        "point_fp",
        "point_fn",
        "point_tn",
    ]
    aggregate: dict[str, Any] = {"defect_type": "__all__"}
    for field in count_fields:
        aggregate[field] = int(sum(int(row[field]) for row in rows))
    event_prediction_count = aggregate["event_tp"] + aggregate["event_fp"]
    event_truth_count = aggregate["event_tp"] + aggregate["event_fn"]
    aggregate["event_precision"] = round(
        aggregate["event_tp"] / event_prediction_count if event_prediction_count else 1.0,
        6,
    )
    aggregate["event_recall"] = round(
        aggregate["event_tp"] / event_truth_count if event_truth_count else 1.0,
        6,
    )
    aggregate["event_f1"] = round(_f1(aggregate["event_precision"], aggregate["event_recall"]), 6)
    point_prediction_count = aggregate["point_tp"] + aggregate["point_fp"]
    point_truth_count = aggregate["point_tp"] + aggregate["point_fn"]
    aggregate["point_precision"] = round(
        aggregate["point_tp"] / point_prediction_count if point_prediction_count else 1.0,
        6,
    )
    aggregate["point_recall"] = round(
        aggregate["point_tp"] / point_truth_count if point_truth_count else 1.0,
        6,
    )
    aggregate["point_f1"] = round(_f1(aggregate["point_precision"], aggregate["point_recall"]), 6)
    aggregate["false_positive_rate"] = round(
        aggregate["point_fp"] / max(aggregate["point_fp"] + aggregate["point_tn"], 1),
        6,
    )
    delays = [row["mean_detection_delay_seconds"] for row in rows]
    finite_delays = [float(delay) for delay in delays if pd.notna(delay)]
    aggregate["mean_detection_delay_seconds"] = (
        round(float(np.mean(finite_delays)), 3) if finite_delays else np.nan
    )
    return pd.DataFrame([aggregate, *rows])
