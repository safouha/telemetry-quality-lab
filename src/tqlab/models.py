sed: --: No such file or directory
"""Typed records shared by generation, detection, evaluation, and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


def _timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


@dataclass(frozen=True, slots=True)
class TruthEvent:
    event_id: str
    defect_type: str
    signal: str
    start: pd.Timestamp
    end: pd.Timestamp
    severity: str = "medium"

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _timestamp(self.start))
        object.__setattr__(self, "end", _timestamp(self.end))
        if self.end < self.start:
            raise ValueError("truth event end must not precede its start")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["start"] = self.start.isoformat()
        record["end"] = self.end.isoformat()
        return record


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str
    defect_type: str
    signal: str
    start: pd.Timestamp
    end: pd.Timestamp
    detected_at: pd.Timestamp
    score: float
    threshold: float
    details: dict[str, Any]
    finding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _timestamp(self.start))
        object.__setattr__(self, "end", _timestamp(self.end))
        object.__setattr__(self, "detected_at", _timestamp(self.detected_at))
        if self.end < self.start:
            raise ValueError("finding end must not precede its start")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["start"] = self.start.isoformat()
        record["end"] = self.end.isoformat()
        record["detected_at"] = self.detected_at.isoformat()
        record["details"] = json.dumps(self.details, sort_keys=True)
        return record


def truth_frame(events: list[TruthEvent]) -> pd.DataFrame:
    columns = ["event_id", "defect_type", "signal", "start", "end", "severity"]
    return pd.DataFrame([event.as_record() for event in events], columns=columns)


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    columns = [
        "finding_id",
        "detector",
        "defect_type",
        "signal",
        "start",
        "end",
        "detected_at",
        "score",
        "threshold",
        "details",
    ]
    return pd.DataFrame([finding.as_record() for finding in findings], columns=columns)
