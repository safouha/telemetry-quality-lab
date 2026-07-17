sed: --: No such file or directory
from __future__ import annotations

import pandas as pd
import pytest

from tqlab.config import load_config
from tqlab.pipeline import scan_frame
from tqlab.synthetic import generate_dataset


def test_pipeline_returns_structured_findings_and_scores() -> None:
    dataset = generate_dataset(duration="24h", scenario="mixed", seed=5)
    result = scan_frame(dataset.frame)
    assert result.cadence == pd.Timedelta("5min")
    assert result.findings
    assert list(result.findings_table["finding_id"]) == [
        f"F{index:04d}" for index in range(1, len(result.findings) + 1)
    ]
    assert set(result.signal_summary["signal"]) == {
        "__dataset__",
        "temperature",
        "pressure",
        "vibration",
    }
    scores = result.signal_summary["quality_score"]
    assert scores.between(0, 100).all()


def test_pipeline_supports_custom_timestamp_name() -> None:
    dataset = generate_dataset(duration="12h", scenario="clean")
    frame = dataset.frame.rename(columns={"timestamp": "observed_at"})
    result = scan_frame(frame, timestamp_column="observed_at")
    assert result.timestamp_column == "observed_at"


def test_pipeline_consolidates_competing_diagnoses() -> None:
    dataset = generate_dataset(duration="48h", scenario="mixed", seed=11)
    result = scan_frame(dataset.frame)
    defect_types = [finding.defect_type for finding in result.findings]
    assert defect_types.count("spike") == 2
    assert defect_types.count("level_shift") == 1
    assert defect_types.count("noise_burst") == 1


def test_pipeline_rejects_missing_timestamp_or_numeric_signal() -> None:
    with pytest.raises(ValueError, match="was not found"):
        scan_frame(pd.DataFrame({"value": [1, 2, 3]}))
    with pytest.raises(ValueError, match="no numeric signal"):
        scan_frame(pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3)}))


def test_config_merges_and_validates(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("spike:\n  threshold: 7.0\n", encoding="utf-8")
    config = load_config(path)
    assert config["spike"]["threshold"] == 7.0
    assert config["spike"]["window"] == 31

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("calibration_fraction: 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_fraction"):
        load_config(invalid)

    invalid.write_text("quality_weights:\n  completeness: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sum"):
        load_config(invalid)
