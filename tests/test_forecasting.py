from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tqlab.forecasting import forecast_benchmark
from tqlab.reporting import render_forecast_report


def _seasonal_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=96, freq="1h", tz="UTC")
    pattern = np.sin(2.0 * np.pi * np.arange(24) / 24.0)
    return pd.DataFrame({"timestamp": timestamps, "signal": np.tile(pattern, 4)})


def test_forecast_benchmark_is_chronological_and_reproducible() -> None:
    frame = _seasonal_frame()
    result = forecast_benchmark(
        frame,
        signals=["signal"],
        seasonal_period=24,
        rolling_window=6,
        holdout_fraction=0.20,
    )
    seasonal = result.metrics.loc[result.metrics["model"] == "seasonal_naive"].iloc[0]
    assert seasonal["mae"] == 0.0
    assert result.split_timestamp == frame["timestamp"].iloc[int(len(frame) * 0.8)]

    split = int(len(frame) * 0.8)
    first = result.predictions.iloc[0]
    assert first["seasonal_naive"] == pytest.approx(frame["signal"].iloc[split - 24])
    assert first["rolling_mean"] == pytest.approx(frame["signal"].iloc[split - 6 : split].mean())

    changed = frame.copy()
    changed.loc[split:, "signal"] += 100.0
    changed_result = forecast_benchmark(
        changed,
        signals=["signal"],
        seasonal_period=24,
        rolling_window=6,
        holdout_fraction=0.20,
    )
    assert changed_result.predictions.iloc[0]["seasonal_naive"] == first["seasonal_naive"]
    assert changed_result.predictions.iloc[0]["rolling_mean"] == first["rolling_mean"]


def test_forecast_benchmark_infers_daily_period_and_renders(tmp_path) -> None:
    frame = _seasonal_frame()
    result = forecast_benchmark(frame, signals=["signal"], rolling_window=6)
    assert result.seasonal_period == 24
    report = render_forecast_report(result, tmp_path / "forecast.html")
    html = report.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "strictly before" in html
    assert "http://" not in html
    assert "https://" not in html


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestamp_column": "missing"}, "timestamp column"),
        ({"holdout_fraction": 0.8}, "holdout_fraction"),
        ({"rolling_window": 1}, "rolling_window"),
        ({"seasonal_period": 1}, "seasonal_period"),
        ({"seasonal_period": 90}, "too short"),
        ({"signals": ["missing"]}, "signal column"),
    ],
)
def test_forecast_benchmark_validates_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        forecast_benchmark(_seasonal_frame(), **kwargs)


def test_forecast_requires_numeric_signals_and_distinct_timestamps() -> None:
    frame = _seasonal_frame()
    frame["text"] = "value"
    with pytest.raises(ValueError, match="no numeric"):
        forecast_benchmark(frame[["timestamp", "text"]])
    frame["timestamp"] = frame["timestamp"].iloc[0]
    with pytest.raises(ValueError, match="distinct"):
        forecast_benchmark(frame, signals=["signal"])
