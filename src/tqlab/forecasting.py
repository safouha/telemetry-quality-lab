"""Chronological one-step-ahead forecasting baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ForecastResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    cadence: pd.Timedelta
    seasonal_period: int
    rolling_window: int
    split_timestamp: pd.Timestamp
    timestamp_column: str
    signals: list[str]


def _metric_row(
    signal: str, model: str, actual: pd.Series, predicted: pd.Series
) -> dict[str, object]:
    valid = actual.notna() & predicted.notna()
    observed = actual.loc[valid].to_numpy(dtype=float)
    forecast = predicted.loc[valid].to_numpy(dtype=float)
    if observed.size == 0:
        raise ValueError(f"model {model!r} has no evaluable points for signal {signal!r}")
    error = forecast - observed
    denominator = np.abs(observed) + np.abs(forecast)
    smape_terms = np.divide(
        200.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 1e-12,
    )
    return {
        "signal": signal,
        "model": model,
        "evaluation_points": int(observed.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(np.mean(smape_terms)),
    }


def forecast_benchmark(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    signals: list[str] | None = None,
    holdout_fraction: float = 0.20,
    seasonal_period: int | None = None,
    rolling_window: int = 12,
) -> ForecastResult:
    """Compare seasonal-naive and rolling-mean forecasts on a final holdout.

    Predictions are one-step ahead: each holdout prediction may use actual observations
    strictly before that timestamp, but never the current or future value.
    """

    if timestamp_column not in frame:
        raise ValueError(f"timestamp column {timestamp_column!r} was not found")
    if not 0.10 <= holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction must be between 0.10 and 0.50")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    parsed = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    prepared = pd.DataFrame({timestamp_column: parsed})
    candidates = signals or [column for column in frame.columns if column != timestamp_column]
    selected: list[str] = []
    for signal in candidates:
        if signal not in frame:
            raise ValueError(f"signal column {signal!r} was not found")
        converted = pd.to_numeric(frame[signal], errors="coerce")
        if converted.notna().sum() >= 3:
            prepared[signal] = converted.astype(float)
            selected.append(signal)
    if not selected:
        raise ValueError("no numeric signal columns were found")

    prepared = prepared.dropna(subset=[timestamp_column]).sort_values(
        timestamp_column, kind="stable"
    )
    prepared = prepared.groupby(timestamp_column, as_index=False, sort=True)[selected].mean()
    deltas = prepared[timestamp_column].diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("at least two distinct timestamps are required")
    cadence = pd.Timedelta(positive.median())
    inferred_season = max(round(pd.Timedelta("1d") / cadence), 2)
    season = inferred_season if seasonal_period is None else int(seasonal_period)
    if season < 2:
        raise ValueError("seasonal_period must be at least 2")

    split_position = int(len(prepared) * (1.0 - holdout_fraction))
    if split_position <= max(season, rolling_window) or split_position >= len(prepared):
        raise ValueError(
            "timeline is too short for the selected season, rolling window, and holdout"
        )

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for signal in selected:
        actual = prepared[signal]
        seasonal = actual.shift(season)
        rolling = actual.shift(1).rolling(rolling_window, min_periods=rolling_window).mean()
        holdout = pd.DataFrame(
            {
                timestamp_column: prepared[timestamp_column].iloc[split_position:],
                "signal": signal,
                "actual": actual.iloc[split_position:],
                "seasonal_naive": seasonal.iloc[split_position:],
                "rolling_mean": rolling.iloc[split_position:],
            }
        )
        prediction_parts.append(holdout)
        metric_rows.append(
            _metric_row(signal, "seasonal_naive", holdout["actual"], holdout["seasonal_naive"])
        )
        metric_rows.append(
            _metric_row(signal, "rolling_mean", holdout["actual"], holdout["rolling_mean"])
        )

    metrics = pd.DataFrame(metric_rows)
    metrics[["mae", "rmse", "smape"]] = metrics[["mae", "rmse", "smape"]].round(6)
    return ForecastResult(
        predictions=pd.concat(prediction_parts, ignore_index=True),
        metrics=metrics,
        cadence=cadence,
        seasonal_period=season,
        rolling_window=rolling_window,
        split_timestamp=pd.Timestamp(prepared[timestamp_column].iloc[split_position]),
        timestamp_column=timestamp_column,
        signals=selected,
    )
