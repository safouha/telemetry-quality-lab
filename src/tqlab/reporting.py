# ruff: noqa: E501
"""Self-contained HTML reporting with embedded charts and styles."""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import BaseLoader, Environment, select_autoescape

from tqlab.forecasting import ForecastResult
from tqlab.pipeline import ScanResult

REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root { --ink:#172033; --muted:#64748b; --line:#e2e8f0; --panel:#fff;
            --accent:#2563eb; --good:#15803d; --warn:#b45309; --bad:#b91c1c; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f8fafc; color:var(--ink); font:14px/1.55 system-ui,sans-serif; }
    main { width:min(1180px, 94vw); margin:36px auto 60px; }
    header { padding:26px 30px; color:#fff; border-radius:14px;
             background:linear-gradient(125deg,#172554,#1d4ed8); box-shadow:0 12px 30px #1e3a8a2b; }
    h1 { margin:0 0 4px; font-size:30px; }
    h2 { margin:0 0 16px; font-size:20px; }
    .subtitle { margin:0; color:#dbeafe; }
    .meta { margin-top:12px; font-size:12px; color:#bfdbfe; }
    section { margin-top:22px; padding:22px; background:var(--panel); border:1px solid var(--line);
              border-radius:12px; box-shadow:0 3px 14px #0f172a0b; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:12px; }
    .card { padding:16px; border:1px solid var(--line); border-radius:10px; background:#f8fafc; }
    .card strong { display:block; margin-top:5px; font-size:25px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
    .score-good { color:var(--good); } .score-warn { color:var(--warn); } .score-bad { color:var(--bad); }
    .chart { width:100%; height:auto; display:block; }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
    th { color:#475569; background:#f8fafc; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
    tbody tr:hover { background:#f8fafc; }
    code { padding:2px 5px; border-radius:4px; background:#e2e8f0; }
    .note { color:var(--muted); }
    footer { padding:24px 4px; color:var(--muted); font-size:12px; text-align:center; }
  </style>
</head>
<body><main>
  <header>
    <h1>{{ title }}</h1>
    <p class="subtitle">Explainable checks over timestamp integrity, completeness, validity, and stability.</p>
    <div class="meta">Generated {{ generated_at }} · cadence {{ cadence }}</div>
  </header>

  <section>
    <h2>Run summary</h2>
    <div class="cards">
      <div class="card"><span class="label">Rows analyzed</span><strong>{{ rows }}</strong></div>
      <div class="card"><span class="label">Signals</span><strong>{{ signals }}</strong></div>
      <div class="card"><span class="label">Findings</span><strong>{{ finding_count }}</strong></div>
      <div class="card"><span class="label">Quality score</span><strong class="{{ score_class }}">{{ quality_score }}</strong></div>
      {% if aggregate_metric %}
      <div class="card"><span class="label">Event F1</span><strong>{{ aggregate_metric.event_f1 }}</strong></div>
      <div class="card"><span class="label">Point F1</span><strong>{{ aggregate_metric.point_f1 }}</strong></div>
      {% endif %}
    </div>
  </section>

  <section>
    <h2>Telemetry timeline</h2>
    <img class="chart" alt="Telemetry timeline with labeled and detected intervals" src="data:image/png;base64,{{ chart }}">
    <p class="note">Red shading marks labeled synthetic defects. Amber shading marks detected intervals.</p>
  </section>

  <section>
    <h2>Quality dimensions</h2>
    {{ summary_table | safe }}
  </section>

  {% if metrics_table %}
  <section>
    <h2>Benchmark metrics</h2>
    {{ metrics_table | safe }}
    <p class="note">Event metrics use one-to-one temporal matching. Point metrics use the inferred regular cadence.</p>
  </section>
  {% endif %}

  <section>
    <h2>Detected findings</h2>
    {% if findings_table %}{{ findings_table | safe }}{% else %}<p>No defects were detected.</p>{% endif %}
  </section>

  <section>
    <h2>Interpretation</h2>
    <p>All detectors are causal: each score uses the current observation and information available before it.
       Scores are diagnostic summaries, not service-level guarantees. Thresholds should be calibrated for the
       sampling cadence, operating range, and risk tolerance of the dataset being inspected.</p>
  </section>
  <footer>Telemetry Quality Lab · report contains no external scripts, fonts, images, or network requests.</footer>
</main></body></html>"""

BENCHMARK_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Benchmark report</title>
<style>body{margin:0;background:#f8fafc;color:#172033;font:14px/1.5 system-ui,sans-serif}
main{width:min(1100px,94vw);margin:36px auto}header,section{border-radius:12px;padding:24px;margin-bottom:20px}
header{color:white;background:linear-gradient(125deg,#172554,#1d4ed8)}section{background:white;border:1px solid #e2e8f0}
h1,h2{margin-top:0}.chart{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}th{background:#f8fafc}.wrap{overflow:auto}
.note{color:#64748b}</style></head><body><main><header><h1>Telemetry quality benchmark</h1>
<p>Reproducible evaluation across labeled synthetic scenarios and seeds.</p></header><section><h2>Event F1 by scenario</h2>
<img class="chart" alt="Mean event F1 by scenario" src="data:image/png;base64,{{ chart }}"></section>
<section><h2>Run metrics</h2>{{ metrics_table | safe }}<p class="note">Every value in this report is calculated from the bundled generator and detector output.</p></section>
</main></body></html>"""

FORECAST_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Forecast benchmark</title>
<style>body{margin:0;background:#f8fafc;color:#172033;font:14px/1.5 system-ui,sans-serif}
main{width:min(1100px,94vw);margin:36px auto}header,section{border-radius:12px;padding:24px;margin-bottom:20px}
header{color:white;background:linear-gradient(125deg,#172554,#1d4ed8)}section{background:white;border:1px solid #e2e8f0}
h1,h2{margin-top:0}.chart{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}th{background:#f8fafc}.wrap{overflow:auto}
.meta,.note{color:#64748b}</style></head><body><main><header><h1>Chronological forecast benchmark</h1>
<p>One-step-ahead seasonal-naive and rolling-mean baselines over the final holdout.</p></header>
<section><h2>Holdout predictions</h2><img class="chart" alt="Actual and baseline forecasts" src="data:image/png;base64,{{ chart }}">
<p class="meta">Split {{ split_timestamp }} · cadence {{ cadence }} · seasonal period {{ seasonal_period }} observations · rolling window {{ rolling_window }}</p></section>
<section><h2>Error metrics</h2>{{ metrics_table | safe }}<p class="note">Lower MAE, RMSE, and sMAPE values are better. Every prediction uses observations strictly before its timestamp.</p></section>
</main></body></html>"""


def _png_data(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _timeline_chart(result: ScanResult, truth: pd.DataFrame | None) -> str:
    signals = [column for column in result.frame.columns if column != result.timestamp_column]
    figure, axes = plt.subplots(
        len(signals),
        1,
        figsize=(11, max(2.5 * len(signals), 3.2)),
        sharex=True,
        squeeze=False,
    )
    timestamps = pd.to_datetime(result.frame[result.timestamp_column], utc=True)
    truth_table = truth.copy() if truth is not None else pd.DataFrame()
    if not truth_table.empty:
        truth_table["start"] = pd.to_datetime(truth_table["start"], utc=True)
        truth_table["end"] = pd.to_datetime(truth_table["end"], utc=True)

    for axis, signal in zip(axes[:, 0], signals, strict=True):
        axis.plot(timestamps, result.frame[signal], color="#1d4ed8", linewidth=1.0)
        axis.set_ylabel(signal.replace("_", " ").title())
        axis.grid(alpha=0.20)
        if not truth_table.empty:
            relevant = truth_table.loc[truth_table["signal"].isin([signal, "__timestamp__"])]
            for _, event in relevant.iterrows():
                axis.axvspan(event["start"], event["end"], color="#dc2626", alpha=0.13)
        for finding in result.findings:
            if finding.signal in {signal, "__timestamp__"}:
                axis.axvspan(finding.start, finding.end, color="#f59e0b", alpha=0.13)
    axes[-1, 0].set_xlabel("Timestamp (UTC)")
    figure.tight_layout()
    return _png_data(figure)


def _html_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int = 200) -> str:
    if frame.empty:
        return ""
    displayed = frame.copy()
    if columns:
        displayed = displayed[[column for column in columns if column in displayed.columns]]
    displayed = displayed.head(limit)
    float_columns = displayed.select_dtypes(include=[np.floating]).columns
    displayed[float_columns] = displayed[float_columns].round(3)
    return '<div class="table-wrap">' + displayed.to_html(index=False, escape=True) + "</div>"


def render_report(
    result: ScanResult,
    output_path: Path | str,
    *,
    truth: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    title: str = "Telemetry quality report",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_row = result.signal_summary.loc[result.signal_summary["signal"] == "__dataset__"]
    quality_score = float(dataset_row.iloc[0]["quality_score"]) if not dataset_row.empty else 0.0
    score_class = (
        "score-good"
        if quality_score >= 90
        else "score-warn"
        if quality_score >= 75
        else "score-bad"
    )
    aggregate_metric: dict[str, Any] | None = None
    if metrics is not None and not metrics.empty:
        aggregate = metrics.loc[metrics["defect_type"] == "__all__"]
        if not aggregate.empty:
            aggregate_metric = {
                "event_f1": f"{float(aggregate.iloc[0]['event_f1']):.3f}",
                "point_f1": f"{float(aggregate.iloc[0]['point_f1']):.3f}",
            }

    environment = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=True))
    html = environment.from_string(REPORT_TEMPLATE).render(
        title=title,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        cadence=str(result.cadence) if result.cadence is not None else "undetermined",
        rows=len(result.frame),
        signals=len(result.frame.columns) - 1,
        finding_count=len(result.findings),
        quality_score=f"{quality_score:.1f}",
        score_class=score_class,
        aggregate_metric=aggregate_metric,
        chart=_timeline_chart(result, truth),
        summary_table=_html_table(result.signal_summary),
        metrics_table=_html_table(
            metrics if metrics is not None else pd.DataFrame(),
            [
                "defect_type",
                "truth_events",
                "predicted_events",
                "event_precision",
                "event_recall",
                "event_f1",
                "point_f1",
                "false_positive_rate",
                "mean_detection_delay_seconds",
            ],
        ),
        findings_table=_html_table(
            result.findings_table,
            [
                "finding_id",
                "defect_type",
                "signal",
                "start",
                "end",
                "detected_at",
                "score",
                "threshold",
            ],
        ),
    )
    output.write_text(html, encoding="utf-8")
    return output


def render_benchmark_report(metrics: pd.DataFrame, output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = metrics.loc[metrics["defect_type"] == "__all__"]
    means = aggregate.groupby("scenario", sort=False)["event_f1"].mean().sort_values()
    figure, axis = plt.subplots(figsize=(9, max(3.0, 0.48 * len(means) + 1.5)))
    axis.barh(means.index, means.values, color="#2563eb")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean event F1")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()

    environment = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=True))
    html = environment.from_string(BENCHMARK_TEMPLATE).render(
        chart=_png_data(figure),
        metrics_table=_html_table(
            aggregate,
            [
                "scenario",
                "seed",
                "truth_events",
                "predicted_events",
                "event_precision",
                "event_recall",
                "event_f1",
                "point_f1",
                "false_positive_rate",
                "mean_detection_delay_seconds",
            ],
            limit=1000,
        ),
    )
    output.write_text(html, encoding="utf-8")
    return output


def _forecast_chart(result: ForecastResult) -> str:
    figure, axes = plt.subplots(
        len(result.signals),
        1,
        figsize=(11, max(2.7 * len(result.signals), 3.4)),
        sharex=True,
        squeeze=False,
    )
    for axis, signal in zip(axes[:, 0], result.signals, strict=True):
        rows = result.predictions.loc[result.predictions["signal"] == signal]
        timestamps = pd.to_datetime(rows[result.timestamp_column], utc=True)
        axis.plot(timestamps, rows["actual"], label="Actual", color="#0f172a", linewidth=1.4)
        axis.plot(
            timestamps,
            rows["seasonal_naive"],
            label="Seasonal naive",
            color="#2563eb",
            linewidth=1.0,
        )
        axis.plot(
            timestamps,
            rows["rolling_mean"],
            label="Rolling mean",
            color="#d97706",
            linewidth=1.0,
        )
        axis.set_ylabel(signal.replace("_", " ").title())
        axis.grid(alpha=0.20)
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Timestamp (UTC)")
    figure.tight_layout()
    return _png_data(figure)


def render_forecast_report(result: ForecastResult, output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = Environment(loader=BaseLoader(), autoescape=select_autoescape(default=True))
    html = environment.from_string(FORECAST_TEMPLATE).render(
        chart=_forecast_chart(result),
        split_timestamp=result.split_timestamp.isoformat(),
        cadence=str(result.cadence),
        seasonal_period=result.seasonal_period,
        rolling_window=result.rolling_window,
        metrics_table=_html_table(
            result.metrics,
            ["signal", "model", "evaluation_points", "mae", "rmse", "smape"],
        ),
    )
    output.write_text(html, encoding="utf-8")
    return output
