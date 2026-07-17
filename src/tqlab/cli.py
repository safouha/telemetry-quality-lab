"""Command-line interface for generation, scanning, and benchmarking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from tqlab import __version__
from tqlab.benchmarking import run_benchmark
from tqlab.config import load_config
from tqlab.evaluation import evaluate_findings
from tqlab.forecasting import forecast_benchmark
from tqlab.pipeline import ScanResult, scan_frame
from tqlab.synthetic import SCENARIOS, SyntheticDataset, generate_dataset

app = typer.Typer(
    name="tqlab",
    help="Generate, inspect, and benchmark operational telemetry quality.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)
console = Console()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_dataset(dataset: SyntheticDataset, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(output / "telemetry.csv", index=False)
    dataset.truth.to_csv(output / "ground_truth.csv", index=False)
    _json_write(output / "generation_manifest.json", dataset.metadata)


def _save_scan(result: ScanResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result.findings_table.to_csv(output / "findings.csv", index=False)
    result.signal_summary.to_csv(output / "signal_summary.csv", index=False)


def _show_summary(result: ScanResult) -> None:
    table = Table(title="Telemetry quality summary")
    table.add_column("Signal")
    table.add_column("Score", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Findings", justify="right")
    for _, row in result.signal_summary.iterrows():
        if row["signal"] == "__dataset__":
            continue
        table.add_row(
            str(row["signal"]),
            f"{float(row['quality_score']):.1f}",
            str(int(row["missing_points"])),
            str(int(row["finding_count"])),
        )
    console.print(table)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command("demo")
def demo_command(
    output: Path = typer.Option(Path("artifacts/demo"), "--output", "-o"),
    seed: int = typer.Option(42, "--seed"),
    duration: str = typer.Option("72h", "--duration"),
    frequency: str = typer.Option("5min", "--frequency"),
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
) -> None:
    """Run generation, scanning, evaluation, and reporting in one command."""

    from tqlab.reporting import render_report

    detector_config = load_config(config)
    dataset = generate_dataset(scenario="mixed", duration=duration, frequency=frequency, seed=seed)
    _save_dataset(dataset, output)
    result = scan_frame(dataset.frame, config=detector_config)
    metrics = evaluate_findings(
        dataset.truth,
        result.findings,
        cadence=result.cadence or pd.Timedelta(frequency),
        timeline=result.frame[result.timestamp_column],
    )
    _save_scan(result, output)
    metrics.to_csv(output / "metrics.csv", index=False)
    _json_write(
        output / "run_manifest.json",
        {
            "tqlab_version": __version__,
            "scenario": "mixed",
            "seed": seed,
            "duration": duration,
            "frequency": frequency,
            "configuration_sha256": _config_fingerprint(detector_config),
        },
    )
    report = render_report(
        result,
        output / "report.html",
        truth=dataset.truth,
        metrics=metrics,
        title="Synthetic telemetry quality report",
    )
    _show_summary(result)
    console.print(f"Report: {report}")


@app.command("generate")
def generate_command(
    output: Path = typer.Option(Path("artifacts/generated"), "--output", "-o"),
    scenario: str = typer.Option("mixed", "--scenario"),
    duration: str = typer.Option("72h", "--duration"),
    frequency: str = typer.Option("5min", "--frequency"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Generate reproducible telemetry and its labeled defect events."""

    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"choose one of: {', '.join(sorted(SCENARIOS))}")
    dataset = generate_dataset(
        scenario=scenario,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )
    _save_dataset(dataset, output)
    console.print(f"Wrote {len(dataset.frame):,} rows and {len(dataset.truth)} labels to {output}")


@app.command("scan")
def scan_command(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(Path("artifacts/scan"), "--output", "-o"),
    timestamp: str | None = typer.Option(None, "--timestamp"),
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
) -> None:
    """Scan numeric signal columns in a local CSV file."""

    from tqlab.reporting import render_report

    frame = pd.read_csv(input_file)
    detector_config = load_config(config)
    result = scan_frame(frame, timestamp_column=timestamp, config=detector_config)
    _save_scan(result, output)
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    _json_write(
        output / "run_manifest.json",
        {
            "tqlab_version": __version__,
            "input_file": input_file.name,
            "input_sha256": digest,
            "configuration_sha256": _config_fingerprint(result.config),
        },
    )
    report = render_report(result, output / "report.html")
    _show_summary(result)
    console.print(f"Report: {report}")


@app.command("benchmark")
def benchmark_command(
    output: Path = typer.Option(Path("artifacts/benchmark"), "--output", "-o"),
    suite: Path | None = typer.Option(None, "--suite", exists=True, dir_okay=False),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated integer seeds."),
) -> None:
    """Evaluate the detector suite across labeled scenarios and random seeds."""

    from tqlab.reporting import render_benchmark_report

    specification: dict[str, Any] = {
        "scenarios": ["clean", "spikes", "outage", "flatline", "shift", "noise", "mixed"],
        "duration": "48h",
        "frequency": "5min",
        "seeds": [11, 23, 47],
    }
    if suite is not None:
        loaded = yaml.safe_load(suite.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise typer.BadParameter("suite root must be a mapping")
        specification.update(loaded)
    if seeds is not None:
        try:
            specification["seeds"] = [int(value.strip()) for value in seeds.split(",")]
        except ValueError as error:
            raise typer.BadParameter("seeds must be comma-separated integers") from error

    metrics = run_benchmark(
        scenarios=specification["scenarios"],
        seeds=specification["seeds"],
        duration=str(specification["duration"]),
        frequency=str(specification["frequency"]),
        detector_config=specification.get("detector_config"),
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "benchmark_metrics.csv", index=False)
    numeric = metrics.select_dtypes(include=["number"]).columns.tolist()
    summary = metrics.groupby(["scenario", "defect_type"], as_index=False)[numeric].mean()
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    report = render_benchmark_report(metrics, output / "report.html")
    _json_write(
        output / "run_manifest.json",
        {
            "tqlab_version": __version__,
            "scenarios": list(specification["scenarios"]),
            "seeds": list(specification["seeds"]),
            "duration": str(specification["duration"]),
            "frequency": str(specification["frequency"]),
        },
    )
    aggregate = metrics.loc[metrics["defect_type"] == "__all__"]
    table = Table(title="Benchmark summary")
    table.add_column("Scenario")
    table.add_column("Mean event F1", justify="right")
    table.add_column("Mean point F1", justify="right")
    for scenario, group in aggregate.groupby("scenario", sort=False):
        table.add_row(
            str(scenario),
            f"{group['event_f1'].mean():.3f}",
            f"{group['point_f1'].mean():.3f}",
        )
    console.print(table)
    console.print(f"Report: {report}")


@app.command("forecast")
def forecast_command(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(Path("artifacts/forecast"), "--output", "-o"),
    timestamp: str = typer.Option("timestamp", "--timestamp"),
    signal: str | None = typer.Option(None, "--signal"),
    holdout: float = typer.Option(0.20, "--holdout", min=0.10, max=0.50),
    seasonal_period: int | None = typer.Option(None, "--seasonal-period", min=2),
    rolling_window: int = typer.Option(12, "--rolling-window", min=2),
) -> None:
    """Compare chronological seasonal-naive and rolling-mean forecasts."""

    from tqlab.reporting import render_forecast_report

    frame = pd.read_csv(input_file)
    result = forecast_benchmark(
        frame,
        timestamp_column=timestamp,
        signals=[signal] if signal else None,
        holdout_fraction=holdout,
        seasonal_period=seasonal_period,
        rolling_window=rolling_window,
    )
    output.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output / "forecast_predictions.csv", index=False)
    result.metrics.to_csv(output / "forecast_metrics.csv", index=False)
    _json_write(
        output / "run_manifest.json",
        {
            "tqlab_version": __version__,
            "input_file": input_file.name,
            "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
            "timestamp_column": timestamp,
            "signals": result.signals,
            "holdout_fraction": holdout,
            "seasonal_period": result.seasonal_period,
            "rolling_window": result.rolling_window,
            "split_timestamp": result.split_timestamp.isoformat(),
        },
    )
    report = render_forecast_report(result, output / "report.html")
    table = Table(title="Chronological forecast benchmark")
    table.add_column("Signal")
    table.add_column("Model")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("sMAPE", justify="right")
    for _, row in result.metrics.iterrows():
        table.add_row(
            str(row["signal"]),
            str(row["model"]),
            f"{float(row['mae']):.4f}",
            f"{float(row['rmse']):.4f}",
            f"{float(row['smape']):.2f}%",
        )
    console.print(table)
    console.print(f"Report: {report}")
sed: --: No such file or directory
