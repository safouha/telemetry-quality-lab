sed: --: No such file or directory
from __future__ import annotations

from typer.testing import CliRunner

from tqlab.cli import app

runner = CliRunner()


def test_version_and_generate_commands(tmp_path) -> None:
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert "1.0.0" in version.stdout

    output = tmp_path / "generated"
    result = runner.invoke(
        app,
        [
            "generate",
            "--scenario",
            "outage",
            "--duration",
            "12h",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert (output / "telemetry.csv").exists()
    assert (output / "ground_truth.csv").exists()


def test_demo_and_scan_commands(tmp_path) -> None:
    demo_output = tmp_path / "demo"
    demo = runner.invoke(
        app,
        ["demo", "--duration", "12h", "--seed", "8", "--output", str(demo_output)],
    )
    assert demo.exit_code == 0, demo.exception
    assert (demo_output / "report.html").exists()
    assert (demo_output / "metrics.csv").exists()

    scan_output = tmp_path / "scan"
    scan = runner.invoke(
        app,
        [
            "scan",
            str(demo_output / "telemetry.csv"),
            "--output",
            str(scan_output),
        ],
    )
    assert scan.exit_code == 0, scan.exception
    assert (scan_output / "findings.csv").exists()
    assert (scan_output / "report.html").exists()


def test_benchmark_command_with_small_suite(tmp_path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "scenarios: [clean]\nduration: 12h\nfrequency: 5min\nseeds: [2]\n",
        encoding="utf-8",
    )
    output = tmp_path / "benchmark"
    result = runner.invoke(
        app,
        ["benchmark", "--suite", str(suite), "--output", str(output)],
    )
    assert result.exit_code == 0, result.exception
    assert (output / "benchmark_metrics.csv").exists()
    assert (output / "report.html").exists()


def test_forecast_command(tmp_path) -> None:
    generated = tmp_path / "generated"
    generation = runner.invoke(
        app,
        [
            "generate",
            "--scenario",
            "clean",
            "--duration",
            "48h",
            "--output",
            str(generated),
        ],
    )
    assert generation.exit_code == 0, generation.exception
    output = tmp_path / "forecast"
    result = runner.invoke(
        app,
        [
            "forecast",
            str(generated / "telemetry.csv"),
            "--signal",
            "temperature",
            "--seasonal-period",
            "288",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert (output / "forecast_predictions.csv").exists()
    assert (output / "forecast_metrics.csv").exists()
    assert (output / "report.html").exists()
