sed: --: No such file or directory
from __future__ import annotations

import pandas as pd
import pytest

from tqlab.benchmarking import run_benchmark
from tqlab.evaluation import evaluate_findings
from tqlab.models import Finding, TruthEvent, truth_frame
from tqlab.pipeline import scan_frame
from tqlab.reporting import render_benchmark_report, render_report
from tqlab.synthetic import generate_dataset


def _finding(defect_type: str, signal: str, start, end=None, detected_at=None) -> Finding:
    finish = start if end is None else end
    detected = finish if detected_at is None else detected_at
    return Finding(
        detector="test_detector",
        defect_type=defect_type,
        signal=signal,
        start=start,
        end=finish,
        detected_at=detected,
        score=8.0,
        threshold=4.0,
        details={},
    )


def test_event_and_point_evaluation_for_perfect_predictions() -> None:
    timeline = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    truth = truth_frame(
        [
            TruthEvent("E1", "spike", "a", timeline[5], timeline[5]),
            TruthEvent("E2", "flatline", "b", timeline[10], timeline[13]),
        ]
    )
    findings = [
        _finding("spike", "a", timeline[5]),
        _finding("flatline", "b", timeline[10], timeline[13], timeline[11]),
    ]
    metrics = evaluate_findings(truth, findings, cadence=pd.Timedelta("1min"), timeline=timeline)
    aggregate = metrics.loc[metrics["defect_type"] == "__all__"].iloc[0]
    assert aggregate["matched_events"] == 2
    assert aggregate["event_f1"] == 1.0
    assert aggregate["point_f1"] == 1.0


def test_evaluation_counts_clean_false_positive_and_validates_inputs() -> None:
    timeline = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
    findings = [_finding("spike", "a", timeline[4])]
    metrics = evaluate_findings(
        truth_frame([]), findings, cadence=pd.Timedelta("1min"), timeline=timeline
    )
    spike = metrics.loc[metrics["defect_type"] == "spike"].iloc[0]
    assert spike["event_fp"] == 1
    assert spike["event_precision"] == 0.0
    assert spike["false_positive_rate"] > 0
    empty_metrics = evaluate_findings(
        truth_frame([]), [], cadence=pd.Timedelta("1min"), timeline=timeline
    )
    empty_aggregate = empty_metrics.loc[empty_metrics["defect_type"] == "__all__"].iloc[0]
    assert empty_aggregate["event_f1"] == 1.0
    assert empty_aggregate["point_f1"] == 1.0
    with pytest.raises(ValueError, match="cadence"):
        evaluate_findings(truth_frame([]), [], cadence=pd.Timedelta(0), timeline=timeline)
    with pytest.raises(ValueError, match="timeline"):
        evaluate_findings(truth_frame([]), [], cadence=pd.Timedelta("1min"), timeline=[])


def test_reports_are_self_contained(tmp_path) -> None:
    dataset = generate_dataset(duration="12h", scenario="mixed", seed=9)
    result = scan_frame(dataset.frame)
    metrics = evaluate_findings(
        dataset.truth,
        result.findings,
        cadence=result.cadence,
        timeline=result.frame[result.timestamp_column],
    )
    report = render_report(
        result,
        tmp_path / "report.html",
        truth=dataset.truth,
        metrics=metrics,
    )
    html = report.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "http://" not in html
    assert "https://" not in html

    benchmark = run_benchmark(scenarios=["clean"], seeds=[3], duration="12h", frequency="5min")
    benchmark_report = render_benchmark_report(benchmark, tmp_path / "benchmark.html")
    assert "data:image/png;base64," in benchmark_report.read_text(encoding="utf-8")


def test_benchmark_validates_selection() -> None:
    with pytest.raises(ValueError, match="scenario"):
        run_benchmark(scenarios=[], seeds=[1])
    with pytest.raises(ValueError, match="seed"):
        run_benchmark(scenarios=["clean"], seeds=[])
    with pytest.raises(ValueError, match="unknown"):
        run_benchmark(scenarios=["not-a-scenario"], seeds=[1])
