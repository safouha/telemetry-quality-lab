"""Reproducible multi-scenario benchmark orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from tqlab.evaluation import evaluate_findings
from tqlab.pipeline import scan_frame
from tqlab.synthetic import SCENARIOS, generate_dataset


def run_benchmark(
    *,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    duration: str = "48h",
    frequency: str = "5min",
    detector_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected_scenarios = list(scenarios)
    selected_seeds = list(seeds)
    if not selected_scenarios:
        raise ValueError("benchmark requires at least one scenario")
    if not selected_seeds:
        raise ValueError("benchmark requires at least one seed")
    unknown = set(selected_scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown benchmark scenarios: {', '.join(sorted(unknown))}")

    for scenario in selected_scenarios:
        for seed in selected_seeds:
            dataset = generate_dataset(
                scenario=scenario,
                duration=duration,
                frequency=frequency,
                seed=int(seed),
            )
            result = scan_frame(dataset.frame, config=detector_config)
            cadence = result.cadence or pd.Timedelta(frequency)
            metrics = evaluate_findings(
                dataset.truth,
                result.findings,
                cadence=cadence,
                timeline=result.frame[result.timestamp_column],
            )
            metrics.insert(0, "seed", int(seed))
            metrics.insert(0, "scenario", scenario)
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)
