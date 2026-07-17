from __future__ import annotations

import pandas as pd
import pytest

from tqlab.synthetic import generate_dataset


def test_generation_is_deterministic() -> None:
    first = generate_dataset(duration="12h", frequency="5min", seed=17)
    second = generate_dataset(duration="12h", frequency="5min", seed=17)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.truth, second.truth)
    assert first.metadata == second.metadata


def test_mixed_scenario_contains_every_supported_injection() -> None:
    dataset = generate_dataset(duration="24h", frequency="5min", scenario="mixed")
    assert set(dataset.truth["defect_type"]) == {
        "missing_value",
        "timestamp_gap",
        "duplicate_timestamp",
        "spike",
        "flatline",
        "level_shift",
        "noise_burst",
    }
    timestamps = pd.to_datetime(dataset.frame["timestamp"], utc=True)
    gap = dataset.truth.loc[dataset.truth["defect_type"] == "timestamp_gap"].iloc[0]
    assert not timestamps.between(pd.Timestamp(gap["start"]), pd.Timestamp(gap["end"])).any()
    duplicate = dataset.truth.loc[dataset.truth["defect_type"] == "duplicate_timestamp"].iloc[0]
    assert int((timestamps == pd.Timestamp(duplicate["start"])).sum()) == 2


def test_calibration_region_is_clean() -> None:
    dataset = generate_dataset(duration="24h", frequency="5min", scenario="mixed")
    calibration_end = pd.Timestamp(dataset.frame["timestamp"].iloc[int(len(dataset.frame) * 0.2)])
    starts = pd.to_datetime(dataset.truth["start"], utc=True)
    assert bool((starts > calibration_end).all())


def test_clean_scenario_has_no_labels() -> None:
    dataset = generate_dataset(duration="12h", scenario="clean")
    assert dataset.truth.empty
    assert not dataset.frame.isna().any().any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scenario": "unknown"}, "unknown scenario"),
        ({"duration": "1h", "frequency": "5min"}, "at least 120"),
        ({"duration": "-1h"}, "must be positive"),
    ],
)
def test_generation_rejects_invalid_inputs(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_dataset(**kwargs)
