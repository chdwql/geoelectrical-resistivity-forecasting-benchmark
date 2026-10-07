from __future__ import annotations

import math

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.metrics import (
    interval_coverage,
    interval_score,
    interval_width,
    mae,
    mase,
    pinball_loss,
    rmse,
    station_equal_mean,
    weighted_interval_score,
)


def test_point_metrics_match_hand_calculation() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.0, 4.0, 2.0])
    assert mae(actual, forecast) == 1.0
    assert rmse(actual, forecast) == math.sqrt(5 / 3)
    assert mase(actual, forecast, np.array([0.0, 1.0, 2.0])) == 1.0


def test_mase_zero_scale_is_undefined() -> None:
    assert math.isnan(mase(np.array([1.0]), np.array([1.0]), np.ones(10)))


def test_mase_does_not_bridge_missing_history_days() -> None:
    history = np.array([0.0, np.nan, 100.0, 101.0])
    assert mase(np.array([2.0]), np.array([1.0]), history) == 1.0


def test_probabilistic_scores() -> None:
    actual = np.array([0.0, 2.0])
    median = np.array([1.0, 1.0])
    assert pinball_loss(actual, median, 0.5) == 0.5
    assert interval_score(actual, np.array([-1.0, 0.0]), np.array([1.0, 1.0]), 0.2) == 6.5


def test_interval_coverage_width_and_wis() -> None:
    actual = np.array([0.0, 2.0])
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 1.0])
    assert interval_coverage(actual, lower, upper) == 0.5
    assert interval_width(lower, upper) == 1.5
    assert weighted_interval_score(
        actual,
        np.array([0.0, 1.0]),
        [(0.2, lower, upper)],
    ) == 0.6


def test_station_equal_aggregation_does_not_weight_row_count() -> None:
    frame = pd.DataFrame(
        {
            "station": ["A", "A", "A", "B"],
            "series_id": ["A/1", "A/1", "A/2", "B/1"],
            "window_id": ["w1", "w2", "w3", "w4"],
            "mase": [1.0, 1.0, 3.0, 10.0],
        }
    )
    # Station A: mean(series A/1=1, A/2=3)=2; station B=10; overall=6.
    assert station_equal_mean(frame, value_column="mase") == 6.0
