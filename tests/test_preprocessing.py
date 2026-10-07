from __future__ import annotations

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.preprocessing import causal_forward_fill
from geoelectrical_resistivity_forecasting_benchmark.robust_preprocessing import (
    causal_hampel_clip,
    prepare_causal_input,
)


def test_causal_fill_stops_after_fixed_limit() -> None:
    series = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 2.0])
    result = causal_forward_fill(series, limit=3)
    assert result.iloc[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert np.isnan(result.iloc[4])


def test_future_values_cannot_change_earlier_fill() -> None:
    index = pd.date_range("2024-01-01", periods=6)
    first = pd.Series([1.0, np.nan, np.nan, 2.0, 3.0, 4.0], index=index)
    second = first.copy()
    second.iloc[3:] += 1000
    left = causal_forward_fill(first, limit=3).iloc[:3]
    right = causal_forward_fill(second, limit=3).iloc[:3]
    pd.testing.assert_series_equal(left, right)


def test_causal_hampel_clips_spike_but_does_not_use_future() -> None:
    index = pd.date_range("2024-01-01", periods=25)
    original = pd.Series([1.0, 1.1] * 10 + [100.0, 1.0, 1.1, 1.0, 1.1], index=index)
    changed_future = original.copy()
    changed_future.iloc[21:] = 5000.0
    left = causal_hampel_clip(original, window=10, threshold=6.0, minimum_history=6)
    right = causal_hampel_clip(changed_future, window=10, threshold=6.0, minimum_history=6)
    assert left.iloc[20] < 2.0
    pd.testing.assert_series_equal(left.iloc[:21], right.iloc[:21])


def test_q0_input_is_only_causally_filled() -> None:
    values = pd.Series([1.0, float("nan"), 100.0])
    result = prepare_causal_input(values, maximum_fill_days=1, variant="Q0")
    assert result.tolist() == [1.0, 1.0, 100.0]
