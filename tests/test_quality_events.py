from __future__ import annotations

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.quality_events import detect_series_events


def test_retrospective_detector_marks_step_and_spike_without_changing_values() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-01", periods=500, freq="D")
    values = pd.Series(rng.normal(0, 0.2, len(dates)), index=dates)
    values.loc[dates[250:]] += 5
    values.loc[dates[400]] += 8
    original = values.copy()

    events = detect_series_events(
        values,
        reference_start="2016-01-01",
        reference_end="2016-06-30",
        window_days=14,
        minimum_side_observations=7,
        threshold_mad=8,
        cluster_days=14,
    )

    assert "step_candidate" in set(events["event_type"])
    assert "spike_candidate" in set(events["event_type"])
    pd.testing.assert_series_equal(values, original)
    assert (events["available_at"] > events["event_date"]).all()
