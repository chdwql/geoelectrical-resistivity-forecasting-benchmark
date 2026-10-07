from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.adapters import (
    DriftAdapter,
    PersistenceAdapter,
    SeasonalPersistenceAdapter,
)
from geoelectrical_resistivity_forecasting_benchmark.contracts import ForecastRequest, Track


def request(history: pd.Series, horizon: int = 3) -> ForecastRequest:
    return ForecastRequest(
        window_id="window",
        series_id="11003/D/3211",
        origin=date(2024, 1, 1),
        horizon=horizon,
        track=Track.U,
        history_target=history,
    )


def test_persistence_and_drift() -> None:
    history = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2023-12-29", periods=3))
    assert np.array_equal(PersistenceAdapter().predict(request(history)).point_forecast, [3, 3, 3])
    assert np.array_equal(DriftAdapter().predict(request(history)).point_forecast, [4, 5, 6])


def test_seasonal_persistence_uses_only_lagged_history() -> None:
    history = pd.Series(
        np.arange(370, dtype=float),
        index=pd.date_range("2022-12-27", periods=370, freq="D"),
    )
    result = SeasonalPersistenceAdapter().predict(request(history))
    expected_dates = result.target_dates - pd.Timedelta(days=365)
    assert np.array_equal(result.point_forecast, history.reindex(expected_dates).to_numpy())
