from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from geoelectrical_resistivity_forecasting_benchmark.adapters import (
    ARIMAAdapter,
    ETSAdapter,
    FourierRidgeAdapter,
)
from geoelectrical_resistivity_forecasting_benchmark.contracts import ForecastRequest, Track


def request_with_missing() -> ForecastRequest:
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    values = pd.Series(np.sin(np.arange(400) * 2 * np.pi / 365.25), index=dates)
    values.iloc[100:110] = np.nan
    return ForecastRequest(
        window_id="stat-test",
        series_id="11003/D/3211",
        origin=date(2021, 2, 4),
        horizon=7,
        track=Track.U,
        history_target=values,
    )


@pytest.mark.parametrize("adapter", [ETSAdapter(), ARIMAAdapter(), FourierRidgeAdapter()])
def test_statistical_adapters_preserve_daily_horizon_with_missing_values(adapter) -> None:
    result = adapter.predict(request_with_missing())
    assert result.point_forecast.shape == (7,)
    assert np.isfinite(result.point_forecast).all()
    assert result.target_dates.equals(pd.date_range("2021-02-04", periods=7, freq="D"))
