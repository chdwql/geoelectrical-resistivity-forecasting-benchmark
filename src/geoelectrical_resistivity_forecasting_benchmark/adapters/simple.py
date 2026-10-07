from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import ForecastRequest, ForecastResult
from .base import ForecastAdapter


def _target_dates(request: ForecastRequest) -> pd.DatetimeIndex:
    return pd.date_range(request.origin, periods=request.horizon, freq="D")


def _clean_history(request: ForecastRequest) -> pd.Series:
    history = request.history_target.dropna().sort_index()
    if history.empty:
        raise ValueError("Target history contains no valid values")
    return history


class PersistenceAdapter(ForecastAdapter):
    model_id = "persistence"
    supported_tracks = frozenset({"U"})

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = _clean_history(request)
        values = np.full(request.horizon, float(history.iloc[-1]))
        return ForecastResult(target_dates=_target_dates(request), point_forecast=values)


class DriftAdapter(ForecastAdapter):
    model_id = "drift"
    supported_tracks = frozenset({"U"})

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = _clean_history(request)
        if len(history) < 2:
            raise ValueError("Drift requires at least two valid history values")
        elapsed = max((history.index[-1] - history.index[0]).days, 1)
        slope = (float(history.iloc[-1]) - float(history.iloc[0])) / elapsed
        steps = np.arange(1, request.horizon + 1, dtype=float)
        values = float(history.iloc[-1]) + slope * steps
        return ForecastResult(target_dates=_target_dates(request), point_forecast=values)


class SeasonalPersistenceAdapter(ForecastAdapter):
    model_id = "seasonal_365"
    supported_tracks = frozenset({"U"})

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = request.history_target.sort_index()
        dates = _target_dates(request)
        lookup = dates - pd.Timedelta(days=365)
        values = history.reindex(lookup).to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("Seasonal persistence requires all lag-365 values")
        return ForecastResult(target_dates=dates, point_forecast=values)
