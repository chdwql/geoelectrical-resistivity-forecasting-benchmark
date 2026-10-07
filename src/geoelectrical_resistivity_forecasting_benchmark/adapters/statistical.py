from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

from ..contracts import ForecastRequest, ForecastResult
from .base import ForecastAdapter
from .simple import _target_dates


def _regular_history(request: ForecastRequest, minimum_valid: int = 30) -> pd.Series:
    history = request.history_target.sort_index().astype(float)
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History index must be daily timestamps")
    expected = pd.date_range(history.index.min(), history.index.max(), freq="D")
    history = history.reindex(expected)
    first_valid = history.first_valid_index()
    if first_valid is None or history.notna().sum() < minimum_valid:
        raise ValueError(f"At least {minimum_valid} valid observations are required")
    return history.loc[first_valid:]


class ETSAdapter(ForecastAdapter):
    """Linear Gaussian state-space ETS(A,A,N), retaining missing calendar days."""

    model_id = "ets"
    supported_tracks = frozenset({"U"})

    def __init__(self, *, trend: bool = True, damped_trend: bool = True, maxiter: int = 200):
        self.trend = trend
        self.damped_trend = damped_trend if trend else False
        self.maxiter = maxiter

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = _regular_history(request)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                history,
                trend=self.trend,
                damped_trend=self.damped_trend,
                initialization_method="estimated",
            )
            fitted = model.fit(disp=False, maxiter=self.maxiter)
        values = np.asarray(fitted.forecast(request.horizon), dtype=float)
        return ForecastResult(
            target_dates=_target_dates(request),
            point_forecast=values,
            metadata={"converged": bool(fitted.mle_retvals.get("converged", False))},
        )


class ARIMAAdapter(ForecastAdapter):
    model_id = "arima"
    supported_tracks = frozenset({"U"})

    def __init__(self, *, order: tuple[int, int, int] = (2, 1, 0), maxiter: int = 200):
        self.order = order
        self.maxiter = maxiter

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = _regular_history(request)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                history,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(method_kwargs={"maxiter": self.maxiter})
        values = np.asarray(fitted.forecast(request.horizon), dtype=float)
        return ForecastResult(
            target_dates=_target_dates(request),
            point_forecast=values,
            metadata={"order": self.order, "converged": bool(fitted.mle_retvals.get("converged", False))},
        )


def _fourier_design(offsets: np.ndarray, harmonics: int) -> np.ndarray:
    columns = [offsets / 365.25]
    for order in range(1, harmonics + 1):
        angle = 2 * np.pi * order * offsets / 365.25
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(columns)


class FourierRidgeAdapter(ForecastAdapter):
    model_id = "fourier_ridge"
    supported_tracks = frozenset({"U"})

    def __init__(self, *, harmonics: int = 3, alpha: float = 1.0):
        self.harmonics = harmonics
        self.alpha = alpha

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = _regular_history(request)
        observed = history.notna().to_numpy()
        offsets = (history.index - history.index[0]).days.to_numpy(dtype=float)
        model = Ridge(alpha=self.alpha, fit_intercept=True)
        model.fit(_fourier_design(offsets[observed], self.harmonics), history.to_numpy()[observed])
        future_offsets = np.arange(offsets[-1] + 1, offsets[-1] + request.horizon + 1, dtype=float)
        values = model.predict(_fourier_design(future_offsets, self.harmonics)).astype(float)
        return ForecastResult(
            target_dates=_target_dates(request),
            point_forecast=values,
            metadata={"harmonics": self.harmonics, "alpha": self.alpha},
        )
