from __future__ import annotations

import numpy as np
import pandas as pd


def _paired_arrays(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have identical shapes")
    mask = np.isfinite(actual) & np.isfinite(forecast)
    return actual[mask], forecast[mask]


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = _paired_arrays(actual, forecast)
    return float(np.mean(np.abs(actual - forecast))) if actual.size else float("nan")


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = _paired_arrays(actual, forecast)
    return float(np.sqrt(np.mean(np.square(actual - forecast)))) if actual.size else float("nan")


def mase(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray) -> float:
    history = np.asarray(history, dtype=float)
    if history.size < 2:
        return float("nan")
    adjacent = np.isfinite(history[:-1]) & np.isfinite(history[1:])
    if not adjacent.any():
        return float("nan")
    scale = np.mean(np.abs(history[1:][adjacent] - history[:-1][adjacent]))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(actual, forecast) / float(scale)


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one")
    actual, quantile_forecast = _paired_arrays(actual, quantile_forecast)
    if not actual.size:
        return float("nan")
    error = actual - quantile_forecast
    return float(np.mean(np.maximum(level * error, (level - 1) * error)))


def interval_score(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("actual, lower and upper must have identical shapes")
    mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    actual, lower, upper = actual[mask], lower[mask], upper[mask]
    if np.any(lower > upper):
        raise ValueError("lower interval must not exceed upper interval")
    if not actual.size:
        return float("nan")
    score = upper - lower
    score += (2 / alpha) * (lower - actual) * (actual < lower)
    score += (2 / alpha) * (actual - upper) * (actual > upper)
    return float(np.mean(score))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("actual, lower and upper must have identical shapes")
    mask = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[mask] > upper[mask]):
        raise ValueError("lower interval must not exceed upper interval")
    if not mask.any():
        return float("nan")
    return float(np.mean((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have identical shapes")
    mask = np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[mask] > upper[mask]):
        raise ValueError("lower interval must not exceed upper interval")
    return float(np.mean(upper[mask] - lower[mask])) if mask.any() else float("nan")


def weighted_interval_score(
    actual: np.ndarray,
    median: np.ndarray,
    intervals: list[tuple[float, np.ndarray, np.ndarray]],
) -> float:
    """Weighted interval score using the standard finite central-interval form."""

    actual = np.asarray(actual, dtype=float)
    median = np.asarray(median, dtype=float)
    if actual.shape != median.shape:
        raise ValueError("actual and median must have identical shapes")
    if not intervals:
        return float("nan")
    mask = np.isfinite(actual) & np.isfinite(median)
    prepared: list[tuple[float, np.ndarray, np.ndarray]] = []
    for alpha, lower, upper in intervals:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if not (actual.shape == lower.shape == upper.shape):
            raise ValueError("all WIS arrays must have identical shapes")
        mask &= np.isfinite(lower) & np.isfinite(upper)
        prepared.append((alpha, lower, upper))
    if not mask.any():
        return float("nan")
    actual_valid = actual[mask]
    total = 0.5 * np.abs(actual_valid - median[mask])
    for alpha, lower, upper in prepared:
        lower_valid = lower[mask]
        upper_valid = upper[mask]
        if np.any(lower_valid > upper_valid):
            raise ValueError("lower interval must not exceed upper interval")
        component = upper_valid - lower_valid
        component += (2 / alpha) * (lower_valid - actual_valid) * (actual_valid < lower_valid)
        component += (2 / alpha) * (actual_valid - upper_valid) * (actual_valid > upper_valid)
        total += (alpha / 2) * component
    return float(np.mean(total / (len(prepared) + 0.5)))


def station_equal_mean(
    scores: pd.DataFrame,
    *,
    value_column: str,
    station_column: str = "station",
    series_column: str = "series_id",
    window_column: str = "window_id",
) -> float:
    required = {value_column, station_column, series_column, window_column}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    window = scores.groupby([station_column, series_column, window_column], as_index=False)[value_column].mean()
    series = window.groupby([station_column, series_column], as_index=False)[value_column].mean()
    station = series.groupby(station_column, as_index=False)[value_column].mean()
    return float(station[value_column].mean())
