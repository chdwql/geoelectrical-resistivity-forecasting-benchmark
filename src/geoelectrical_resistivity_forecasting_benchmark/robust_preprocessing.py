from __future__ import annotations

import pandas as pd

from .preprocessing import causal_forward_fill


def causal_hampel_clip(
    series: pd.Series,
    *,
    window: int,
    threshold: float,
    minimum_history: int,
) -> pd.Series:
    """Clip isolated extremes using robust statistics from strictly earlier values."""

    if window < 2:
        raise ValueError("window must be at least two")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not 2 <= minimum_history <= window:
        raise ValueError("minimum_history must be between two and window")
    ordered = pd.to_numeric(series.sort_index(), errors="coerce").astype(float)
    past = ordered.shift(1)
    center = past.rolling(window=window, min_periods=minimum_history).median()
    absolute_deviation = past.sub(center).abs()
    mad = absolute_deviation.rolling(window=window, min_periods=minimum_history).median()
    robust_scale = 1.4826 * mad
    usable = center.notna() & robust_scale.gt(0) & ordered.notna()
    lower = center - threshold * robust_scale
    upper = center + threshold * robust_scale
    result = ordered.copy()
    result.loc[usable] = ordered.loc[usable].clip(lower=lower.loc[usable], upper=upper.loc[usable])
    return result


def prepare_causal_input(
    series: pd.Series,
    *,
    maximum_fill_days: int,
    variant: str = "Q0",
    q1_window: int = 90,
    q1_threshold: float = 8.0,
    q1_minimum_history: int = 30,
) -> pd.Series:
    filled = causal_forward_fill(series, limit=maximum_fill_days)
    if variant == "Q0":
        return filled
    if variant != "Q1":
        raise ValueError("input variant must be Q0 or Q1")
    return causal_hampel_clip(
        filled,
        window=q1_window,
        threshold=q1_threshold,
        minimum_history=q1_minimum_history,
    )
