from __future__ import annotations

import pandas as pd


def causal_forward_fill(series: pd.Series, limit: int = 3) -> pd.Series:
    """Fill short gaps using only values available at or before each timestamp."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    ordered = series.sort_index()
    if limit == 0:
        return ordered.copy()
    return ordered.ffill(limit=limit)
