from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_yaml


def _future_rolling_median(values: pd.Series, window: int, minimum: int) -> pd.Series:
    return values.iloc[::-1].shift(1).rolling(window, min_periods=minimum).median().iloc[::-1]


def _cluster_candidates(candidates: pd.DataFrame, cluster_days: int) -> pd.DataFrame:
    selected: list[int] = []
    for index, row in candidates.sort_values("score", ascending=False).iterrows():
        if all(abs((row["event_date"] - candidates.loc[other, "event_date"]).days) > cluster_days for other in selected):
            selected.append(index)
    return candidates.loc[selected].sort_values("event_date")


def detect_series_events(
    values: pd.Series,
    *,
    reference_start: str,
    reference_end: str,
    window_days: int,
    minimum_side_observations: int,
    threshold_mad: float,
    cluster_days: int,
) -> pd.DataFrame:
    """Generate retrospective step/spike candidates without modifying observations."""

    values = values.sort_index().astype(float)
    calendar = pd.date_range(values.index.min(), values.index.max(), freq="D")
    values = values.reindex(calendar)
    reference = values.loc[reference_start:reference_end].diff().dropna()
    if reference.empty:
        return pd.DataFrame()
    difference_center = float(reference.median())
    mad = float((reference - difference_center).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return pd.DataFrame()

    pre = values.shift(1).rolling(window_days, min_periods=minimum_side_observations).median()
    post = _future_rolling_median(values, window_days, minimum_side_observations)
    step_score = (post - pre).abs() / scale
    surrounding = pd.concat([pre, post], axis=1).median(axis=1)
    spike_score = (values - surrounding).abs() / scale
    event_type = pd.Series(pd.NA, index=values.index, dtype="string")
    score = pd.Series(np.nan, index=values.index, dtype=float)
    step_mask = values.notna() & pre.notna() & post.notna() & step_score.ge(threshold_mad)
    spike_mask = (
        values.notna()
        & pre.notna()
        & post.notna()
        & ~step_mask
        & spike_score.ge(threshold_mad)
        & step_score.lt(threshold_mad / 2)
    )
    event_type.loc[step_mask] = "step_candidate"
    event_type.loc[spike_mask] = "spike_candidate"
    score.loc[step_mask] = step_score.loc[step_mask]
    score.loc[spike_mask] = spike_score.loc[spike_mask]
    candidates = pd.DataFrame(
        {"event_date": values.index, "event_type": event_type.to_numpy(), "score": score.to_numpy()}
    ).dropna(subset=["event_type"])
    if candidates.empty:
        return candidates
    candidates["reference_scale"] = scale
    candidates["available_at"] = candidates["event_date"] + pd.Timedelta(days=window_days)
    return _cluster_candidates(candidates, cluster_days).reset_index(drop=True)


def build_event_ledger(
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    quality_path: str | Path = "configs/quality_v1.yaml",
    series_path: str | Path = "configs/series_v1.yaml",
    output_path: str | Path = "artifacts/data_v1/quality_event_candidates.csv",
) -> pd.DataFrame:
    paths = sorted(PROJECT_ROOT.glob(daily_glob))
    if not paths:
        raise FileNotFoundError(f"No daily partitions matched: {daily_glob}")
    quality = load_yaml(quality_path)
    series_config = load_yaml(series_path)
    detector: dict[str, Any] = quality["event_ledger"]["retrospective_detector"]
    daily = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["series_id"] = daily[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
    targets = set(str(value) for value in series_config["target_series"])
    daily = daily.loc[daily["series_id"].isin(targets) & daily["factor"].eq("resistivity")]
    records: list[pd.DataFrame] = []
    for series_id, group in daily.groupby("series_id", sort=True):
        values = group.set_index("date")["value"].sort_index()
        events = detect_series_events(
            values,
            reference_start=str(detector["reference_start"]),
            reference_end=str(detector["reference_end"]),
            window_days=int(detector["window_days"]),
            minimum_side_observations=int(detector["minimum_side_observations"]),
            threshold_mad=float(detector["threshold_mad"]),
            cluster_days=int(detector["cluster_days"]),
        )
        if events.empty:
            continue
        station, point, item = series_id.split("/", 2)
        events.insert(0, "series_id", series_id)
        events.insert(1, "station", station)
        events.insert(2, "point", point)
        events.insert(3, "item", item)
        events["detector_version"] = str(detector["version"])
        events["retrospective_only"] = True
        events["review_status"] = "unreviewed_candidate"
        records.append(events)
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result
