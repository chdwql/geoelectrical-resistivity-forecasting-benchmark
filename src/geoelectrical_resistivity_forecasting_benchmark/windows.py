from __future__ import annotations

from datetime import date

import pandas as pd

from .config import stable_hash


def origins_for_split(start: date, end: date, horizon: int) -> pd.DatetimeIndex:
    last_origin = pd.Timestamp(end) - pd.Timedelta(days=horizon - 1)
    if last_origin < pd.Timestamp(start):
        return pd.DatetimeIndex([])
    if horizon in {1, 7}:
        origins = pd.date_range(start, last_origin, freq="W-MON")
    elif horizon in {30, 90}:
        origins = pd.date_range(start, last_origin, freq="MS")
    elif horizon == 180:
        origins = pd.date_range(start, last_origin, freq="2QS-JAN")
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return origins


def generate_windows(
    daily: pd.DataFrame,
    *,
    splits: dict[str, tuple[date, date]],
    horizons: list[int],
    tracks: list[str],
    context_days: int,
    minimum_target_coverage: float,
    minimum_context_coverage: float = 0.8,
    minimum_auxiliary_context_coverage: float = 0.8,
    minimum_oracle_future_coverage: float = 0.9,
    target_series_ids: list[str] | None = None,
) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["series_id"] = daily[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
    target = daily.loc[daily["factor"].eq("resistivity")].copy()
    target["series_id"] = target[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
    if target_series_ids is not None:
        requested = set(target_series_ids)
        available = set(target["series_id"])
        missing = requested - available
        if missing:
            raise ValueError(f"Configured target series are absent from daily data: {sorted(missing)}")
        target = target.loc[target["series_id"].isin(requested)].copy()
    station_targets = {
        str(station): group.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
        for station, group in daily.loc[daily["factor"].eq("resistivity")].assign(
            series_id=lambda frame: frame[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
        ).groupby("station")
    }
    station_environment = {
        str(station): group.loc[group["factor"].isin(["water_level", "temperature", "rainfall"])]
        .pivot_table(index="date", columns="factor", values="value", aggfunc="first")
        for station, group in daily.groupby("station")
    }
    rows: list[dict[str, object]] = []
    for series_id, series in target.groupby("series_id", sort=True):
        station = str(series.iloc[0]["station"])
        values = series.set_index("date")["value"].sort_index()
        first_date = values.index.min()
        for split, (split_start, split_end) in splits.items():
            for horizon in horizons:
                for origin in origins_for_split(split_start, split_end, horizon):
                    target_dates = pd.date_range(origin, periods=horizon, freq="D")
                    mask = values.reindex(target_dates).notna().to_numpy()
                    coverage = float(mask.mean())
                    context_start = origin - pd.Timedelta(days=context_days)
                    context_dates = pd.date_range(context_start, origin - pd.Timedelta(days=1), freq="D")
                    base_reasons: list[str] = []
                    if context_start < first_date:
                        base_reasons.append("insufficient_context_span")
                    target_context_coverage = float(values.reindex(context_dates).notna().mean())
                    if target_context_coverage < minimum_context_coverage:
                        base_reasons.append("insufficient_target_context")
                    if coverage < minimum_target_coverage:
                        base_reasons.append("insufficient_target_coverage")
                    mask_hash = stable_hash(mask.astype(int).tolist())
                    for track in tracks:
                        reasons = list(base_reasons)
                        target_panel = station_targets[station].reindex(context_dates)
                        usable_resistivity = int(
                            (target_panel.notna().mean() >= minimum_context_coverage).sum()
                        )
                        if track in {"M", "X", "O"} and usable_resistivity < 2:
                            reasons.append("insufficient_station_resistivity_panel")
                        if track in {"X", "O"}:
                            environment = station_environment.get(station, pd.DataFrame()).reindex(context_dates)
                            for factor in ("water_level", "temperature", "rainfall"):
                                factor_coverage = (
                                    float(environment[factor].notna().mean())
                                    if factor in environment.columns
                                    else 0.0
                                )
                                if factor_coverage < minimum_auxiliary_context_coverage:
                                    reasons.append(f"insufficient_{factor}_context")
                        if track == "O":
                            environment = station_environment.get(station, pd.DataFrame()).reindex(target_dates)
                            for factor in ("water_level", "temperature", "rainfall"):
                                factor_coverage = (
                                    float(environment[factor].notna().mean())
                                    if factor in environment.columns
                                    else 0.0
                                )
                                if factor_coverage < minimum_oracle_future_coverage:
                                    reasons.append(f"insufficient_future_{factor}")
                        identity = {
                            "series_id": series_id,
                            "origin": origin.date().isoformat(),
                            "horizon": horizon,
                            "track": track,
                            "split": split,
                            "mask_hash": mask_hash,
                        }
                        rows.append(
                            {
                                "window_id": stable_hash(identity)[:24],
                                "series_id": series_id,
                                "origin": origin.date(),
                                "context_start": context_start.date(),
                                "context_end": (origin - pd.Timedelta(days=1)).date(),
                                "target_start": origin.date(),
                                "target_end": target_dates[-1].date(),
                                "horizon": horizon,
                                "track": track,
                                "split": split,
                                "mask_hash": mask_hash,
                                "eligible": not reasons,
                                "exclusion_reason": "|".join(dict.fromkeys(reasons)) or None,
                            }
                        )
    return pd.DataFrame(rows)
