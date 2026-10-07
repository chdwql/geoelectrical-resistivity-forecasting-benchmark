from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .metrics import (
    interval_coverage,
    interval_width,
    mae,
    mase,
    pinball_loss,
    rmse,
    station_equal_mean,
    weighted_interval_score,
)
from .runner import _load_daily


def _parse_quantiles(value: object) -> dict[float, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        return {}
    return {float(level): float(forecast) for level, forecast in parsed.items()}


def _probabilistic_scores(actual: np.ndarray, quantile_values: pd.Series) -> dict[str, object]:
    parsed = [_parse_quantiles(value) for value in quantile_values]
    levels = sorted({level for row in parsed for level in row})
    empty = {
        "mean_quantile_loss": float("nan"),
        "wis": float("nan"),
        "coverage_80": float("nan"),
        "width_80": float("nan"),
        "coverage_90": float("nan"),
        "width_90": float("nan"),
        "quantile_crossing_rate": float("nan"),
        "quantile_levels": "",
    }
    if not levels:
        return empty
    forecasts = {
        level: np.asarray([row.get(level, np.nan) for row in parsed], dtype=float)
        for level in levels
    }
    raw_ordered = np.column_stack([forecasts[level] for level in levels])
    adjacent = np.isfinite(raw_ordered[:, :-1]) & np.isfinite(raw_ordered[:, 1:])
    crossing_rate = (
        float(np.mean((raw_ordered[:, :-1] > raw_ordered[:, 1:])[adjacent]))
        if adjacent.any()
        else float("nan")
    )
    # Increasing rearrangement is deterministic and does not inspect truth. It
    # preserves the marginal values while making interval metrics well-defined.
    complete_rows = np.isfinite(raw_ordered).all(axis=1)
    raw_ordered[complete_rows] = np.sort(raw_ordered[complete_rows], axis=1)
    forecasts = {level: raw_ordered[:, index] for index, level in enumerate(levels)}
    losses = [pinball_loss(actual, forecasts[level], level) for level in levels]
    finite_losses = [value for value in losses if np.isfinite(value)]

    def central_interval(alpha: float) -> tuple[np.ndarray, np.ndarray] | None:
        level_lookup = {round(level, 10): level for level in levels}
        lower_key = round(alpha / 2, 10)
        upper_key = round(1 - alpha / 2, 10)
        if lower_key not in level_lookup or upper_key not in level_lookup:
            return None
        return forecasts[level_lookup[lower_key]], forecasts[level_lookup[upper_key]]

    result = dict(empty)
    result["mean_quantile_loss"] = (
        float(np.mean(finite_losses)) if finite_losses else float("nan")
    )
    result["quantile_levels"] = json.dumps(levels)
    result["quantile_crossing_rate"] = crossing_rate
    for coverage, alpha in ((80, 0.2), (90, 0.1)):
        interval = central_interval(alpha)
        if interval is None:
            continue
        lower, upper = interval
        result[f"coverage_{coverage}"] = interval_coverage(actual, lower, upper)
        result[f"width_{coverage}"] = interval_width(lower, upper)

    level_lookup = {round(level, 10): level for level in levels}
    if 0.5 in level_lookup:
        intervals: list[tuple[float, np.ndarray, np.ndarray]] = []
        for lower_level in levels:
            if lower_level >= 0.5:
                continue
            upper_key = round(1 - lower_level, 10)
            if upper_key not in level_lookup:
                continue
            lower = forecasts[lower_level]
            upper = forecasts[level_lookup[upper_key]]
            intervals.append((2 * lower_level, lower, upper))
        if intervals:
            result["wis"] = weighted_interval_score(
                actual,
                forecasts[level_lookup[0.5]],
                intervals,
            )
    return result


def _scoring_context(
    daily_glob: str, windows_path: str | Path
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    windows = pd.read_parquet(PROJECT_ROOT / windows_path).set_index("window_id")
    daily = _load_daily(daily_glob)
    target = daily.loc[daily["factor"].eq("resistivity")]
    lookup = {
        series_id: group.set_index("date")["value"].sort_index()
        for series_id, group in target.groupby("series_id")
    }
    return windows, lookup


def _score_loaded_run(
    run_path: Path,
    windows: pd.DataFrame,
    lookup: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(run_path / "predictions.parquet")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    rows: list[dict[str, object]] = []
    for (model_id, window_id, series_id), group in predictions.groupby(
        ["model_id", "window_id", "series_id"], sort=False
    ):
        window = windows.loc[window_id]
        values = lookup[series_id]
        group = group.sort_values("target_date")
        actual = values.reindex(group["target_date"]).to_numpy(dtype=float)
        forecast = group["point_forecast"].to_numpy(dtype=float)
        context_dates = pd.date_range(window["context_start"], window["context_end"], freq="D")
        history = values.reindex(context_dates).to_numpy(dtype=float)
        probability = (
            _probabilistic_scores(actual, group["quantiles"])
            if "quantiles" in group.columns
            else _probabilistic_scores(actual, pd.Series([None] * len(group)))
        )
        rows.append(
            {
                "run_id": group.iloc[0]["run_id"],
                "model_id": model_id,
                "window_id": window_id,
                "series_id": series_id,
                "station": series_id.split("/", 1)[0],
                "split": window["split"],
                "horizon": int(window["horizon"]),
                "target_points": int(pd.Series(actual).notna().sum()),
                "mae": mae(actual, forecast),
                "rmse": rmse(actual, forecast),
                "mase": mase(actual, forecast, history),
                **probability,
            }
        )
    scores = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for model_id, group in scores.groupby("model_id"):
        summaries.append(
            {
                "model_id": model_id,
                "windows": len(group),
                "stations": group["station"].nunique(),
                "station_equal_mase": station_equal_mean(group, value_column="mase"),
                "station_equal_mae": station_equal_mean(group, value_column="mae"),
                "station_equal_rmse": station_equal_mean(group, value_column="rmse"),
                "station_equal_mean_quantile_loss": station_equal_mean(
                    group, value_column="mean_quantile_loss"
                ),
                "station_equal_wis": station_equal_mean(group, value_column="wis"),
                "station_equal_coverage_80": station_equal_mean(
                    group, value_column="coverage_80"
                ),
                "station_equal_width_80": station_equal_mean(group, value_column="width_80"),
                "station_equal_coverage_90": station_equal_mean(
                    group, value_column="coverage_90"
                ),
                "station_equal_width_90": station_equal_mean(group, value_column="width_90"),
                "station_equal_quantile_crossing_rate": station_equal_mean(
                    group, value_column="quantile_crossing_rate"
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    for filename, frame in {"scores.parquet": scores, "summary.csv": summary}.items():
        final_path = run_path / filename
        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        if final_path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, final_path)
    return scores, summary


def score_run(
    run_directory: str | Path,
    *,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_directory)
    if not run_path.is_absolute():
        run_path = PROJECT_ROOT / run_path
    windows, lookup = _scoring_context(daily_glob, windows_path)
    return _score_loaded_run(run_path, windows, lookup)


def score_runs(
    run_directories: list[str | Path],
    *,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
) -> pd.DataFrame:
    """Score several runs while loading restricted truth exactly once."""

    windows, lookup = _scoring_context(daily_glob, windows_path)
    summaries: list[pd.DataFrame] = []
    for directory in run_directories:
        run_path = Path(directory)
        if not run_path.is_absolute():
            run_path = PROJECT_ROOT / run_path
        _, summary = _score_loaded_run(run_path, windows, lookup)
        summaries.append(summary.assign(run_directory=str(run_path.relative_to(PROJECT_ROOT))))
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
