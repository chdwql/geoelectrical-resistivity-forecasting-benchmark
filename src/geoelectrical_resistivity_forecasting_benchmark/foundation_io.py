from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, file_sha256, stable_hash
from .robust_preprocessing import prepare_causal_input
from .protocol_guard import require_model_freeze_for_test
from .runner import _load_daily


def _protocol_hash() -> str:
    freeze = json.loads(
        (PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json").read_text(encoding="utf-8")
    )
    return str(freeze["protocol_hash"])


def _calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    day = dates.dayofyear.to_numpy(dtype=float)
    weekday = dates.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "calendar/year_sin": np.sin(2 * np.pi * day / 365.25),
            "calendar/year_cos": np.cos(2 * np.pi * day / 365.25),
            "calendar/week_sin": np.sin(2 * np.pi * weekday / 7),
            "calendar/week_cos": np.cos(2 * np.pi * weekday / 7),
        },
        index=dates,
    )


def prepare_foundation_requests(
    *,
    model_id: str,
    split: str,
    horizon: int,
    track: str,
    stations: list[str] | None = None,
    limit: int | None = None,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    output_root: str | Path | None = None,
    input_variant: str = "Q0",
    context_days: int = 730,
    q1_window: int = 30,
    q1_threshold: float = 10.0,
    q1_minimum_history: int = 14,
) -> Path:
    require_model_freeze_for_test(split)
    if track not in {"U", "M", "X"}:
        raise ValueError("Foundation-model practical requests support U, M or X")
    if not model_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in model_id):
        raise ValueError("model_id must contain only lowercase letters, digits, underscores or hyphens")
    if context_days not in {365, 730}:
        raise ValueError("context_days must be 365 or 730")
    if input_variant not in {"Q0", "Q1"}:
        raise ValueError("input_variant must be Q0 or Q1")
    output_root = output_root or f"artifacts/model_requests/{model_id}"
    daily = _load_daily(daily_glob)
    windows_file = PROJECT_ROOT / windows_path
    windows = pd.read_parquet(windows_file)
    selected = windows.loc[
        windows["split"].eq(split)
        & windows["horizon"].eq(horizon)
        & windows["track"].eq(track)
        & windows["eligible"]
    ].sort_values(["series_id", "origin"])
    if stations:
        selected = selected.loc[selected["series_id"].str.split("/").str[0].isin(stations)]
    if limit is not None:
        if stations and len(stations) > 1:
            per_station = max(1, math.ceil(limit / len(stations)))
            selected = (
                selected.assign(_station=selected["series_id"].str.split("/").str[0])
                .groupby("_station", sort=False, group_keys=False)
                .head(per_station)
                .head(limit)
                .drop(columns="_station")
            )
        else:
            selected = selected.head(limit)
    if selected.empty:
        raise ValueError("No eligible windows matched the request selection")

    daily_manifest = PROJECT_ROOT / "artifacts/data_v1/daily_q0_manifest.json"
    identity = {
        "model_id": model_id,
        "protocol_hash": _protocol_hash(),
        "daily_manifest_hash": file_sha256(daily_manifest),
        "windows_hash": file_sha256(windows_file),
        "split": split,
        "horizon": horizon,
        "track": track,
        "stations": stations,
        "input_variant": input_variant,
        "context_days": context_days,
        "q1_parameters": (
            {"window": q1_window, "threshold": q1_threshold, "minimum_history": q1_minimum_history}
            if input_variant == "Q1"
            else None
        ),
        "window_ids": selected["window_id"].tolist(),
    }
    request_set_id = stable_hash(identity)[:20]
    destination = PROJECT_ROOT / output_root / request_set_id
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") == stable_hash(identity):
            return destination

    maximum_fill = 3
    target_daily = daily.loc[daily["factor"].eq("resistivity")]
    target_lookup = {
        series_id: group.set_index("date")["value"].sort_index()
        for series_id, group in target_daily.groupby("series_id")
    }
    station_targets = {
        str(station): group.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
        for station, group in target_daily.groupby("station")
    }
    environment = daily.loc[daily["factor"].isin(["water_level", "temperature", "rainfall"])]
    station_environment = {
        str(station): group.pivot_table(index="date", columns="factor", values="value", aggfunc="first")
        for station, group in environment.groupby("station")
    }
    value_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    for window in selected.itertuples(index=False):
        station = str(window.series_id).split("/", 1)[0]
        context_dates = pd.date_range(window.context_start, window.context_end, freq="D")
        context_dates = context_dates[-context_days:]
        target_dates = pd.date_range(window.target_start, window.target_end, freq="D")
        all_dates = context_dates.append(target_dates)
        if track == "U":
            target_channels = [str(window.series_id)]
        else:
            panel = station_targets[station].reindex(context_dates)
            usable = panel.columns[panel.notna().mean().ge(0.8)].tolist()
            target_channels = [str(window.series_id), *sorted(set(usable) - {str(window.series_id)})]
        for channel_index, channel in enumerate(target_channels):
            values = prepare_causal_input(
                target_lookup[channel].reindex(context_dates),
                maximum_fill_days=maximum_fill,
                variant=input_variant,
                q1_window=q1_window,
                q1_threshold=q1_threshold,
                q1_minimum_history=q1_minimum_history,
            ).to_numpy(dtype=float)
            for time_index, value in enumerate(values):
                value_rows.append(
                    {
                        "request_id": str(window.window_id),
                        "role": "target",
                        "channel": channel,
                        "channel_index": channel_index,
                        "time_index": time_index,
                        "value": value,
                    }
                )
        if track == "X":
            covariates = station_environment[station].reindex(context_dates)
            for channel_index, factor in enumerate(("water_level", "temperature", "rainfall")):
                values = prepare_causal_input(
                    covariates[factor],
                    maximum_fill_days=maximum_fill,
                    variant=input_variant,
                    q1_window=q1_window,
                    q1_threshold=q1_threshold,
                    q1_minimum_history=q1_minimum_history,
                ).to_numpy(dtype=float)
                for time_index, value in enumerate(values):
                    value_rows.append(
                        {
                            "request_id": str(window.window_id),
                            "role": "past_only",
                            "channel": factor,
                            "channel_index": channel_index,
                            "time_index": time_index,
                            "value": value,
                        }
                    )
        calendar = _calendar_frame(all_dates)
        for channel_index, channel in enumerate(calendar.columns):
            for time_index, value in enumerate(calendar[channel].to_numpy(dtype=float)):
                value_rows.append(
                    {
                        "request_id": str(window.window_id),
                        "role": "past_future",
                        "channel": channel,
                        "channel_index": channel_index,
                        "time_index": time_index,
                        "value": value,
                    }
                )
        request_rows.append(
            {
                "request_id": str(window.window_id),
                "window_id": str(window.window_id),
                "series_id": str(window.series_id),
                "origin": str(window.origin),
                "target_start": str(window.target_start),
                "horizon": int(window.horizon),
                "track": track,
                "focal_channel_index": 0,
                "target_channels": target_channels,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    values_path = destination / "values.parquet"
    temporary_values = values_path.with_suffix(".parquet.tmp")
    pd.DataFrame(value_rows).to_parquet(temporary_values, index=False)
    os.replace(temporary_values, values_path)
    requests_path = destination / "requests.jsonl"
    temporary_requests = requests_path.with_suffix(".jsonl.tmp")
    temporary_requests.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in request_rows) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_requests, requests_path)
    manifest = {
        "request_set_id": request_set_id,
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "requests": len(request_rows),
        "values_sha256": file_sha256(values_path),
        "requests_sha256": file_sha256(requests_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return destination


def prepare_timesfm_requests(**kwargs: object) -> Path:
    """Backward-compatible wrapper for the original TimesFM-specific CLI."""

    return prepare_foundation_requests(model_id="timesfm_3", **kwargs)
