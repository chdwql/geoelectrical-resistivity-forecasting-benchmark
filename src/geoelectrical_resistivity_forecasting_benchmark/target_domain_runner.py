from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters.lightgbm_direct import LightGBMDirectAdapter, request_feature_frame
from .config import PROJECT_ROOT, file_sha256, load_yaml, stable_hash, tree_hash
from .contracts import ForecastRequest, Track
from .foundation_io import _calendar_frame, _protocol_hash
from .robust_preprocessing import prepare_causal_input
from .protocol_guard import require_model_freeze_for_test
from .runner import _load_daily


def _lookups(daily: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    resistivity = daily.loc[daily["factor"].eq("resistivity")]
    targets = {
        str(series_id): group.set_index("date")["value"].sort_index()
        for series_id, group in resistivity.groupby("series_id")
    }
    panels = {
        str(station): group.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
        for station, group in resistivity.groupby("station")
    }
    environment = daily.loc[daily["factor"].isin(["water_level", "temperature", "rainfall"])]
    environments = {
        str(station): group.pivot_table(index="date", columns="factor", values="value", aggfunc="first")
        for station, group in environment.groupby("station")
    }
    return targets, panels, environments


def training_splits_for(prediction_split: str) -> set[str]:
    return {"train"} if prediction_split == "validation" else {"train", "validation"}


def _filled_frame(
    frame: pd.DataFrame,
    limit: int,
    *,
    input_variant: str,
    q1_window: int,
    q1_threshold: float,
    q1_minimum_history: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: prepare_causal_input(
                frame[column],
                maximum_fill_days=limit,
                variant=input_variant,
                q1_window=q1_window,
                q1_threshold=q1_threshold,
                q1_minimum_history=q1_minimum_history,
            )
            for column in frame.columns
        },
        index=frame.index,
    )


def make_request(
    window: object,
    *,
    track: str,
    targets: dict[str, pd.Series],
    panels: dict[str, pd.DataFrame],
    environments: dict[str, pd.DataFrame],
    maximum_fill: int,
    input_variant: str = "Q0",
    context_days: int = 730,
    q1_window: int = 30,
    q1_threshold: float = 10.0,
    q1_minimum_history: int = 14,
) -> ForecastRequest:
    series_id = str(getattr(window, "series_id"))
    station = series_id.split("/", 1)[0]
    context_dates = pd.date_range(getattr(window, "context_start"), getattr(window, "context_end"), freq="D")
    context_dates = context_dates[-context_days:]
    horizon = int(getattr(window, "horizon"))
    target = prepare_causal_input(
        targets[series_id].reindex(context_dates),
        maximum_fill_days=maximum_fill,
        variant=input_variant,
        q1_window=q1_window,
        q1_threshold=q1_threshold,
        q1_minimum_history=q1_minimum_history,
    )
    covariates: pd.DataFrame | None = None
    if track in {"M", "X"}:
        station_panel = panels[station].reindex(context_dates)
        usable = station_panel.columns[station_panel.notna().mean().ge(0.8)]
        other = [column for column in usable if str(column) != series_id]
        covariates = _filled_frame(
            station_panel[other], maximum_fill,
            input_variant=input_variant,
            q1_window=q1_window,
            q1_threshold=q1_threshold,
            q1_minimum_history=q1_minimum_history,
        )
    if track == "X":
        environment = environments[station].reindex(context_dates)
        environment = _filled_frame(
            environment.reindex(columns=["water_level", "temperature", "rainfall"]), maximum_fill,
            input_variant=input_variant,
            q1_window=q1_window,
            q1_threshold=q1_threshold,
            q1_minimum_history=q1_minimum_history,
        )
        covariates = pd.concat([covariates, environment], axis=1)
    origin = pd.Timestamp(getattr(window, "origin"))
    future_dates = pd.date_range(origin, periods=horizon, freq="D")
    calendar = _calendar_frame(future_dates)
    return ForecastRequest(
        window_id=str(getattr(window, "window_id")),
        series_id=series_id,
        origin=origin.date(),
        horizon=horizon,
        track=Track(track),
        history_target=target,
        history_covariates=covariates,
        future_calendar=calendar,
    )


def run_lightgbm(
    *,
    split: str,
    horizon: int,
    track: str,
    stations: list[str] | None = None,
    limit: int | None = None,
    max_train_windows_per_series: int | None = None,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    output_root: str | Path = "artifacts/benchmark",
    model_specs: list[dict[str, Any]] | None = None,
    input_variant: str = "Q0",
    context_days: int = 730,
    q1_window: int = 30,
    q1_threshold: float = 10.0,
    q1_minimum_history: int = 14,
) -> Path:
    if split not in {"validation", "test_a", "test_b"}:
        raise ValueError("LightGBM supports validation, test_a or test_b splits")
    require_model_freeze_for_test(split)
    if track not in {"U", "M", "X"}:
        raise ValueError("LightGBM supports U, M and X")
    if context_days not in {365, 730}:
        raise ValueError("context_days must be 365 or 730")
    if input_variant not in {"Q0", "Q1"}:
        raise ValueError("input_variant must be Q0 or Q1")
    protocol = load_yaml("configs/protocol_v1.yaml")
    maximum_fill = int(protocol["maximum_causal_fill_days"])
    daily = _load_daily(daily_glob)
    targets, panels, environments = _lookups(daily)
    windows_file = PROJECT_ROOT / windows_path
    windows = pd.read_parquet(windows_file)
    training_splits = training_splits_for(split)
    training = windows.loc[
        windows["split"].isin(training_splits)
        & windows["horizon"].eq(horizon)
        & windows["track"].eq(track)
        & windows["eligible"]
    ].sort_values(["series_id", "origin"])
    if max_train_windows_per_series is not None:
        training = training.groupby("series_id", group_keys=False).tail(max_train_windows_per_series)
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
            selected = selected.assign(
                _station=selected["series_id"].str.split("/").str[0]
            )
            per_station = max(1, math.ceil(limit / len(stations)))
            selected = (
                selected.groupby("_station", sort=False, group_keys=False)
                .head(per_station)
                .head(limit)
                .drop(columns="_station")
            )
        else:
            selected = selected.head(limit)
    if training.empty or selected.empty:
        raise ValueError("No eligible training or prediction windows matched")

    if model_specs is None and split in {"test_a", "test_b"}:
        selected_parameters = dict(load_yaml("configs/target_domain_models.yaml")["lightgbm"]["selected"])
        selected_parameters.pop("validation_station_equal_mase", None)
        specs = [{"model_id": "lightgbm", "parameters": selected_parameters}]
    else:
        specs = model_specs or [{"model_id": "lightgbm", "parameters": {}}]
    output_model_ids = [str(spec["model_id"]) for spec in specs]
    if len(set(output_model_ids)) != len(output_model_ids):
        raise ValueError("model_specs must have unique model_id values")

    identity = {
        "model_id": "lightgbm",
        "protocol_hash": _protocol_hash(),
        "daily_manifest_hash": file_sha256(PROJECT_ROOT / "artifacts/data_v1/daily_q0_manifest.json"),
        "windows_hash": file_sha256(windows_file),
        "code_hash": tree_hash(list((PROJECT_ROOT / "src").rglob("*.py"))),
        "config_hash": file_sha256(PROJECT_ROOT / "configs/target_domain_models.yaml"),
        "split": split,
        "horizon": horizon,
        "track": track,
        "stations": stations,
        "limit": limit,
        "max_train_windows_per_series": max_train_windows_per_series,
        "training_splits": sorted(training_splits),
        "input_variant": input_variant,
        "context_days": context_days,
        "q1_parameters": (
            {"window": q1_window, "threshold": q1_threshold, "minimum_history": q1_minimum_history}
            if input_variant == "Q1"
            else None
        ),
        "train_window_ids": training["window_id"].tolist(),
        "prediction_window_ids": selected["window_id"].tolist(),
    }
    if model_specs is not None:
        identity["model_specs"] = specs
    run_id = f"lightgbm-{stable_hash(identity)[:16]}"
    destination = PROJECT_ROOT / output_root / run_id
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") == stable_hash(identity):
            return destination

    training_frames: list[pd.DataFrame] = []
    for window in training.itertuples(index=False):
        request = make_request(
            window,
            track=track,
            targets=targets,
            panels=panels,
            environments=environments,
            maximum_fill=maximum_fill,
            input_variant=input_variant,
            context_days=context_days,
            q1_window=q1_window,
            q1_threshold=q1_threshold,
            q1_minimum_history=q1_minimum_history,
        )
        target_dates = pd.date_range(window.target_start, window.target_end, freq="D")
        features = request_feature_frame(request)
        raw_label = targets[str(window.series_id)].reindex(target_dates).to_numpy(dtype=float)
        anchor = float(features["target_last_finite"].iloc[0])
        scale = float(features["target_diff_scale"].iloc[0])
        features["label_original"] = raw_label
        features["label"] = (raw_label - anchor) / scale if scale > 0 else float("nan")
        features["origin"] = str(window.origin)
        features["target_date"] = target_dates
        features["window_id"] = str(window.window_id)
        features["series_id"] = str(window.series_id)
        training_frames.append(features)
    train_manifest = pd.concat(training_frames, ignore_index=True)
    config = load_yaml("configs/target_domain_models.yaml")
    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    fit_runtime_seconds: dict[str, float] = {}
    for spec in specs:
        model_id = str(spec["model_id"])
        adapter = LightGBMDirectAdapter()
        fit_started = time.perf_counter()
        adapter.fit(
            train_manifest,
            {"seeds": config["seeds"], "parameters": dict(spec.get("parameters", {}))},
        )
        fit_runtime_seconds[model_id] = time.perf_counter() - fit_started
        for window in selected.itertuples(index=False):
            started = time.perf_counter()
            status = "success"
            failure_reason = None
            try:
                request = make_request(
                    window,
                    track=track,
                    targets=targets,
                    panels=panels,
                    environments=environments,
                    maximum_fill=maximum_fill,
                    input_variant=input_variant,
                    context_days=context_days,
                    q1_window=q1_window,
                    q1_threshold=q1_threshold,
                    q1_minimum_history=q1_minimum_history,
                )
                result = adapter.predict(request)
                for target_date, forecast in zip(result.target_dates, result.point_forecast, strict=True):
                    predictions.append(
                        {
                            "run_id": run_id,
                            "protocol_hash": identity["protocol_hash"],
                            "model_id": model_id,
                            "weight_hash": None,
                            "window_id": str(window.window_id),
                            "series_id": str(window.series_id),
                            "target_date": target_date,
                            "point_forecast": float(forecast),
                            "quantiles": "{}",
                            "seed": None,
                            "status": "success",
                        }
                    )
            except Exception as exc:
                status = "error"
                failure_reason = f"{type(exc).__name__}: {exc}"
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": identity["protocol_hash"],
                    "model_id": model_id,
                    "window_id": str(window.window_id),
                    "series_id": str(window.series_id),
                    "track": track,
                    "status": status,
                    "runtime_seconds": time.perf_counter() - started,
                    "failure_reason": failure_reason,
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "training_manifest.parquet": train_manifest,
        "predictions.parquet": pd.DataFrame(predictions),
        "tasks.parquet": pd.DataFrame(tasks),
    }.items():
        final = destination / name
        temporary = final.with_suffix(final.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, final)
    manifest = {
        "run_id": run_id,
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train_manifest),
        "training_windows": len(training),
        "successful_tasks": sum(task["status"] == "success" for task in tasks),
        "failed_tasks": sum(task["status"] != "success" for task in tasks),
        "fit_runtime_seconds": fit_runtime_seconds,
        "smoke_limited_training": max_train_windows_per_series is not None,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return destination
