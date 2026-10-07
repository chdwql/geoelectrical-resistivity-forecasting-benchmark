from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, file_sha256, stable_hash
from .foundation_io import _protocol_hash
from .protocol_guard import require_model_freeze_for_test
from .runner import _load_daily
from .target_domain_runner import _lookups, make_request, training_splits_for


def _causal_mean_fill(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    observed = numeric.notna().to_numpy()
    past_mean = numeric.expanding(min_periods=1).mean().shift(1)
    filled = numeric.fillna(past_mean).fillna(0.0).to_numpy(dtype=np.float32)
    return filled, observed


def _select_balanced(frame: pd.DataFrame, stations: list[str] | None, limit: int | None) -> pd.DataFrame:
    selected = frame
    if stations:
        selected = selected.loc[selected["series_id"].str.split("/").str[0].isin(stations)]
    if limit is None:
        return selected
    if stations and len(stations) > 1:
        per_station = max(1, math.ceil(limit / len(stations)))
        return (
            selected.assign(_station=selected["series_id"].str.split("/").str[0])
            .groupby("_station", sort=False, group_keys=False)
            .head(per_station)
            .head(limit)
            .drop(columns="_station")
        )
    return selected.head(limit)


def prepare_deep_requests(
    *,
    model_id: str,
    split: str,
    horizon: int,
    track: str,
    stations: list[str] | None = None,
    limit: int | None = None,
    max_train_windows_per_series: int | None = None,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    output_root: str | Path = "artifacts/model_requests/target_deep",
    input_variant: str = "Q0",
    context_days: int = 730,
    q1_window: int = 30,
    q1_threshold: float = 10.0,
    q1_minimum_history: int = 14,
) -> Path:
    if model_id not in {"dlinear", "patchtst"}:
        raise ValueError("Deep request model must be dlinear or patchtst")
    if split not in {"validation", "test_a", "test_b"}:
        raise ValueError("Deep models support validation, test_a or test_b splits")
    require_model_freeze_for_test(split)
    if track not in {"U", "M", "X"}:
        raise ValueError("Deep models support U, M and X")
    if context_days not in {365, 730}:
        raise ValueError("context_days must be 365 or 730")
    if input_variant not in {"Q0", "Q1"}:
        raise ValueError("input_variant must be Q0 or Q1")
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
    prediction = windows.loc[
        windows["split"].eq(split)
        & windows["horizon"].eq(horizon)
        & windows["track"].eq(track)
        & windows["eligible"]
    ].sort_values(["series_id", "origin"])
    prediction = _select_balanced(prediction, stations, limit)
    if training.empty or prediction.empty:
        raise ValueError("No eligible deep training or prediction windows matched")

    identity = {
        "model_id": model_id,
        "protocol_hash": _protocol_hash(),
        "daily_manifest_hash": file_sha256(PROJECT_ROOT / "artifacts/data_v1/daily_q0_manifest.json"),
        "windows_hash": file_sha256(windows_file),
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
        "prediction_window_ids": prediction["window_id"].tolist(),
    }
    request_set_id = stable_hash(identity)[:20]
    destination = PROJECT_ROOT / output_root / request_set_id
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") == stable_hash(identity):
            return destination

    samples: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    maximum_fill = 3
    for role, frame in (("train", training), ("predict", prediction)):
        for window in frame.itertuples(index=False):
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
            channel_series = [request.history_target]
            channel_names = ["target"]
            if track in {"M", "X"}:
                covariates = request.history_covariates
                resistivity_columns = [
                    column
                    for column in covariates.columns
                    if str(column) not in {"water_level", "temperature", "rainfall"}
                ]
                channel_series.append(covariates[resistivity_columns].mean(axis=1))
                channel_names.append("station_panel_mean")
            if track == "X":
                for factor in ("water_level", "temperature", "rainfall"):
                    channel_series.append(request.history_covariates[factor])
                    channel_names.append(factor)

            target_finite = request.history_target.dropna()
            anchor = float(target_finite.iloc[-1])
            differences = request.history_target.diff().abs().dropna()
            scale = float(differences.mean())
            if not np.isfinite(scale) or scale <= 0:
                continue
            sample_id = f"{role}:{window.window_id}"
            for channel_index, (channel_name, series) in enumerate(zip(channel_names, channel_series, strict=True)):
                values, observed = _causal_mean_fill(series)
                if channel_index == 0:
                    values = (values - anchor) / scale
                else:
                    finite = values[observed]
                    center = float(finite.mean()) if len(finite) else 0.0
                    spread = float(finite.std()) if len(finite) else 1.0
                    values = (values - center) / max(spread, 1e-6)
                for time_index, (value, is_observed) in enumerate(zip(values, observed, strict=True)):
                    value_rows.append(
                        {
                            "sample_id": sample_id,
                            "role": "input",
                            "channel": channel_name,
                            "channel_index": channel_index,
                            "time_index": time_index,
                            "value": float(value),
                            "observed": bool(is_observed),
                        }
                    )
            if role == "train":
                target_dates = pd.date_range(window.target_start, window.target_end, freq="D")
                labels = targets[str(window.series_id)].reindex(target_dates)
                for time_index, value in enumerate(labels):
                    value_rows.append(
                        {
                            "sample_id": sample_id,
                            "role": "label",
                            "channel": "target",
                            "channel_index": 0,
                            "time_index": time_index,
                            "value": (float(value) - anchor) / scale if pd.notna(value) else 0.0,
                            "observed": bool(pd.notna(value)),
                        }
                    )
            samples.append(
                {
                    "sample_id": sample_id,
                    "role": role,
                    "window_id": str(window.window_id),
                    "series_id": str(window.series_id),
                    "origin": str(window.origin),
                    "target_start": str(window.target_start),
                    "horizon": horizon,
                    "track": track,
                    "anchor": anchor,
                    "scale": scale,
                    "channels": channel_names,
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    values_path = destination / "values.parquet"
    temporary_values = values_path.with_suffix(".parquet.tmp")
    pd.DataFrame(value_rows).to_parquet(temporary_values, index=False)
    os.replace(temporary_values, values_path)
    samples_path = destination / "samples.jsonl"
    temporary_samples = samples_path.with_suffix(".jsonl.tmp")
    temporary_samples.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in samples) + "\n", encoding="utf-8"
    )
    os.replace(temporary_samples, samples_path)
    manifest = {
        "request_set_id": request_set_id,
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "train_samples": sum(row["role"] == "train" for row in samples),
        "prediction_samples": sum(row["role"] == "predict" for row in samples),
        "values_sha256": file_sha256(values_path),
        "samples_sha256": file_sha256(samples_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return destination
