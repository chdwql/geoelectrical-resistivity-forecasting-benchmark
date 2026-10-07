from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import (
    ARIMAAdapter,
    DriftAdapter,
    ETSAdapter,
    FourierRidgeAdapter,
    PersistenceAdapter,
    SeasonalPersistenceAdapter,
)
from .config import PROJECT_ROOT, file_sha256, load_yaml, stable_hash, tree_hash
from .contracts import ForecastRequest, Track
from .robust_preprocessing import prepare_causal_input
from .protocol_guard import require_model_freeze_for_test


LOCAL_ADAPTERS = {
    "persistence": PersistenceAdapter,
    "drift": DriftAdapter,
    "seasonal_365": SeasonalPersistenceAdapter,
    "ets": ETSAdapter,
    "arima": ARIMAAdapter,
    "fourier_ridge": FourierRidgeAdapter,
}


def _frozen_protocol_hash(protocol_path: str | Path, protocol: dict[str, object]) -> str:
    freeze_path = PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json"
    candidate = Path(protocol_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected = freeze.get("config_hashes", {}).get("configs/protocol_v1.yaml")
        if candidate.resolve() == (PROJECT_ROOT / "configs/protocol_v1.yaml").resolve():
            if expected != file_sha256(candidate):
                raise RuntimeError("Frozen protocol file hash no longer matches DATA_PROTOCOL_V1.json")
            return str(freeze["protocol_hash"])
    return stable_hash(protocol)


def _load_daily(glob: str) -> pd.DataFrame:
    paths = sorted(PROJECT_ROOT.glob(glob))
    if not paths:
        raise FileNotFoundError(f"No daily data matched {glob}")
    daily = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["series_id"] = daily[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
    return daily


def run_local_models(
    *,
    split: str,
    horizon: int,
    model_ids: list[str],
    stations: list[str] | None = None,
    limit: int | None = None,
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    protocol_path: str | Path = "configs/protocol_v1.yaml",
    output_root: str | Path = "artifacts/benchmark",
    model_specs: list[dict[str, Any]] | None = None,
    input_variant: str = "Q0",
    context_days: int = 730,
    q1_window: int = 90,
    q1_threshold: float = 8.0,
    q1_minimum_history: int = 30,
) -> Path:
    require_model_freeze_for_test(split)
    selected_statistical: dict[str, dict[str, object]] = {}
    if split in {"test_a", "test_b"}:
        selected = load_yaml("configs/statistical_models.yaml")["selected"]
        selected_statistical = {
            "ets": {
                "trend": bool(selected["ets"]["trend"]),
                "damped_trend": bool(selected["ets"]["damped_trend"]),
            },
            "arima": {"order": tuple(int(value) for value in selected["arima"]["order"])},
            "fourier_ridge": {
                "harmonics": int(selected["fourier_ridge"]["annual_harmonics"]),
                "alpha": float(selected["fourier_ridge"]["alpha"]),
            },
        }
    specs = model_specs or [
        {
            "base_model_id": model_id,
            "model_id": model_id,
            "params": selected_statistical.get(model_id, {}),
        }
        for model_id in model_ids
    ]
    if split in {"test_a", "test_b"} and model_specs is not None:
        for spec in specs:
            base_model_id = str(spec["base_model_id"])
            if (
                base_model_id in selected_statistical
                and dict(spec.get("params", {})) != selected_statistical[base_model_id]
            ):
                raise ValueError(f"Formal test must use the selected {base_model_id} parameters")
    base_model_ids = [str(spec["base_model_id"]) for spec in specs]
    output_model_ids = [str(spec["model_id"]) for spec in specs]
    if len(set(output_model_ids)) != len(output_model_ids):
        raise ValueError("model_specs must have unique output model_id values")
    unknown = set(base_model_ids) - set(LOCAL_ADAPTERS)
    if unknown:
        raise ValueError(f"Unimplemented local adapters: {sorted(unknown)}")
    protocol = load_yaml(protocol_path)
    if context_days not in {365, 730}:
        raise ValueError("context_days must be 365 or 730")
    if input_variant not in {"Q0", "Q1"}:
        raise ValueError("input_variant must be Q0 or Q1")
    protocol_hash = _frozen_protocol_hash(protocol_path, protocol)
    daily = _load_daily(daily_glob)
    target = daily.loc[daily["factor"].eq("resistivity")]
    target_lookup = {
        series_id: group.set_index("date")["value"].sort_index()
        for series_id, group in target.groupby("series_id")
    }
    windows = pd.read_parquet(PROJECT_ROOT / windows_path)
    selected = windows.loc[
        windows["split"].eq(split)
        & windows["horizon"].eq(horizon)
        & windows["track"].eq("U")
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
    identity = {
        "protocol_hash": protocol_hash,
        "code_hash": tree_hash(list((PROJECT_ROOT / "src").rglob("*.py"))),
        "daily_manifest_hash": file_sha256(PROJECT_ROOT / "artifacts/data_v1/daily_q0_manifest.json"),
        "windows_hash": file_sha256(PROJECT_ROOT / windows_path),
        "split": split,
        "horizon": horizon,
        "models": model_ids,
        "stations": stations,
        "limit": limit,
        "input_variant": input_variant,
        "context_days": context_days,
        "q1_parameters": (
            {
                "window": q1_window,
                "threshold": q1_threshold,
                "minimum_history": q1_minimum_history,
            }
            if input_variant == "Q1"
            else None
        ),
        "window_ids": selected["window_id"].tolist(),
    }
    if model_specs is not None:
        identity["model_specs"] = specs
    run_id = f"local-{stable_hash(identity)[:16]}"
    destination = PROJECT_ROOT / output_root / run_id
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity_hash") == stable_hash(identity):
            return destination

    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for spec in specs:
        base_model_id = str(spec["base_model_id"])
        model_id = str(spec["model_id"])
        adapter = LOCAL_ADAPTERS[base_model_id](**dict(spec.get("params", {})))
        for window in selected.itertuples(index=False):
            started = time.perf_counter()
            status = "success"
            failure_reason = None
            try:
                values = target_lookup[window.series_id]
                context_dates = pd.date_range(window.context_start, window.context_end, freq="D")
                context_dates = context_dates[-context_days:]
                request = ForecastRequest(
                    window_id=window.window_id,
                    series_id=window.series_id,
                    origin=window.origin,
                    horizon=int(window.horizon),
                    track=Track.U,
                    history_target=prepare_causal_input(
                        values.reindex(context_dates),
                        maximum_fill_days=int(protocol["maximum_causal_fill_days"]),
                        variant=input_variant,
                        q1_window=q1_window,
                        q1_threshold=q1_threshold,
                        q1_minimum_history=q1_minimum_history,
                    ),
                )
                result = adapter.predict(request)
                for target_date, point_forecast in zip(result.target_dates, result.point_forecast, strict=True):
                    predictions.append(
                        {
                            "run_id": run_id,
                            "protocol_hash": protocol_hash,
                            "model_id": model_id,
                            "weight_hash": None,
                            "window_id": window.window_id,
                            "series_id": window.series_id,
                            "target_date": target_date,
                            "point_forecast": float(point_forecast),
                            "quantiles": "{}",
                            "seed": None,
                            "status": status,
                        }
                    )
            except Exception as exc:
                status = "error"
                failure_reason = f"{type(exc).__name__}: {exc}"
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": protocol_hash,
                    "model_id": model_id,
                    "window_id": window.window_id,
                    "series_id": window.series_id,
                    "status": status,
                    "runtime_seconds": time.perf_counter() - started,
                    "failure_reason": failure_reason,
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "predictions.parquet": pd.DataFrame(predictions),
        "tasks.parquet": pd.DataFrame(tasks),
    }.items():
        final_path = destination / name
        temporary = final_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, final_path)
    manifest = {
        "run_id": run_id,
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "successful_tasks": sum(task["status"] == "success" for task in tasks),
        "failed_tasks": sum(task["status"] != "success" for task in tasks),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return destination
