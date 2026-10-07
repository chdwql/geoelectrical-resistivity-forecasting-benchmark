from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.adapters import (
    DriftAdapter,
    PersistenceAdapter,
    SeasonalPersistenceAdapter,
)
from geoelectrical_resistivity_forecasting_benchmark.contracts import (
    ForecastRequest,
    PredictionRecord,
    TaskStatus,
    Track,
    WindowRecord,
)
from geoelectrical_resistivity_forecasting_benchmark.metrics import mae, mase, rmse


SEED = 20260918
PROTOCOL_HASH = hashlib.sha256(b"public-synthetic-example-v1").hexdigest()
MODELS = (PersistenceAdapter(), DriftAdapter(), SeasonalPersistenceAdapter())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_daily() -> pd.DataFrame:
    """Create unrestricted daily observations matching the benchmark's daily contract."""

    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2021-01-01", "2024-06-30", freq="D")
    rows: list[pd.DataFrame] = []
    definitions = [
        ("SYN_A", "A", "3211", 100.0, 4.0, 0.003),
        ("SYN_A", "B", "3212", 94.0, 3.0, -0.001),
        ("SYN_B", "A", "3211", 125.0, 5.0, 0.001),
        ("SYN_B", "B", "3212", 116.0, 3.5, 0.002),
    ]
    t = np.arange(len(dates), dtype=float)
    for index, (station, point, item, level, amplitude, slope) in enumerate(definitions):
        seasonal = amplitude * np.sin(2 * np.pi * t / 365.2425 + index * 0.4)
        weekly = 0.25 * np.sin(2 * np.pi * t / 7 + index)
        noise = rng.normal(0, 0.35 + index * 0.03, len(dates))
        value = level + seasonal + weekly + slope * t + noise
        value[dates >= pd.Timestamp("2023-08-15")] += 1.2 if station == "SYN_A" else -0.8
        value[dates == pd.Timestamp("2024-02-14")] += 7.0
        missing = (dates >= pd.Timestamp("2022-06-10")) & (dates <= pd.Timestamp("2022-06-12"))
        value[missing] = np.nan
        flags = np.full(len(dates), "", dtype=object)
        flags[missing] = "synthetic_missing"
        flags[dates == pd.Timestamp("2024-02-14")] = "synthetic_spike"
        frame = pd.DataFrame(
            {
                "date": dates,
                "station": station,
                "point": point,
                "item": item,
                "factor": "apparent_resistivity",
                "value": value,
                "valid_count": np.where(np.isfinite(value), 24, 0),
                "expected_count": 24,
                "coverage": np.where(np.isfinite(value), 1.0, 0.0),
                "quality_flags": flags,
                "source_hash": hashlib.sha256(f"{station}/{point}/{item}".encode()).hexdigest(),
            }
        )
        frame["series_id"] = frame["station"] + "/" + frame["point"] + "/" + frame["item"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_windows(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_id in sorted(daily["series_id"].unique()):
        for origin in (date(2024, 1, 1), date(2024, 4, 1)):
            for horizon in (30, 90):
                origin_ts = pd.Timestamp(origin)
                context_end = origin_ts - pd.Timedelta(days=1)
                context_start = context_end - pd.Timedelta(days=729)
                target_end = origin_ts + pd.Timedelta(days=horizon - 1)
                window_id = hashlib.sha256(f"{series_id}|{origin}|{horizon}|U".encode()).hexdigest()[:20]
                mask = daily.loc[
                    daily["series_id"].eq(series_id)
                    & daily["date"].between(context_start, context_end),
                    "value",
                ].notna().astype("uint8").to_numpy()
                record = WindowRecord(
                    window_id=window_id,
                    series_id=series_id,
                    origin=origin,
                    context_start=context_start.date(),
                    context_end=context_end.date(),
                    target_start=origin,
                    target_end=target_end.date(),
                    horizon=horizon,
                    track=Track.U,
                    split="synthetic_test",
                    mask_hash=hashlib.sha256(mask.tobytes()).hexdigest(),
                    eligible=True,
                )
                rows.append(record.model_dump(mode="json"))
    return pd.DataFrame(rows)


def forecast_and_score(daily: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    indexed = daily.set_index(["series_id", "date"])["value"].sort_index()
    for window in windows.itertuples(index=False):
        series = indexed.loc[window.series_id]
        history = series.loc[pd.Timestamp(window.context_start) : pd.Timestamp(window.context_end)]
        actual = series.reindex(pd.date_range(window.target_start, window.target_end, freq="D"))
        request = ForecastRequest(
            window_id=window.window_id,
            series_id=window.series_id,
            origin=pd.Timestamp(window.origin).date(),
            horizon=int(window.horizon),
            track=Track.U,
            history_target=history,
        )
        for adapter in MODELS:
            run_id = f"synthetic-{adapter.model_id}"
            try:
                result = adapter.predict(request)
                status, reason = TaskStatus.SUCCESS, None
                point = result.point_forecast
            except Exception as exc:  # Explicit failure rows are part of the public contract demo.
                status, reason = TaskStatus.ERROR, f"{type(exc).__name__}: {exc}"
                point = np.full(int(window.horizon), np.nan)
                result_dates = pd.date_range(window.target_start, window.target_end, freq="D")
                result = type("FailedResult", (), {"target_dates": result_dates})()
            for target_date, value in zip(result.target_dates, point, strict=True):
                record = PredictionRecord(
                    run_id=run_id,
                    protocol_hash=PROTOCOL_HASH,
                    model_id=adapter.model_id,
                    weight_hash=None,
                    window_id=window.window_id,
                    target_date=target_date.date(),
                    point_forecast=float(value) if np.isfinite(value) else None,
                    seed=None,
                    status=status,
                    runtime_seconds=None,
                    failure_reason=reason,
                    created_at=datetime(2026, 9, 18),
                )
                dumped = record.model_dump(mode="json")
                dumped["quantiles"] = json.dumps(dumped["quantiles"], sort_keys=True)
                predictions.append(dumped)
            scores.append(
                {
                    "window_id": window.window_id,
                    "series_id": window.series_id,
                    "model_id": adapter.model_id,
                    "origin": window.origin,
                    "horizon": window.horizon,
                    "status": status.value,
                    "mae": mae(actual.to_numpy(), point),
                    "rmse": rmse(actual.to_numpy(), point),
                    "mase": mase(actual.to_numpy(), point, history.to_numpy()),
                    "failure_reason": reason,
                }
            )
    return pd.DataFrame(predictions), pd.DataFrame(scores)


def run_example(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = synthetic_daily()
    windows = build_windows(daily)
    predictions, scores = forecast_and_score(daily, windows)
    outputs = {
        "daily.csv": daily,
        "windows.csv": windows,
        "predictions.csv": predictions,
        "scores.csv": scores,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, date_format="%Y-%m-%d", float_format="%.10g")
    files = {name: sha256(output_dir / name) for name in outputs}
    manifest = {
        "example": "public-synthetic-example-v1",
        "seed": SEED,
        "protocol_hash": PROTOCOL_HASH,
        "contains_restricted_data": False,
        "daily_rows": len(daily),
        "windows": len(windows),
        "prediction_rows": len(predictions),
        "score_rows": len(scores),
        "successful_scores": int(scores["status"].eq("success").sum()),
        "files": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unrestricted synthetic benchmark example.")
    parser.add_argument("--output", type=Path, default=Path("examples/output"))
    args = parser.parse_args()
    print(json.dumps(run_example(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
