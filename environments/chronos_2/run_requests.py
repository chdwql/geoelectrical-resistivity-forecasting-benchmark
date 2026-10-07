from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def matrix(group: pd.DataFrame, role: str) -> np.ndarray | None:
    selected = group.loc[group["role"].eq(role)]
    if selected.empty:
        return None
    return (
        selected.pivot(index="channel_index", columns="time_index", values="value")
        .sort_index()
        .to_numpy(dtype=np.float32)
    )


def named_channels(group: pd.DataFrame, role: str) -> dict[str, np.ndarray]:
    selected = group.loc[group["role"].eq(role)]
    channels: dict[str, np.ndarray] = {}
    for channel, rows in selected.groupby("channel", sort=False):
        channels[str(channel)] = rows.sort_values("time_index")["value"].to_numpy(dtype=np.float32)
    return channels


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chronos-2 requests from the file contract")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    request_root = args.requests.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_manifest = json.loads((request_root / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest["identity"]["model_id"] != "chronos_2":
        raise ValueError("Request set was not prepared for chronos_2")
    requests = [
        json.loads(line)
        for line in (request_root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    values = pd.read_parquet(request_root / "values.parquet")
    grouped = {str(key): group for key, group in values.groupby("request_id", sort=False)}
    checkpoint = args.checkpoint.resolve()
    weight_hash = tree_hash(checkpoint)
    pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map=args.device_map)

    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    run_id = f"chronos2-{request_manifest['request_set_id']}"

    grouped_batches: dict[tuple[object, ...], list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for request in requests:
        group = grouped[str(request["request_id"])]
        target = matrix(group, "target")
        if target is None:
            raise ValueError(f"Request has no target history: {request['request_id']}")
        context_length = target.shape[1]
        past_covariates = named_channels(group, "past_only")
        future_covariates: dict[str, np.ndarray] = {}
        for name, values_all in named_channels(group, "past_future").items():
            past_covariates[name] = values_all[:context_length]
            future_covariates[name] = values_all[context_length:]
        inputs = {
            "target": target,
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
        }
        key = (
            target.shape,
            tuple((name, value.shape) for name, value in sorted(past_covariates.items())),
            tuple((name, value.shape) for name, value in sorted(future_covariates.items())),
        )
        grouped_batches[key].append((request, inputs))

    def record_failure(request: dict[str, object], started: float, exc: Exception) -> None:
        tasks.append(
            {
                "run_id": run_id,
                "protocol_hash": request_manifest["identity"]["protocol_hash"],
                "model_id": "chronos_2",
                "weight_hash": weight_hash,
                "window_id": request["window_id"],
                "series_id": request["series_id"],
                "track": request["track"],
                "status": "error",
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )

    def run_batch(batch: list[tuple[dict[str, object], dict[str, object]]]) -> None:
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            quantile_batches, mean_batches = pipeline.predict_quantiles(
                [entry[1] for entry in batch],
                prediction_length=int(batch[0][0]["horizon"]),
                quantile_levels=QUANTILES,
            )
            if len(mean_batches) != len(batch) or len(quantile_batches) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} batch outputs")
        except Exception as exc:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                run_batch(batch[:midpoint])
                run_batch(batch[midpoint:])
            else:
                record_failure(batch[0][0], started, exc)
            return

        runtime = (time.perf_counter() - started) / len(batch)
        for (request, _inputs), quantile_batch, mean_batch in zip(
            batch, quantile_batches, mean_batches, strict=True
        ):
            forecast = as_numpy(mean_batch)
            quantiles = as_numpy(quantile_batch)
            if forecast.ndim == 1:
                forecast = forecast[None, :]
                quantiles = quantiles[None, :, :]
            focal = int(request["focal_channel_index"])
            target_dates = pd.date_range(
                request["target_start"], periods=int(request["horizon"]), freq="D"
            )
            for offset, target_date in enumerate(target_dates):
                predictions.append(
                    {
                        "run_id": run_id,
                        "protocol_hash": request_manifest["identity"]["protocol_hash"],
                        "model_id": "chronos_2",
                        "weight_hash": weight_hash,
                        "window_id": request["window_id"],
                        "series_id": request["series_id"],
                        "target_date": target_date,
                        "point_forecast": float(forecast[focal, offset]),
                        "quantiles": json.dumps(
                            {
                                str(level): float(quantiles[focal, offset, index])
                                for index, level in enumerate(QUANTILES)
                            }
                        ),
                        "seed": None,
                        "status": "success",
                    }
                )
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": request_manifest["identity"]["protocol_hash"],
                    "model_id": "chronos_2",
                    "weight_hash": weight_hash,
                    "window_id": request["window_id"],
                    "series_id": request["series_id"],
                    "track": request["track"],
                    "status": "success",
                    "runtime_seconds": runtime,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
                    "failure_reason": None,
                }
            )
        print(f"batch={len(batch)} success", flush=True)

    for compatible in grouped_batches.values():
        for start in range(0, len(compatible), args.batch_size):
            run_batch(compatible[start : start + args.batch_size])

    atomic_parquet(pd.DataFrame(predictions), output / "predictions.parquet")
    atomic_parquet(pd.DataFrame(tasks), output / "tasks.parquet")
    manifest = {
        "run_id": run_id,
        "model_id": "chronos_2",
        "weight_hash": weight_hash,
        "request_manifest": request_manifest,
        "successful_tasks": sum(row["status"] == "success" for row in tasks),
        "failed_tasks": sum(row["status"] != "success" for row in tasks),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")


if __name__ == "__main__":
    main()
