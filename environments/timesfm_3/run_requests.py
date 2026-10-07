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
from timesfm3 import ModelConfig, TimesFM3Evaluator


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TimesFM 3 requests from the file contract")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    request_root = args.requests.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_manifest = json.loads((request_root / "manifest.json").read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in (request_root / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    values = pd.read_parquet(request_root / "values.parquet")
    grouped = {str(key): group for key, group in values.groupby("request_id", sort=False)}
    checkpoint = args.checkpoint.resolve()
    weight_hash = tree_hash(checkpoint)
    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=str(checkpoint),
            per_core_batch_size=args.batch_size,
            device=args.device,
        )
    )
    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    run_id = f"timesfm3-{request_manifest['request_set_id']}"
    prepared: list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]] = []
    grouped_batches: dict[tuple[object, ...], list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]]] = defaultdict(list)
    for request in requests:
        group = grouped[str(request["request_id"])]
        target = matrix(group, "target")
        if target is None:
            raise ValueError(f"Request has no target history: {request['request_id']}")
        past_only = matrix(group, "past_only")
        past_future = matrix(group, "past_future")
        entry = (request, target, past_only, past_future)
        key = (
            target.shape,
            past_only.shape if past_only is not None else None,
            past_future.shape if past_future is not None else None,
        )
        grouped_batches[key].append(entry)

    def record_failure(request: dict[str, object], started: float, exc: Exception) -> None:
        tasks.append(
            {
                "run_id": run_id,
                "protocol_hash": request_manifest["identity"]["protocol_hash"],
                "model_id": "timesfm_3",
                "weight_hash": weight_hash,
                "window_id": request["window_id"],
                "series_id": request["series_id"],
                "track": request["track"],
                "status": "error",
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )

    def run_batch(batch: list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]]) -> None:
        started = time.perf_counter()
        try:
            results = list(
                forecaster.predict_batch(
                    [entry[1] for entry in batch],
                    horizon=int(batch[0][0]["horizon"]),
                    past_only_covariates=[entry[2] for entry in batch]
                    if batch[0][2] is not None
                    else None,
                    past_future_covariates=[entry[3] for entry in batch]
                    if batch[0][3] is not None
                    else None,
                    ts_ids=[str(entry[0]["request_id"]) for entry in batch],
                    return_quantiles=True,
                    use_symmetric_averaging=False,
                )
            )
            if len(results) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} batch outputs, received {len(results)}")
        except Exception as exc:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                run_batch(batch[:midpoint])
                run_batch(batch[midpoint:])
            else:
                record_failure(batch[0][0], started, exc)
            return

        runtime = (time.perf_counter() - started) / len(batch)
        for (request, _target, _past_only, _past_future), result in zip(batch, results, strict=True):
            forecast = np.asarray(result.forecast)
            quantiles = np.asarray(result.quantiles)
            if forecast.ndim == 1:
                forecast = forecast[None, :]
                quantiles = quantiles[None, :, :]
            focal = int(request["focal_channel_index"])
            target_dates = pd.date_range(request["target_start"], periods=int(request["horizon"]), freq="D")
            for offset, target_date in enumerate(target_dates):
                predictions.append(
                    {
                        "run_id": run_id,
                        "protocol_hash": request_manifest["identity"]["protocol_hash"],
                        "model_id": "timesfm_3",
                        "weight_hash": weight_hash,
                        "window_id": request["window_id"],
                        "series_id": request["series_id"],
                        "target_date": target_date,
                        "point_forecast": float(forecast[focal, offset]),
                        "quantiles": json.dumps(
                            {str(level): float(quantiles[focal, offset, index]) for index, level in enumerate(QUANTILES)}
                        ),
                        "seed": None,
                        "status": "success",
                    }
                )
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": request_manifest["identity"]["protocol_hash"],
                    "model_id": "timesfm_3",
                    "weight_hash": weight_hash,
                    "window_id": request["window_id"],
                    "series_id": request["series_id"],
                    "track": request["track"],
                    "status": "success",
                    "runtime_seconds": runtime,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
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
        "model_id": "timesfm_3",
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
