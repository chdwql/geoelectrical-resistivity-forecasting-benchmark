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
from toto2 import Toto2Model


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def target_matrix(group: pd.DataFrame) -> np.ndarray:
    selected = group.loc[group["role"].eq("target")]
    if selected.empty:
        raise ValueError("Request has no target history")
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
    parser = argparse.ArgumentParser(description="Run Toto-2 requests from the file contract")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    request_root = args.requests.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_manifest = json.loads((request_root / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest["identity"]["model_id"] != "toto_2_313m":
        raise ValueError("Request set was not prepared for toto_2_313m")
    requests = [
        json.loads(line)
        for line in (request_root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    values = pd.read_parquet(request_root / "values.parquet")
    grouped = {str(key): group for key, group in values.groupby("request_id", sort=False)}
    checkpoint = args.checkpoint.resolve()
    weight_hash = tree_hash(checkpoint)
    model = Toto2Model.from_pretrained(str(checkpoint)).to("cuda").eval()
    patch_size = 32

    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    run_id = f"toto2-{request_manifest['request_set_id']}"
    compatible_batches: dict[tuple[object, ...], list[tuple[dict[str, object], np.ndarray, np.ndarray, bool]]] = defaultdict(list)
    for request in requests:
        if request["track"] == "X":
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": request_manifest["identity"]["protocol_hash"],
                    "model_id": "toto_2_313m",
                    "weight_hash": weight_hash,
                    "window_id": request["window_id"],
                    "series_id": request["series_id"],
                    "track": request["track"],
                    "status": "N/A",
                    "runtime_seconds": 0.0,
                    "peak_memory_mb": 0.0,
                    "failure_reason": "Toto-2.0 does not support exogenous variables",
                }
            )
            continue
        target = target_matrix(grouped[str(request["request_id"])])
        observed = np.isfinite(target)
        target = np.nan_to_num(target, nan=0.0)
        padding = (-target.shape[1]) % patch_size
        if padding:
            target = np.pad(target, ((0, 0), (padding, 0)), constant_values=0.0)
            observed = np.pad(observed, ((0, 0), (padding, 0)), constant_values=False)
        has_missing = not bool(observed.all())
        entry = (request, target, observed, has_missing)
        key = (int(request["horizon"]), target.shape, has_missing)
        compatible_batches[key].append(entry)

    def record_failure(request: dict[str, object], started: float, exc: Exception) -> None:
        tasks.append(
            {
                "run_id": run_id,
                "protocol_hash": request_manifest["identity"]["protocol_hash"],
                "model_id": "toto_2_313m",
                "weight_hash": weight_hash,
                "window_id": request["window_id"],
                "series_id": request["series_id"],
                "track": request["track"],
                "status": "error",
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )

    def run_batch(batch: list[tuple[dict[str, object], np.ndarray, np.ndarray, bool]]) -> None:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            targets = np.stack([entry[1] for entry in batch])
            observed = np.stack([entry[2] for entry in batch])
            inputs = {
                "target": torch.tensor(targets, dtype=torch.float32, device="cuda"),
                "target_mask": torch.tensor(observed, dtype=torch.bool, device="cuda"),
                "series_ids": torch.zeros((len(batch), targets.shape[1]), dtype=torch.long, device="cuda"),
            }
            with torch.no_grad():
                raw_batch = model.forecast(
                    inputs,
                    horizon=int(batch[0][0]["horizon"]),
                    decode_block_size=None,
                    has_missing_values=batch[0][3],
                ).detach().cpu().numpy()
            if raw_batch.shape[1] != len(batch):
                raise RuntimeError(f"Expected {len(batch)} batch outputs, received shape {raw_batch.shape}")
        except Exception as exc:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                run_batch(batch[:midpoint])
                run_batch(batch[midpoint:])
            else:
                record_failure(batch[0][0], started, exc)
            return

        runtime = (time.perf_counter() - started) / len(batch)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        for batch_index, (request, _target, _observed, _has_missing) in enumerate(batch):
            focal = int(request["focal_channel_index"])
            horizon = int(request["horizon"])
            target_dates = pd.date_range(request["target_start"], periods=horizon, freq="D")
            for offset, target_date in enumerate(target_dates):
                predictions.append(
                    {
                        "run_id": run_id,
                        "protocol_hash": request_manifest["identity"]["protocol_hash"],
                        "model_id": "toto_2_313m",
                        "weight_hash": weight_hash,
                        "window_id": request["window_id"],
                        "series_id": request["series_id"],
                        "target_date": target_date,
                        "point_forecast": float(raw_batch[4, batch_index, focal, offset]),
                        "quantiles": json.dumps(
                            {
                                str(level): float(raw_batch[index, batch_index, focal, offset])
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
                    "model_id": "toto_2_313m",
                    "weight_hash": weight_hash,
                    "window_id": request["window_id"],
                    "series_id": request["series_id"],
                    "track": request["track"],
                    "status": "success",
                    "runtime_seconds": runtime,
                    "peak_memory_mb": peak_memory,
                    "failure_reason": None,
                }
            )
        print(f"batch={len(batch)} success", flush=True)

    for compatible in compatible_batches.values():
        for start in range(0, len(compatible), args.batch_size):
            run_batch(compatible[start : start + args.batch_size])

    prediction_columns = [
        "run_id", "protocol_hash", "model_id", "weight_hash", "window_id", "series_id",
        "target_date", "point_forecast", "quantiles", "seed", "status",
    ]
    atomic_parquet(pd.DataFrame(predictions, columns=prediction_columns), output / "predictions.parquet")
    atomic_parquet(pd.DataFrame(tasks), output / "tasks.parquet")
    manifest = {
        "run_id": run_id,
        "model_id": "toto_2_313m",
        "weight_hash": weight_hash,
        "request_manifest": request_manifest,
        "successful_tasks": sum(row["status"] == "success" for row in tasks),
        "na_tasks": sum(row["status"] == "N/A" for row in tasks),
        "failed_tasks": sum(row["status"] not in {"success", "N/A"} for row in tasks),
        "torch_version": torch.__version__,
        "left_padding": "unobserved values to a multiple of 32; information budget remains 730 days",
        "batch_size": args.batch_size,
        "quantile_levels": QUANTILES,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")


if __name__ == "__main__":
    main()
