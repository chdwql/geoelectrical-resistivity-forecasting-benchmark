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
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


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


def masked_tensor(values: np.ndarray | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if values is None:
        return None, None
    observed = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0)
    return (
        torch.tensor(filled, dtype=torch.float32, device="cuda"),
        torch.tensor(observed, dtype=torch.bool, device="cuda"),
    )


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Moirai-2 requests from the file contract")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    request_root = args.requests.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_manifest = json.loads((request_root / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest["identity"]["model_id"] != "moirai_2_small":
        raise ValueError("Request set was not prepared for moirai_2_small")
    requests = [
        json.loads(line)
        for line in (request_root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    values = pd.read_parquet(request_root / "values.parquet")
    grouped = {str(key): group for key, group in values.groupby("request_id", sort=False)}
    checkpoint = args.checkpoint.resolve()
    weight_hash = tree_hash(checkpoint)
    module = Moirai2Module.from_pretrained(str(checkpoint)).to("cuda").eval()
    quantile_levels = [float(value) for value in module.quantile_levels]
    native_horizon = int(module.num_predict_token * module.patch_size)

    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    run_id = f"moirai2-{request_manifest['request_set_id']}"
    compatible_batches: dict[tuple[object, ...], list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]]] = defaultdict(list)
    for request in requests:
        group = grouped[str(request["request_id"])]
        target_channels_time = matrix(group, "target")
        if target_channels_time is None:
            raise ValueError(f"Request has no target history: {request['request_id']}")
        target = target_channels_time.T
        horizon = int(request["horizon"])
        if target.shape[1] > 1 and horizon > native_horizon:
            tasks.append(
                {
                    "run_id": run_id,
                    "protocol_hash": request_manifest["identity"]["protocol_hash"],
                    "model_id": "moirai_2_small",
                    "weight_hash": weight_hash,
                    "window_id": request["window_id"],
                    "series_id": request["series_id"],
                    "track": request["track"],
                    "status": "N/A",
                    "runtime_seconds": 0.0,
                    "peak_memory_mb": 0.0,
                    "failure_reason": (
                        f"Upstream Moirai-2 multivariate recursion fails beyond {native_horizon} steps; "
                        "recorded by the frozen capability probe"
                    ),
                }
            )
            continue
        known_channels_time = matrix(group, "past_future")
        past_channels_time = matrix(group, "past_only")
        known = known_channels_time.T if known_channels_time is not None else None
        past = past_channels_time.T if past_channels_time is not None else None
        entry = (request, target, known, past)
        key = (
            horizon,
            target.shape,
            known.shape if known is not None else None,
            past.shape if past is not None else None,
        )
        compatible_batches[key].append(entry)

    def record_failure(request: dict[str, object], started: float, exc: Exception) -> None:
        tasks.append(
            {
                "run_id": run_id,
                "protocol_hash": request_manifest["identity"]["protocol_hash"],
                "model_id": "moirai_2_small",
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

    def run_batch(batch: list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]]) -> None:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            request0, target0, known0, past0 = batch[0]
            targets = np.stack([entry[1] for entry in batch])
            knowns = np.stack([entry[2] for entry in batch]) if known0 is not None else None
            pasts = np.stack([entry[3] for entry in batch]) if past0 is not None else None
            target_value, target_observed = masked_tensor(targets)
            known_value, known_observed = masked_tensor(knowns)
            past_value, past_observed = masked_tensor(pasts)
            model = Moirai2Forecast(
                module=module,
                prediction_length=int(request0["horizon"]),
                context_length=target0.shape[0],
                target_dim=target0.shape[1],
                feat_dynamic_real_dim=known0.shape[1] if known0 is not None else 0,
                past_feat_dynamic_real_dim=past0.shape[1] if past0 is not None else 0,
            ).to("cuda").eval()
            with torch.no_grad():
                raw_batch = model(
                    past_target=target_value,
                    past_observed_target=target_observed,
                    past_is_pad=torch.zeros((len(batch), target0.shape[0]), dtype=torch.bool, device="cuda"),
                    feat_dynamic_real=known_value,
                    observed_feat_dynamic_real=known_observed,
                    past_feat_dynamic_real=past_value,
                    past_observed_feat_dynamic_real=past_observed,
                ).detach().cpu().numpy()
            if raw_batch.shape[0] != len(batch):
                raise RuntimeError(f"Expected {len(batch)} batch outputs, received {raw_batch.shape[0]}")
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
        for (request, _target, _known, _past), raw in zip(batch, raw_batch, strict=True):
            if raw.ndim == 2:
                raw = raw[:, :, None]
            focal = int(request["focal_channel_index"])
            horizon = int(request["horizon"])
            target_dates = pd.date_range(request["target_start"], periods=horizon, freq="D")
            for offset, target_date in enumerate(target_dates):
                predictions.append(
                    {
                        "run_id": run_id,
                        "protocol_hash": request_manifest["identity"]["protocol_hash"],
                        "model_id": "moirai_2_small",
                        "weight_hash": weight_hash,
                        "window_id": request["window_id"],
                        "series_id": request["series_id"],
                        "target_date": target_date,
                        "point_forecast": float(raw[4, offset, focal]),
                        "quantiles": json.dumps(
                            {
                                str(level): float(raw[index, offset, focal])
                                for index, level in enumerate(quantile_levels)
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
                    "model_id": "moirai_2_small",
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

    atomic_parquet(pd.DataFrame(predictions), output / "predictions.parquet")
    atomic_parquet(pd.DataFrame(tasks), output / "tasks.parquet")
    manifest = {
        "run_id": run_id,
        "model_id": "moirai_2_small",
        "weight_hash": weight_hash,
        "request_manifest": request_manifest,
        "successful_tasks": sum(row["status"] == "success" for row in tasks),
        "na_tasks": sum(row["status"] == "N/A" for row in tasks),
        "failed_tasks": sum(row["status"] not in {"success", "N/A"} for row in tasks),
        "torch_version": torch.__version__,
        "native_multivariate_horizon": native_horizon,
        "batch_size": args.batch_size,
        "quantile_levels": quantile_levels,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")


if __name__ == "__main__":
    main()
