from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="TimesFM 3.0 synthetic capability probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    config = ModelConfig(checkpoint_path=str(checkpoint), per_core_batch_size=1, device=args.device)
    started = time.perf_counter()
    forecaster = TimesFM3Evaluator(config)
    load_seconds = time.perf_counter() - started
    axis = np.linspace(0, 30, 730, dtype=np.float32)
    context = np.stack([np.sin(axis), np.cos(axis * 0.7)]).astype(np.float32)
    context[0, 120:124] = np.nan
    context[1, 500] = np.nan
    past_only = np.stack(
        [np.sin(axis * 0.2), np.cos(axis * 0.11), np.sin(axis * 0.03)]
    ).astype(np.float32)
    past_only[0, 300:310] = np.nan
    calendar_axis = np.arange(730 + 180, dtype=np.float32)
    past_future = np.stack(
        [
            np.sin(2 * np.pi * calendar_axis / 365.25),
            np.cos(2 * np.pi * calendar_axis / 365.25),
            np.sin(2 * np.pi * calendar_axis / 7),
            np.cos(2 * np.pi * calendar_axis / 7),
        ]
    ).astype(np.float32)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first = list(
        forecaster.predict_batch(
            [context],
            horizon=180,
            past_only_covariates=[past_only],
            past_future_covariates=[past_future],
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )[0]
    inference_seconds = time.perf_counter() - started
    second = list(
        forecaster.predict_batch(
            [context],
            horizon=180,
            past_only_covariates=[past_only],
            past_future_covariates=[past_future],
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )[0]
    forecast = np.asarray(first.forecast)
    quantiles = np.asarray(first.quantiles)
    result = {
        "status": "success",
        "checkpoint": str(checkpoint),
        "weight_hash": tree_hash(checkpoint),
        "device": args.device,
        "torch_version": torch.__version__,
        "forecast_shape": list(forecast.shape),
        "quantile_shape": list(quantiles.shape),
        "finite": bool(np.isfinite(forecast).all() and np.isfinite(quantiles).all()),
        "repeat_max_abs_difference": float(
            max(
                np.max(np.abs(forecast - np.asarray(second.forecast))),
                np.max(np.abs(quantiles - np.asarray(second.quantiles))),
            )
        ),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None
        ),
    }
    if result["forecast_shape"] != [2, 180] or result["quantile_shape"] != [2, 180, 9]:
        raise RuntimeError(f"Unexpected TimesFM output contract: {result}")
    if not result["finite"]:
        raise RuntimeError("TimesFM returned non-finite values")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered)


if __name__ == "__main__":
    main()
