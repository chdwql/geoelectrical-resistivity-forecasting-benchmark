from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
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


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos-2 multivariate/covariate capability probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map="cuda")
    load_seconds = time.perf_counter() - started

    axis = np.linspace(0, 30, 730, dtype=np.float32)
    target = np.stack([np.sin(axis), np.cos(axis * 0.7)]).astype(np.float32)
    target[0, 120:124] = np.nan
    target[1, 500] = np.nan
    calendar_axis = np.arange(910, dtype=np.float32)
    past_covariates = {
        "water_level": np.sin(axis * 0.2).astype(np.float32),
        "temperature": np.cos(axis * 0.11).astype(np.float32),
        "rainfall": np.sin(axis * 0.03).astype(np.float32),
        "year_sin": np.sin(2 * np.pi * calendar_axis[:730] / 365.25).astype(np.float32),
        "year_cos": np.cos(2 * np.pi * calendar_axis[:730] / 365.25).astype(np.float32),
        "week_sin": np.sin(2 * np.pi * calendar_axis[:730] / 7).astype(np.float32),
        "week_cos": np.cos(2 * np.pi * calendar_axis[:730] / 7).astype(np.float32),
    }
    past_covariates["water_level"][300:310] = np.nan
    future_covariates = {
        "year_sin": np.sin(2 * np.pi * calendar_axis[730:] / 365.25).astype(np.float32),
        "year_cos": np.cos(2 * np.pi * calendar_axis[730:] / 365.25).astype(np.float32),
        "week_sin": np.sin(2 * np.pi * calendar_axis[730:] / 7).astype(np.float32),
        "week_cos": np.cos(2 * np.pi * calendar_axis[730:] / 7).astype(np.float32),
    }
    inputs = [{"target": target, "past_covariates": past_covariates, "future_covariates": future_covariates}]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first_quantiles, first_mean = pipeline.predict_quantiles(
        inputs, prediction_length=180, quantile_levels=QUANTILES
    )
    inference_seconds = time.perf_counter() - started
    second_quantiles, second_mean = pipeline.predict_quantiles(
        inputs, prediction_length=180, quantile_levels=QUANTILES
    )
    mean = as_numpy(first_mean[0])
    quantiles = as_numpy(first_quantiles[0])
    result = {
        "status": "success",
        "checkpoint": str(checkpoint),
        "weight_hash": tree_hash(checkpoint),
        "torch_version": torch.__version__,
        "forecast_shape": list(mean.shape),
        "quantile_shape": list(quantiles.shape),
        "finite": bool(np.isfinite(mean).all() and np.isfinite(quantiles).all()),
        "repeat_max_abs_difference": float(
            max(
                np.max(np.abs(mean - as_numpy(second_mean[0]))),
                np.max(np.abs(quantiles - as_numpy(second_quantiles[0]))),
            )
        ),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    if result["forecast_shape"] != [2, 180] or result["quantile_shape"] != [2, 180, 9]:
        raise RuntimeError(f"Unexpected Chronos-2 output contract: {result}")
    if not result["finite"]:
        raise RuntimeError("Chronos-2 returned non-finite values")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, output)
    print(rendered)


if __name__ == "__main__":
    main()
