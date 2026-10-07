from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
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


def masked_tensor(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    observed = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0)
    return (
        torch.tensor(filled[None, ...], dtype=torch.float32, device="cuda"),
        torch.tensor(observed[None, ...], dtype=torch.bool, device="cuda"),
    )


def run_once(model: Moirai2Forecast, target: np.ndarray, known: np.ndarray, past: np.ndarray) -> np.ndarray:
    target_value, target_observed = masked_tensor(target)
    known_value, known_observed = masked_tensor(known)
    past_value, past_observed = masked_tensor(past)
    with torch.no_grad():
        result = model(
            past_target=target_value,
            past_observed_target=target_observed,
            past_is_pad=torch.zeros((1, target.shape[0]), dtype=torch.bool, device="cuda"),
            feat_dynamic_real=known_value,
            observed_feat_dynamic_real=known_observed,
            past_feat_dynamic_real=past_value,
            past_observed_feat_dynamic_real=past_observed,
        )
    return result.detach().cpu().numpy()


def run_case(
    module: Moirai2Module,
    *,
    name: str,
    horizon: int,
    target: np.ndarray,
    known: np.ndarray,
    past: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=730,
            target_dim=target.shape[1],
            feat_dynamic_real_dim=known.shape[1],
            past_feat_dynamic_real_dim=past.shape[1],
        ).to("cuda").eval()
        first = run_once(model, target, known, past)
        second = run_once(model, target, known, past)
        return {
            "name": name,
            "status": "success",
            "forecast_shape": list(first.shape),
            "finite": bool(np.isfinite(first).all()),
            "repeat_max_abs_difference": float(np.max(np.abs(first - second))),
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Moirai-2 multivariate/covariate capability probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    started = time.perf_counter()
    module = Moirai2Module.from_pretrained(str(checkpoint)).to("cuda").eval()
    load_seconds = time.perf_counter() - started

    axis = np.linspace(0, 30, 730, dtype=np.float32)
    target = np.stack([np.sin(axis), np.cos(axis * 0.7)], axis=1).astype(np.float32)
    target[120:124, 0] = np.nan
    target[500, 1] = np.nan
    all_axis = np.arange(910, dtype=np.float32)
    known = np.stack(
        [
            np.sin(2 * np.pi * all_axis / 365.25),
            np.cos(2 * np.pi * all_axis / 365.25),
            np.sin(2 * np.pi * all_axis / 7),
            np.cos(2 * np.pi * all_axis / 7),
        ],
        axis=1,
    ).astype(np.float32)
    past = np.stack([np.sin(axis * 0.2), np.cos(axis * 0.11), np.sin(axis * 0.03)], axis=1)
    past[300:310, 0] = np.nan

    torch.cuda.reset_peak_memory_stats()
    cases = [
        run_case(
            module,
            name="univariate_covariates_h180",
            horizon=180,
            target=target[:, :1],
            known=known,
            past=past,
        ),
        run_case(
            module,
            name="multivariate_covariates_h90",
            horizon=90,
            target=target,
            known=known[:820],
            past=past,
        ),
        run_case(
            module,
            name="multivariate_covariates_h180",
            horizon=180,
            target=target,
            known=known,
            past=past,
        ),
    ]
    successful = sum(case["status"] == "success" for case in cases)
    result = {
        "status": "success" if successful == len(cases) else "partial",
        "checkpoint": str(checkpoint),
        "weight_hash": tree_hash(checkpoint),
        "torch_version": torch.__version__,
        "quantile_levels": [float(value) for value in module.quantile_levels],
        "load_seconds": load_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "cases": cases,
    }
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
