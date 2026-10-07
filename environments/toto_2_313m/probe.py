from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from toto2 import Toto2Model


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toto-2 multivariate/missing-value capability probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    started = time.perf_counter()
    model = Toto2Model.from_pretrained(str(checkpoint)).to("cuda").eval()
    load_seconds = time.perf_counter() - started
    axis = np.linspace(0, 30, 730, dtype=np.float32)
    values = np.stack([np.sin(axis), np.cos(axis * 0.7)], axis=0)[None, ...]
    mask = np.ones_like(values, dtype=bool)
    mask[0, 0, 120:124] = False
    mask[0, 1, 500] = False
    # Toto requires a context length divisible by its patch size. The six
    # left-padding values are explicitly unobserved, so the information budget
    # remains exactly 730 days.
    values = np.pad(values, ((0, 0), (0, 0), (6, 0)), constant_values=0.0)
    mask = np.pad(mask, ((0, 0), (0, 0), (6, 0)), constant_values=False)
    target = torch.tensor(np.where(mask, values, 0.0), dtype=torch.float32, device="cuda")
    target_mask = torch.tensor(mask, dtype=torch.bool, device="cuda")
    series_ids = torch.zeros((1, 2), dtype=torch.long, device="cuda")
    inputs = {"target": target, "target_mask": target_mask, "series_ids": series_ids}

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        first = model.forecast(inputs, horizon=180, decode_block_size=None, has_missing_values=True)
        inference_seconds = time.perf_counter() - started
        second = model.forecast(inputs, horizon=180, decode_block_size=None, has_missing_values=True)
    first_np = first.detach().cpu().numpy()
    second_np = second.detach().cpu().numpy()
    result = {
        "status": "success",
        "checkpoint": str(checkpoint),
        "weight_hash": tree_hash(checkpoint),
        "torch_version": torch.__version__,
        "forecast_shape": list(first_np.shape),
        "finite": bool(np.isfinite(first_np).all()),
        "repeat_max_abs_difference": float(np.max(np.abs(first_np - second_np))),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    if result["forecast_shape"] != [9, 1, 2, 180]:
        raise RuntimeError(f"Unexpected Toto-2 output contract: {result}")
    if not result["finite"]:
        raise RuntimeError("Toto-2 returned non-finite values")
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
