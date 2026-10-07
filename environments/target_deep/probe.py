from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from models import DLinear, PatchTST


def exercise(model: torch.nn.Module, values: torch.Tensor) -> dict[str, object]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(values)
    loss = prediction.square().mean()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        first = model(values)
        second = model(values)
    return {
        "shape": list(first.shape),
        "finite": bool(torch.isfinite(first).all()),
        "repeat_max_abs_difference": float((first - second).abs().max()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-domain deep-model synthetic capability probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(17)
    values = torch.randn(4, 730, 5, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    results = {
        "dlinear": exercise(DLinear(730, 180, 5).to("cuda"), values),
        "patchtst": exercise(PatchTST(730, 180, 5).to("cuda"), values),
    }
    result = {
        "status": "success",
        "torch_version": torch.__version__,
        "seed": 17,
        "input_shape": list(values.shape),
        "models": results,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    for record in results.values():
        if record["shape"] != [4, 180] or not record["finite"]:
            raise RuntimeError(f"Deep-model contract failed: {result}")
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
