from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import DLinear, PatchTST


SEEDS = (17, 29, 43)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def matrix(group: pd.DataFrame, role: str, field: str) -> np.ndarray:
    selected = group.loc[group["role"].eq(role)]
    return (
        selected.pivot(index="channel_index", columns="time_index", values=field)
        .sort_index()
        .to_numpy()
    )


def make_model(model_id: str, context: int, horizon: int, channels: int) -> torch.nn.Module:
    if model_id == "dlinear":
        return DLinear(context, horizon, channels)
    if model_id == "patchtst":
        return PatchTST(context, horizon, channels)
    raise ValueError(f"Unknown deep model: {model_id}")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run target-domain deep requests")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    request_root = args.requests.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_manifest = json.loads((request_root / "manifest.json").read_text(encoding="utf-8"))
    model_id = str(request_manifest["identity"]["model_id"])
    run_identity = {
        **request_manifest["identity"],
        "request_identity_hash": request_manifest["identity_hash"],
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seeds": list(SEEDS),
        },
        "environment_code_hash": tree_hash(Path(__file__).resolve().parent),
    }
    identity_hash = stable_hash(run_identity)
    run_id = f"{model_id}-{identity_hash[:16]}"
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") == identity_hash:
            print(json.dumps({"run_id": run_id, "successful_tasks": prior["successful_tasks"]}, indent=2))
            return
        raise RuntimeError(f"Output directory already contains a different run: {output}")
    samples = [
        json.loads(line)
        for line in (request_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    values = pd.read_parquet(request_root / "values.parquet")
    grouped = {str(key): group for key, group in values.groupby("sample_id", sort=False)}
    train_samples = [sample for sample in samples if sample["role"] == "train"]
    predict_samples = [sample for sample in samples if sample["role"] == "predict"]
    if not train_samples or not predict_samples:
        raise ValueError("Request set must contain train and predict samples")

    train_x = np.stack(
        [matrix(grouped[sample["sample_id"]], "input", "value").T for sample in train_samples]
    ).astype(np.float32)
    train_y = np.stack(
        [matrix(grouped[sample["sample_id"]], "label", "value")[0] for sample in train_samples]
    ).astype(np.float32)
    train_mask = np.stack(
        [matrix(grouped[sample["sample_id"]], "label", "observed")[0] for sample in train_samples]
    ).astype(bool)
    context, channels = train_x.shape[1:]
    horizon = train_y.shape[1]
    dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_mask)
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    models: list[torch.nn.Module] = []
    training_records: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats()
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        model = make_model(model_id, context, horizon, channels).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        started = time.perf_counter()
        final_loss = float("nan")
        model.train()
        for _ in range(args.epochs):
            losses: list[float] = []
            for batch_x, batch_y, batch_mask in loader:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
                batch_mask = batch_mask.to("cuda")
                optimizer.zero_grad(set_to_none=True)
                forecast = model(batch_x)
                squared = (forecast - batch_y).square()
                loss = squared[batch_mask].mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            final_loss = float(np.mean(losses))
        model.eval()
        torch.save(model.state_dict(), checkpoints / f"seed-{seed}.pt")
        models.append(model)
        training_records.append(
            {"seed": seed, "epochs": args.epochs, "final_loss": final_loss, "seconds": time.perf_counter() - started}
        )

    weight_hash = tree_hash(checkpoints)
    predictions: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for sample in predict_samples:
        started = time.perf_counter()
        status = "success"
        failure_reason = None
        try:
            input_array = matrix(grouped[sample["sample_id"]], "input", "value").T.astype(np.float32)
            tensor = torch.from_numpy(input_array[None, ...]).to("cuda")
            with torch.no_grad():
                normalized = torch.stack([model(tensor)[0] for model in models]).mean(dim=0).cpu().numpy()
            forecast = float(sample["anchor"]) + float(sample["scale"]) * normalized
            dates = pd.date_range(sample["target_start"], periods=int(sample["horizon"]), freq="D")
            for target_date, value in zip(dates, forecast, strict=True):
                predictions.append(
                    {
                        "run_id": run_id,
                        "protocol_hash": request_manifest["identity"]["protocol_hash"],
                        "model_id": model_id,
                        "weight_hash": weight_hash,
                        "window_id": sample["window_id"],
                        "series_id": sample["series_id"],
                        "target_date": target_date,
                        "point_forecast": float(value),
                        "quantiles": "{}",
                        "seed": None,
                        "status": "success",
                    }
                )
        except Exception as exc:
            status = "error"
            failure_reason = f"{type(exc).__name__}: {exc}"
        tasks.append(
            {
                "run_id": run_id,
                "protocol_hash": request_manifest["identity"]["protocol_hash"],
                "model_id": model_id,
                "weight_hash": weight_hash,
                "window_id": sample["window_id"],
                "series_id": sample["series_id"],
                "track": sample["track"],
                "status": status,
                "runtime_seconds": time.perf_counter() - started,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "failure_reason": failure_reason,
            }
        )

    atomic_parquet(pd.DataFrame(predictions), output / "predictions.parquet")
    atomic_parquet(pd.DataFrame(tasks), output / "tasks.parquet")
    manifest = {
        "run_id": run_id,
        "identity_hash": identity_hash,
        "identity": run_identity,
        "model_id": model_id,
        "weight_hash": weight_hash,
        "request_manifest": request_manifest,
        "training": training_records,
        "successful_tasks": sum(row["status"] == "success" for row in tasks),
        "failed_tasks": sum(row["status"] != "success" for row in tasks),
        "torch_version": torch.__version__,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, output / "manifest.json")
    print(json.dumps({"run_id": run_id, "successful_tasks": manifest["successful_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
