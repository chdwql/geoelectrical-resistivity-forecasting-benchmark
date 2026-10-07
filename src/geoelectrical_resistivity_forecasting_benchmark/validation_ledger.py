from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, file_sha256, stable_hash


def _active_protocol_hash() -> str:
    record = json.loads(
        (PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json").read_text(encoding="utf-8")
    )
    return str(record["protocol_hash"])


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if value is not None and pd.notna(value) else None


def register_validation_runs(
    run_directories: list[str | Path],
    *,
    output_path: str | Path = "artifacts/validation/trials.jsonl",
    expected_protocol_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Idempotently register scored validation runs for G4 selection."""

    protocol_hash = expected_protocol_hash or _active_protocol_hash()
    destination = _resolve(output_path)
    existing: dict[str, dict[str, Any]] = {}
    if destination.is_file():
        for line in destination.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[str(record["trial_id"])] = record

    added: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for directory in run_directories:
        run_path = _resolve(directory)
        manifest_path = run_path / "manifest.json"
        summary_path = run_path / "summary.csv"
        scores_path = run_path / "scores.parquet"
        predictions_path = run_path / "predictions.parquet"
        required = [manifest_path, summary_path, scores_path, predictions_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Scored validation run is incomplete: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_manifest = manifest.get("request_manifest", {})
        identity = manifest.get("identity") or request_manifest.get("identity", {})
        identity_hash = manifest.get("identity_hash") or request_manifest.get("identity_hash")
        if identity.get("split") != "validation":
            raise ValueError(f"Trial ledger only accepts validation runs: {run_path}")
        if identity.get("protocol_hash") != protocol_hash:
            raise ValueError(f"Run does not use active data protocol: {run_path}")
        summary = pd.read_csv(summary_path)
        try:
            recorded_run_path = str(run_path.relative_to(PROJECT_ROOT))
        except ValueError:
            recorded_run_path = str(run_path)
        for row in summary.to_dict(orient="records"):
            model_id = str(row["model_id"])
            trial_identity = {
                "run_identity_hash": identity_hash,
                "model_id": model_id,
            }
            record = {
                "ledger_version": "validation-trial-v2",
                "trial_id": stable_hash(trial_identity)[:24],
                "registered_at": now,
                "data_protocol_hash": protocol_hash,
                "run_id": manifest["run_id"],
                "run_directory": recorded_run_path,
                "model_id": model_id,
                "split": "validation",
                "horizon": int(identity["horizon"]),
                "track": str(identity.get("track", "U")),
                "configuration": identity,
                "successful_tasks": int(manifest.get("successful_tasks", 0)),
                "failed_tasks": int(manifest.get("failed_tasks", 0)),
                "station_equal_mase": float(row["station_equal_mase"]),
                "station_equal_mae": float(row["station_equal_mae"]),
                "station_equal_rmse": float(row["station_equal_rmse"]),
                "station_equal_mean_quantile_loss": _optional_float(
                    row, "station_equal_mean_quantile_loss"
                ),
                "station_equal_wis": _optional_float(row, "station_equal_wis"),
                "station_equal_coverage_80": _optional_float(
                    row, "station_equal_coverage_80"
                ),
                "station_equal_width_80": _optional_float(row, "station_equal_width_80"),
                "station_equal_coverage_90": _optional_float(
                    row, "station_equal_coverage_90"
                ),
                "station_equal_width_90": _optional_float(row, "station_equal_width_90"),
                "station_equal_quantile_crossing_rate": _optional_float(
                    row, "station_equal_quantile_crossing_rate"
                ),
                "run_manifest_hash": file_sha256(manifest_path),
                "summary_hash": file_sha256(summary_path),
                "predictions_hash": file_sha256(predictions_path),
                "scores_hash": file_sha256(scores_path),
                "eligible_for_selection": (
                    int(manifest.get("failed_tasks", 0)) == 0
                    and identity.get("stations") is None
                    and identity.get("limit") is None
                    and identity.get("max_train_windows_per_series") is None
                    and identity.get("input_variant", "Q0") == "Q0"
                    and int(identity.get("context_days", 730)) == 730
                ),
            }
            prior = existing.get(record["trial_id"])
            if prior is not None:
                comparable_prior = {key: value for key, value in prior.items() if key != "registered_at"}
                comparable_new = {key: value for key, value in record.items() if key != "registered_at"}
                if comparable_prior != comparable_new:
                    raise RuntimeError(f"Conflicting trial record: {record['trial_id']}")
                continue
            existing[record["trial_id"]] = record
            added.append(record)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(existing.values(), key=lambda value: (value["registered_at"], value["trial_id"]))
    )
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)
    return added
