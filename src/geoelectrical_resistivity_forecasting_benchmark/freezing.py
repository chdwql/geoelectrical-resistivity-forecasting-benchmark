from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, file_sha256, load_yaml, stable_hash, tree_hash


DATA_PROTOCOL_CONFIGS = [
    Path("configs/data_v1.yaml"),
    Path("configs/quality_v1.yaml"),
    Path("configs/protocol_v1.yaml"),
    Path("configs/series_v1.yaml"),
]
DATA_PROTOCOL_ARTIFACTS = [
    Path("artifacts/data_v1/source_audit.csv"),
    Path("artifacts/data_v1/raw_acquisition_audit.csv"),
    Path("artifacts/data_v1/daily_q0_manifest.json"),
    Path("artifacts/data_v1/series_catalog.csv"),
    Path("artifacts/data_v1/windows.parquet"),
    Path("artifacts/data_v1/quality_event_candidates.csv"),
]
DATA_PIPELINE_CODE = [
    Path("src/geoelectrical_resistivity_forecasting_benchmark/aggregation.py"),
    Path("src/geoelectrical_resistivity_forecasting_benchmark/config.py"),
    Path("src/geoelectrical_resistivity_forecasting_benchmark/preprocessing.py"),
    Path("src/geoelectrical_resistivity_forecasting_benchmark/quality_events.py"),
    Path("src/geoelectrical_resistivity_forecasting_benchmark/source_data.py"),
    Path("src/geoelectrical_resistivity_forecasting_benchmark/windows.py"),
]
MODEL_PROTOCOL_CONFIGS = [
    Path("configs/model_registry.yaml"),
    Path("configs/statistical_models.yaml"),
    Path("configs/target_domain_models.yaml"),
    Path("configs/robustness_v1.yaml"),
]
FOUNDATION_VALIDATION_MATRIX = {
    ("timesfm_3", "U", 90),
    ("timesfm_3", "M", 90),
    ("timesfm_3", "X", 90),
    ("chronos_2", "U", 90),
    ("chronos_2", "M", 90),
    ("chronos_2", "X", 90),
    ("moirai_2_small", "U", 90),
    ("toto_2_313m", "U", 90),
    ("toto_2_313m", "M", 90),
}


def model_code_paths() -> list[Path]:
    return sorted(
        [*(PROJECT_ROOT / "src").rglob("*.py"), *(PROJECT_ROOT / "scripts/wsl").glob("*.sh")]
    )


def isolated_environment_paths() -> list[Path]:
    return sorted(
        path
        for root in (PROJECT_ROOT / "environments").iterdir()
        if root.is_dir()
        for path in (root / "pyproject.toml", root / "uv.lock", root / "run_requests.py")
        if path.is_file()
    )


def build_data_protocol_freeze() -> dict[str, Any]:
    required = [*DATA_PROTOCOL_CONFIGS, *DATA_PROTOCOL_ARTIFACTS, *DATA_PIPELINE_CODE]
    missing = [str(path) for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; required files are missing: {missing}")
    config_hashes = {str(path): file_sha256(PROJECT_ROOT / path) for path in DATA_PROTOCOL_CONFIGS}
    artifact_hashes = {str(path): file_sha256(PROJECT_ROOT / path) for path in DATA_PROTOCOL_ARTIFACTS}
    code_paths = [PROJECT_ROOT / path for path in DATA_PIPELINE_CODE]
    windows = pd.read_parquet(PROJECT_ROOT / "artifacts/data_v1/windows.parquet")
    catalog = pd.read_csv(
        PROJECT_ROOT / "artifacts/data_v1/series_catalog.csv",
        dtype={"station": str, "point": str, "item": str},
    )
    events = pd.read_csv(PROJECT_ROOT / "artifacts/data_v1/quality_event_candidates.csv")
    raw_audit = pd.read_csv(PROJECT_ROOT / "artifacts/data_v1/raw_acquisition_audit.csv")
    identity = {
        "freeze_kind": "data_protocol_v1",
        "config_hashes": config_hashes,
        "artifact_hashes": artifact_hashes,
        "data_pipeline_code_hash": tree_hash(code_paths),
        "counts": {
            "raw_partitions": len(raw_audit),
            "invalid_raw_partitions": int(raw_audit["validation_status"].ne("valid").sum()),
            "catalog_series": len(catalog),
            "resistivity_series_observed": int(catalog["factor"].eq("resistivity").sum()),
            "evaluation_target_series": int(windows["series_id"].nunique()),
            "candidate_windows": len(windows),
            "eligible_windows": int(windows["eligible"].sum()),
            "quality_event_candidates": len(events),
        },
    }
    return {
        **identity,
        "protocol_hash": stable_hash(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_scores_seen_before_freeze": False,
        "notes": [
            "Validation-only pipeline checks were run before this freeze.",
            "Retrospective quality events are unreviewed candidates and never modify Q0 truth.",
            "Model hyperparameters and Q1 detector parameters remain subject to validation-only selection.",
        ],
    }


def write_data_protocol_freeze(output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"Freeze record already exists and will not be overwritten: {destination}")
    record = build_data_protocol_freeze()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return record


def _formal_test_run_manifests() -> list[str]:
    found: list[str] = []
    artifacts = PROJECT_ROOT / "artifacts"
    if not artifacts.is_dir():
        return found
    for path in artifacts.rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        identity = manifest.get("identity") or manifest.get("request_manifest", {}).get("identity", {})
        if identity.get("split") in {"test_a", "test_b"}:
            found.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(found)


def build_model_protocol_freeze() -> dict[str, Any]:
    data_freeze_path = PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json"
    ledger_path = PROJECT_ROOT / "artifacts/validation/trials.jsonl"
    required = [data_freeze_path, ledger_path, *[PROJECT_ROOT / path for path in MODEL_PROTOCOL_CONFIGS]]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze model protocol; required files are missing: {missing}")
    formal_runs = _formal_test_run_manifests()
    if formal_runs:
        raise RuntimeError(f"Formal test runs already exist before model freeze: {formal_runs}")

    data_freeze = json.loads(data_freeze_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or any(record.get("ledger_version") != "validation-trial-v2" for record in records):
        raise RuntimeError("Validation ledger must contain only v2 records")
    if any(record.get("data_protocol_hash") != data_freeze["protocol_hash"] for record in records):
        raise RuntimeError("Validation ledger contains a stale data protocol")
    eligible = [record for record in records if record.get("eligible_for_selection")]
    observed_foundation = {
        (str(record["model_id"]), str(record["track"]), int(record["horizon"]))
        for record in eligible
        if record["model_id"] in {"timesfm_3", "chronos_2", "moirai_2_small", "toto_2_313m"}
    }
    if observed_foundation != FOUNDATION_VALIDATION_MATRIX:
        raise RuntimeError(
            "Foundation validation matrix is incomplete or unexpected: "
            f"expected={sorted(FOUNDATION_VALIDATION_MATRIX)}, observed={sorted(observed_foundation)}"
        )
    configs = {path.as_posix(): load_yaml(path) for path in MODEL_PROTOCOL_CONFIGS}
    expected_statuses = {
        "configs/model_registry.yaml": "validation_complete_pending_protocol_freeze",
        "configs/statistical_models.yaml": "validation_selection_complete",
        "configs/target_domain_models.yaml": "validation_selection_complete",
        "configs/robustness_v1.yaml": "validation_selection_complete",
    }
    for path, expected_status in expected_statuses.items():
        if configs[path].get("status") != expected_status:
            raise RuntimeError(f"{path} is not ready for model freeze")

    code_paths = model_code_paths()
    environment_paths = isolated_environment_paths()
    identity = {
        "freeze_kind": "model_protocol_v1",
        "data_protocol_hash": data_freeze["protocol_hash"],
        "config_hashes": {
            path.as_posix(): file_sha256(PROJECT_ROOT / path) for path in MODEL_PROTOCOL_CONFIGS
        },
        "validation_ledger_hash": file_sha256(ledger_path),
        "validation_counts": {
            "trials": len(records),
            "eligible_main_selection_trials": len(eligible),
            "sensitivity_trials": len(records) - len(eligible),
            "foundation_primary_endpoint_trials": len(observed_foundation),
        },
        "model_code_hash": tree_hash(code_paths),
        "isolated_environment_hash": tree_hash(environment_paths),
        "probability_protocol": configs["configs/robustness_v1.yaml"]["probability"],
        "q1_protocol": configs["configs/robustness_v1.yaml"]["q1"],
        "context_sensitivity": configs["configs/robustness_v1.yaml"]["context_sensitivity"],
    }
    return {
        "status": "frozen",
        **identity,
        "model_protocol_hash": stable_hash(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_scores_seen_before_freeze": False,
        "formal_test_manifest_audit": [],
        "notes": [
            "All model and robustness choices use validation data only.",
            "Q1 and 365-day context are sensitivity analyses; Q0/730 remains the main protocol.",
            "Native 90% intervals are unavailable in the common foundation-model contract and are N/A.",
        ],
    }


def write_model_protocol_freeze(output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"Freeze record already exists and will not be overwritten: {destination}")
    record = build_model_protocol_freeze()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return record
