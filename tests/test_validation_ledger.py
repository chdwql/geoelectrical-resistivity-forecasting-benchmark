from __future__ import annotations

import json

import pandas as pd
import pytest

from geoelectrical_resistivity_forecasting_benchmark.config import file_sha256, stable_hash
from geoelectrical_resistivity_forecasting_benchmark.validation_ledger import (
    register_validation_runs,
)


def _write_run(path, *, split: str = "validation", protocol_hash: str = "protocol") -> None:
    path.mkdir()
    identity = {"split": split, "horizon": 90, "protocol_hash": protocol_hash, "models": ["persistence"]}
    manifest = {
        "run_id": "run-1",
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "successful_tasks": 2,
        "failed_tasks": 0,
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"point_forecast": [1.0]}).to_parquet(path / "predictions.parquet")
    pd.DataFrame({"mase": [1.0]}).to_parquet(path / "scores.parquet")
    pd.DataFrame(
        {
            "model_id": ["persistence"],
            "station_equal_mase": [1.0],
            "station_equal_mae": [2.0],
            "station_equal_rmse": [3.0],
        }
    ).to_csv(path / "summary.csv", index=False)


def test_registration_is_idempotent(tmp_path) -> None:
    run = tmp_path / "run"
    ledger = tmp_path / "trials.jsonl"
    _write_run(run)
    assert len(register_validation_runs([run], output_path=ledger, expected_protocol_hash="protocol")) == 1
    first_hash = file_sha256(ledger)
    assert register_validation_runs([run], output_path=ledger, expected_protocol_hash="protocol") == []
    assert file_sha256(ledger) == first_hash


def test_rejects_non_validation_or_stale_protocol(tmp_path) -> None:
    test_run = tmp_path / "test"
    _write_run(test_run, split="test_a")
    with pytest.raises(ValueError, match="only accepts validation"):
        register_validation_runs([test_run], output_path=tmp_path / "a.jsonl", expected_protocol_hash="protocol")
    stale = tmp_path / "stale"
    _write_run(stale, protocol_hash="old")
    with pytest.raises(ValueError, match="active data protocol"):
        register_validation_runs([stale], output_path=tmp_path / "b.jsonl", expected_protocol_hash="protocol")
