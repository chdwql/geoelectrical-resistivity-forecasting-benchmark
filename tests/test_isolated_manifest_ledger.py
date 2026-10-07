from __future__ import annotations

import json

import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.config import stable_hash
from geoelectrical_resistivity_forecasting_benchmark.validation_ledger import (
    register_validation_runs,
)


def test_nested_request_manifest_can_be_registered(tmp_path) -> None:
    run = tmp_path / "foundation"
    run.mkdir()
    identity = {
        "split": "validation",
        "horizon": 90,
        "track": "U",
        "protocol_hash": "protocol",
        "stations": None,
        "limit": None,
    }
    manifest = {
        "run_id": "foundation-run",
        "model_id": "foundation",
        "request_manifest": {"identity": identity, "identity_hash": stable_hash(identity)},
        "successful_tasks": 1,
        "failed_tasks": 0,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"point_forecast": [1.0]}).to_parquet(run / "predictions.parquet")
    pd.DataFrame({"mase": [1.0]}).to_parquet(run / "scores.parquet")
    pd.DataFrame(
        {
            "model_id": ["foundation"],
            "station_equal_mase": [1.0],
            "station_equal_mae": [1.0],
            "station_equal_rmse": [1.0],
        }
    ).to_csv(run / "summary.csv", index=False)
    records = register_validation_runs(
        [run], output_path=tmp_path / "trials.jsonl", expected_protocol_hash="protocol"
    )
    assert len(records) == 1
    assert records[0]["eligible_for_selection"] is True
