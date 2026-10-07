from __future__ import annotations

import hashlib
import json

import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.source_data import validate_raw_acquisition


def test_raw_acquisition_validator_checks_partition_contract(tmp_path, monkeypatch) -> None:
    task = {
        "task_id": "a" * 24,
        "station": "11003",
        "point": "D",
        "factor": "resistivity",
        "sample_rate": "小时值",
        "start": "20160101",
        "end": "20161231",
        "status": "planned",
    }
    pd.DataFrame([task]).to_csv(tmp_path / "tasks.csv", index=False)
    folder = tmp_path / "raw" / "11003" / "resistivity"
    folder.mkdir(parents=True)
    parquet = folder / f"20160101_20161231_D_{task['task_id']}.parquet"
    frame = pd.DataFrame(
        {
            "STARTDATE": [pd.Timestamp("2016-01-01")],
            "STATIONID": ["11003"],
            "POINTID": ["D"],
            "ITEMID": ["3211"],
            "SAMPLERATE": ["60"],
            "OBSVALUE": [1.0],
        }
    )
    frame.to_parquet(parquet, index=False)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    record = {
        **{key: value for key, value in task.items() if key != "status"},
        "status": "success",
        "rows": 1,
        "first_timestamp": "2016-01-01T00:00:00",
        "last_timestamp": "2016-01-01T00:00:00",
        "sha256": digest,
    }
    parquet.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
    logs = tmp_path / "raw" / "_logs"
    logs.mkdir(parents=True)
    (logs / f"{task['task_id']}.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        "geoelectrical_resistivity_forecasting_benchmark.source_data.PROJECT_ROOT", tmp_path
    )

    result = validate_raw_acquisition("tasks.csv", "raw", "audit.csv")

    assert result.iloc[0]["validation_status"] == "valid"
    assert result.iloc[0]["rows"] == 1
    assert (tmp_path / "audit.csv").is_file()
