from __future__ import annotations

import json

import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.ingestion import build_fetch_tasks


def test_fetch_plan_uses_monthly_minutes_and_yearly_hours(tmp_path, monkeypatch) -> None:
    audit = pd.DataFrame(
        [
            {
                "station": "11003",
                "factor": "resistivity",
                "status": "available",
                "series": json.dumps([{"point": "D", "item": "3211", "sample_rate": "60"}]),
            },
            {
                "station": "11003",
                "factor": "rainfall",
                "status": "available",
                "series": json.dumps([{"point": "A", "item": "9140", "sample_rate": "1"}]),
            },
            {
                "station": "37005",
                "factor": "water_level",
                "status": "available",
                "series": json.dumps([{"point": "H", "item": "9210", "sample_rate": "90"}]),
            },
        ]
    )
    audit_path = tmp_path / "audit.csv"
    output_path = tmp_path / "tasks.csv"
    audit.to_csv(audit_path, index=False)
    monkeypatch.setattr(
        "geoelectrical_resistivity_forecasting_benchmark.ingestion.PROJECT_ROOT",
        tmp_path,
    )
    tasks = build_fetch_tasks("audit.csv", "tasks.csv")
    counts = tasks.groupby("factor").size().to_dict()
    assert counts == {"rainfall": 60, "resistivity": 5, "water_level": 5}
    assert set(tasks.loc[tasks["factor"].eq("rainfall"), "sample_rate"]) == {"分钟值"}
    assert set(tasks.loc[tasks["factor"].eq("water_level"), "sample_rate"]) == {"日值"}
