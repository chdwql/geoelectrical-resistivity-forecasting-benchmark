from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.synthetic_end_to_end import run_example


def test_public_synthetic_example_is_deterministic_and_restricted_data_free(tmp_path: Path) -> None:
    first = run_example(tmp_path / "first")
    second = run_example(tmp_path / "second")
    assert first == second
    assert first["contains_restricted_data"] is False
    assert first["successful_scores"] == first["score_rows"]
    assert first["windows"] == 16
    predictions = pd.read_csv(tmp_path / "first" / "predictions.csv")
    assert set(predictions["status"]) == {"success"}
    assert set(predictions["model_id"]) == {"persistence", "drift", "seasonal_365"}
