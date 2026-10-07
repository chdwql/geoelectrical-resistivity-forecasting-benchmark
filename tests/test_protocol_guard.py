from __future__ import annotations

import json

import pytest

from geoelectrical_resistivity_forecasting_benchmark.protocol_guard import (
    require_model_freeze_for_test,
)


def test_validation_does_not_require_model_freeze(tmp_path) -> None:
    require_model_freeze_for_test("validation", freeze_path=tmp_path / "missing.json")


def test_test_split_is_locked_without_freeze(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="locked until G4"):
        require_model_freeze_for_test("test_a", freeze_path=tmp_path / "missing.json")


def test_malformed_model_freeze_is_rejected(tmp_path) -> None:
    path = tmp_path / "MODEL_PROTOCOL_V1.json"
    path.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a valid frozen"):
        require_model_freeze_for_test("test_b", freeze_path=path)
