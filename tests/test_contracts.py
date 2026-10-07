from __future__ import annotations

from geoelectrical_resistivity_forecasting_benchmark.contracts import ForecastRequest


def test_forecast_request_cannot_contain_target_truth() -> None:
    assert "target_truth" not in ForecastRequest.model_fields
    assert ForecastRequest.model_config["extra"] == "forbid"
