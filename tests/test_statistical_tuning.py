import pytest

from geoelectrical_resistivity_forecasting_benchmark.statistical_tuning import (
    run_statistical_validation,
    statistical_model_specs,
)


def test_statistical_grid_has_unique_effective_candidates() -> None:
    specs = statistical_model_specs()
    assert len(specs) == 16
    assert len({spec["model_id"] for spec in specs}) == 16
    assert sum(spec["base_model_id"] == "ets" for spec in specs) == 3
    assert sum(spec["base_model_id"] == "arima" for spec in specs) == 4
    assert sum(spec["base_model_id"] == "fourier_ridge" for spec in specs) == 9


def test_statistical_grid_can_select_one_family() -> None:
    specs = statistical_model_specs(families={"fourier_ridge"})
    assert len(specs) == 9
    assert {spec["base_model_id"] for spec in specs} == {"fourier_ridge"}


def test_unknown_candidate_is_rejected_before_running() -> None:
    with pytest.raises(ValueError, match="Unknown candidate"):
        run_statistical_validation(families={"arima"}, candidates={"arima__missing"})
