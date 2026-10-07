from geoelectrical_resistivity_forecasting_benchmark.lightgbm_tuning import (
    lightgbm_model_specs,
)


def test_lightgbm_grid_is_complete_and_unique() -> None:
    specs = lightgbm_model_specs()
    assert len(specs) == 12
    assert len({spec["model_id"] for spec in specs}) == 12
    assert {spec["parameters"]["num_leaves"] for spec in specs} == {15, 31, 63}
    assert {spec["parameters"]["learning_rate"] for spec in specs} == {0.03, 0.05}
    assert {spec["parameters"]["n_estimators"] for spec in specs} == {300, 600}
