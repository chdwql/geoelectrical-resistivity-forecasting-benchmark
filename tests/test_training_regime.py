from __future__ import annotations

from geoelectrical_resistivity_forecasting_benchmark.target_domain_runner import training_splits_for


def test_validation_trains_only_on_initial_training_period() -> None:
    assert training_splits_for("validation") == {"train"}


def test_formal_tests_refit_through_validation_without_test_labels() -> None:
    assert training_splits_for("test_a") == {"train", "validation"}
    assert training_splits_for("test_b") == {"train", "validation"}
