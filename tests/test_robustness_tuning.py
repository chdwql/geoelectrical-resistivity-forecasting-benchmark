from __future__ import annotations

from geoelectrical_resistivity_forecasting_benchmark.robustness_tuning import Q1_CANDIDATES


def test_q1_grid_is_fixed_and_unique() -> None:
    identities = {
        (candidate["window"], candidate["threshold"], candidate["minimum_history"])
        for candidate in Q1_CANDIDATES
    }
    assert len(Q1_CANDIDATES) == 6
    assert len(identities) == 6
