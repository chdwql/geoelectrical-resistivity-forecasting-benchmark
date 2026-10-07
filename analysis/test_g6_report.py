from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.g6_report import (
    hierarchical_effect,
    holm_adjust,
    moving_block_bootstrap,
    validate_standard_index,
)
from analysis.g6_figures import station_equal_summary
from analysis.g6_extended import event_strata, probability_summary
from analysis.g6_remaining import rank_stability


def test_hierarchical_effect_is_station_equal_and_series_equal() -> None:
    pairs = pd.DataFrame(
        {
            "station": ["A", "A", "A", "B"],
            "series_id": ["A/1", "A/1", "A/2", "B/1"],
            "origin": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01"]),
            "delta_mase": [1.0, 3.0, 6.0, 10.0],
        }
    )
    # A: mean(series A/1=2, A/2=6)=4; B=10; station-equal result=7.
    assert hierarchical_effect(pairs) == 7.0


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    raw = np.asarray([0.03, 0.01, 0.20])
    adjusted = holm_adjust(raw)
    assert np.allclose(adjusted, [0.06, 0.03, 0.20])
    assert np.all(adjusted >= raw)


def test_moving_block_bootstrap_is_deterministic() -> None:
    pairs = pd.DataFrame(
        {
            "station": ["A"] * 6,
            "series_id": ["A/1"] * 6,
            "origin": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "delta_mase": np.arange(6, dtype=float),
        }
    )
    first = moving_block_bootstrap(pairs, draws=20, block_origins=3, seed=17)
    second = moving_block_bootstrap(pairs, draws=20, block_origins=3, seed=17)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_vectorized_bootstrap_matches_dataframe_definition() -> None:
    pairs = pd.DataFrame(
        {
            "station": ["A", "A", "A", "A", "A"],
            "series_id": ["A/1", "A/2", "A/1", "A/1", "A/2"],
            "origin": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01", "2024-03-01"]
            ),
            "delta_mase": [1.0, 10.0, 2.0, 3.0, 30.0],
        }
    )

    def dataframe_definition(draws: int, block: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        origins = np.asarray(sorted(pairs["origin"].unique()))
        origin_rows = {origin: pairs.loc[pairs["origin"].eq(origin)] for origin in origins}
        result = np.empty(draws)
        for draw in range(draws):
            sampled = []
            while len(sampled) < len(origins):
                start = int(rng.integers(0, len(origins)))
                for offset in range(block):
                    sampled.append(origin_rows[origins[(start + offset) % len(origins)]])
                    if len(sampled) == len(origins):
                        break
            replicated = pd.concat(sampled, ignore_index=True)
            result[draw] = replicated.groupby("series_id")["delta_mase"].mean().mean()
        return result

    expected = dataframe_definition(draws=50, block=2, seed=23)
    actual = moving_block_bootstrap(pairs, draws=50, block_origins=2, seed=23)
    assert np.allclose(actual, expected)


def test_task_plan_preserves_uppercase_na_status() -> None:
    plan = pd.DataFrame(
        [
            {"model_id": "a", "split": "test_a", "horizon": 90, "track": "U", "status": "planned", "eligible_windows": 2},
            {"model_id": "a", "split": "test_a", "horizon": 90, "track": "M", "status": "N/A", "eligible_windows": 2},
        ]
    )
    index = pd.DataFrame(
        [
            {
                "model_id": "a",
                "split": "test_a",
                "horizon": 90,
                "track": "U",
                "task_count": 2,
                "success_count": 2,
                "error_count": 0,
                "run_directory": "artifacts/example",
            }
        ]
    )
    coverage = validate_standard_index(index, plan)
    assert coverage.loc[coverage["track"].eq("M"), "final_status"].item() == "N/A"


def test_skill_horizon_summary_is_station_and_series_equal() -> None:
    ledger = pd.DataFrame(
        {
            "model_id": ["m"] * 4,
            "split": ["test_a"] * 4,
            "horizon": [1] * 4,
            "track": ["U"] * 4,
            "station": ["A", "A", "A", "B"],
            "series_id": ["A/1", "A/1", "A/2", "B/1"],
            "mase": [1.0, 3.0, 6.0, 10.0],
        }
    )
    summary = station_equal_summary(ledger)
    assert summary["station_equal_mase"].item() == 7.0


def test_probability_summary_uses_absolute_coverage_error() -> None:
    ledger = pd.DataFrame(
        {
            "model_id": ["m"], "split": ["test_a"], "horizon": [90], "track": ["U"],
            "station": ["A"], "series_id": ["A/1"], "mean_quantile_loss": [0.1],
            "wis": [0.2], "coverage_80": [0.7], "width_80": [1.0], "quantile_crossing_rate": [0.0],
        }
    )
    result = probability_summary(ledger)
    assert np.isclose(result["coverage_80_error"].item(), 0.1)


def test_event_strata_marks_events_only_inside_target() -> None:
    ledger = pd.DataFrame(
        {
            "model_id": ["m", "m"], "split": ["test_a", "test_a"], "horizon": [7, 7], "track": ["U", "U"],
            "station": ["A", "A"], "series_id": ["A/1", "A/1"], "window_id": ["w1", "w2"],
            "target_start": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "target_end": pd.to_datetime(["2024-01-07", "2024-02-07"]), "mase": [1.0, 2.0], "mae": [0.1, 0.2],
        }
    )
    events = pd.DataFrame({"series_id": ["A/1"], "event_date": ["2024-01-03"], "event_type": ["step_candidate"]})
    result = event_strata(ledger, events)
    assert set(result["event_stratum"]) == {"step_candidate", "no_target_event"}


def test_rank_stability_uses_within_split_horizon_ranks() -> None:
    skill = pd.DataFrame(
        {"model_id": ["a", "b", "a", "b"], "split": ["test_a", "test_a", "test_b", "test_b"],
         "horizon": [90, 90, 90, 90], "station_equal_mase": [1.0, 2.0, 3.0, 2.0]}
    )
    ranks, stats = rank_stability(skill)
    assert ranks.loc[(ranks.model_id == "a") & (ranks.split == "test_a"), "rank"].item() == 1
    assert np.isclose(stats["spearman_rank_correlation"].item(), -1)
