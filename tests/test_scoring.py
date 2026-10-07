from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.scoring import _probabilistic_scores


def test_probabilistic_scores_report_supported_intervals_only() -> None:
    actual = np.array([0.0, 1.0])
    quantiles = pd.Series(
        [
            json.dumps({"0.1": -1.0, "0.5": 0.0, "0.9": 1.0}),
            json.dumps({"0.1": 0.0, "0.5": 1.0, "0.9": 2.0}),
        ]
    )
    scores = _probabilistic_scores(actual, quantiles)
    assert scores["coverage_80"] == 1.0
    assert scores["width_80"] == 2.0
    assert math.isnan(scores["coverage_90"])
    assert math.isnan(scores["width_90"])
    assert scores["quantile_crossing_rate"] == 0.0
    assert np.isfinite(scores["mean_quantile_loss"])
    assert np.isfinite(scores["wis"])


def test_probabilistic_scores_detect_and_rearrange_crossing_without_truth() -> None:
    quantiles = pd.Series([json.dumps({"0.1": 2.0, "0.5": 1.0, "0.9": 0.0})])
    scores = _probabilistic_scores(np.array([1.0]), quantiles)
    assert scores["quantile_crossing_rate"] == 1.0
    assert scores["coverage_80"] == 1.0
    assert np.isfinite(scores["wis"])


def test_probabilistic_scores_allow_missing_quantiles() -> None:
    scores = _probabilistic_scores(np.array([1.0]), pd.Series([None]))
    assert math.isnan(scores["mean_quantile_loss"])
    assert scores["quantile_levels"] == ""
