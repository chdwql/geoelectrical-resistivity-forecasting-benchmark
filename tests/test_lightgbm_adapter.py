from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.adapters.lightgbm_direct import (
    LightGBMDirectAdapter,
    request_feature_frame,
)
from geoelectrical_resistivity_forecasting_benchmark.contracts import ForecastRequest, Track


def make_request(track: Track = Track.X) -> ForecastRequest:
    history_dates = pd.date_range("2020-01-02", periods=730, freq="D")
    future_dates = pd.date_range("2022-01-01", periods=7, freq="D")
    history = pd.Series(np.linspace(10, 20, 730), index=history_dates)
    covariates = pd.DataFrame(
        {
            "11003/D/3212": np.linspace(20, 30, 730),
            "water_level": np.linspace(1, 2, 730),
            "temperature": np.linspace(-5, 25, 730),
            "rainfall": np.zeros(730),
        },
        index=history_dates,
    )
    return ForecastRequest(
        window_id="window",
        series_id="11003/D/3211",
        origin=date(2022, 1, 1),
        horizon=7,
        track=track,
        history_target=history,
        history_covariates=covariates,
        future_calendar=None,
    )


def test_feature_frame_is_causal_and_track_specific() -> None:
    request = make_request()
    features = request_feature_frame(request)
    assert len(features) == 7
    assert features.loc[0, "target_lag_1"] == 20
    assert features.loc[0, "station_panel_lag_1"] == 30
    assert features.loc[0, "env_temperature_lag_1"] == 25
    assert features.loc[0, "target_diff_scale"] > 0
    assert features["lead"].tolist() == list(range(1, 8))

    changed = request.model_copy(
        update={"history_target": request.history_target.copy()}
    )
    changed.history_target.iloc[-1] = 999
    changed_features = request_feature_frame(changed)
    assert changed_features.loc[0, "target_lag_1"] == 999
    assert changed_features.loc[0, "target_lag_2"] == features.loc[0, "target_lag_2"]


def test_lightgbm_fit_predict_contract() -> None:
    request = make_request(Track.U)
    features = request_feature_frame(request)
    training = pd.concat([features.assign(label=np.arange(7) + offset) for offset in range(4)])
    adapter = LightGBMDirectAdapter()
    adapter.fit(
        training,
        {"seeds": [17, 29, 43], "parameters": {"n_estimators": 5, "num_leaves": 4}},
    )
    result = adapter.predict(request)
    assert result.point_forecast.shape == (7,)
    assert np.isfinite(result.point_forecast).all()
    assert result.metadata["seeds"] == [17, 29, 43]
