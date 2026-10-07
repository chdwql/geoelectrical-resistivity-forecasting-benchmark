from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..contracts import ForecastRequest, ForecastResult, Track
from .base import ForecastAdapter


LAGS = (1, 2, 3, 7, 14, 30, 90, 180, 365, 730)
ROLLING_WINDOWS = (7, 30, 90, 365)
ENVIRONMENT_COLUMNS = frozenset({"water_level", "temperature", "rainfall"})
DEFAULT_SEEDS = (17, 29, 43)


def _stable_code(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def _last_finite(values: pd.Series) -> float:
    finite = values.dropna()
    return float(finite.iloc[-1]) if not finite.empty else np.nan


def _series_features(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output: dict[str, float] = {}
    for lag in LAGS:
        output[f"{prefix}_lag_{lag}"] = float(numeric.iloc[-lag]) if len(numeric) >= lag else np.nan
    for window in ROLLING_WINDOWS:
        tail = numeric.iloc[-window:]
        output[f"{prefix}_mean_{window}"] = float(tail.mean()) if tail.notna().any() else np.nan
        output[f"{prefix}_std_{window}"] = float(tail.std(ddof=0)) if tail.notna().any() else np.nan
    output[f"{prefix}_last_finite"] = _last_finite(numeric)
    output[f"{prefix}_coverage"] = float(numeric.notna().mean()) if len(numeric) else 0.0
    differences = numeric.diff().abs().dropna()
    output[f"{prefix}_diff_scale"] = float(differences.mean()) if not differences.empty else np.nan
    return output


def request_feature_frame(request: ForecastRequest) -> pd.DataFrame:
    """Create causal, fixed-width features from one inference request."""

    dates = pd.date_range(request.origin, periods=request.horizon, freq="D")
    base = _series_features(request.history_target, "target")
    base["series_code"] = float(_stable_code(request.series_id))
    base["station_code"] = float(_stable_code(request.series_id.split("/", 1)[0]))

    covariates = request.history_covariates
    if request.track in {Track.M, Track.X}:
        resistivity = pd.DataFrame(index=request.history_target.index)
        if covariates is not None:
            resistivity_columns = [
                column for column in covariates.columns if str(column) not in ENVIRONMENT_COLUMNS
            ]
            resistivity = covariates[resistivity_columns]
        panel_mean = resistivity.mean(axis=1) if not resistivity.empty else pd.Series(dtype=float)
        base.update(_series_features(panel_mean, "station_panel"))

    if request.track is Track.X:
        for factor in sorted(ENVIRONMENT_COLUMNS):
            values = (
                covariates[factor]
                if covariates is not None and factor in covariates.columns
                else pd.Series(index=request.history_target.index, dtype=float)
            )
            base.update(_series_features(values, f"env_{factor}"))

    frame = pd.DataFrame([base] * request.horizon)
    frame["lead"] = np.arange(1, request.horizon + 1, dtype=float)
    if request.future_calendar is not None:
        calendar = request.future_calendar.reindex(dates)
        for column in ("year_sin", "year_cos", "week_sin", "week_cos"):
            matching = [name for name in calendar.columns if str(name).split("/")[-1] == column]
            if matching:
                frame[column] = calendar[matching[0]].to_numpy(dtype=float)
            else:
                frame[column] = np.nan
    else:
        day = dates.dayofyear.to_numpy(dtype=float)
        weekday = dates.dayofweek.to_numpy(dtype=float)
        frame["year_sin"] = np.sin(2 * np.pi * day / 365.25)
        frame["year_cos"] = np.cos(2 * np.pi * day / 365.25)
        frame["week_sin"] = np.sin(2 * np.pi * weekday / 7)
        frame["week_cos"] = np.cos(2 * np.pi * weekday / 7)
    return frame


class LightGBMDirectAdapter(ForecastAdapter):
    model_id = "lightgbm"
    supported_tracks = frozenset({"U", "M", "X"})

    def __init__(self) -> None:
        self.models: list[LGBMRegressor] = []
        self.feature_columns: list[str] = []

    def fit(self, train_manifest: Any, config: dict[str, Any]) -> None:
        if not isinstance(train_manifest, pd.DataFrame) or "label" not in train_manifest:
            raise TypeError("LightGBM train_manifest must be a DataFrame with a label column")
        excluded = {
            "label",
            "label_original",
            "origin",
            "target_date",
            "window_id",
            "series_id",
        }
        self.feature_columns = [column for column in train_manifest.columns if column not in excluded]
        if not self.feature_columns:
            raise ValueError("No LightGBM features were provided")
        training = train_manifest.loc[train_manifest["label"].notna()]
        if training.empty:
            raise ValueError("No observed LightGBM training labels")
        seeds = tuple(int(seed) for seed in config.get("seeds", DEFAULT_SEEDS))
        parameters = {
            "objective": "l1",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 4,
            "verbosity": -1,
            **config.get("parameters", {}),
        }
        self.models = []
        for seed in seeds:
            model = LGBMRegressor(random_state=seed, **parameters)
            model.fit(training[self.feature_columns], training["label"].to_numpy(dtype=float))
            self.models.append(model)

    def predict(self, request: ForecastRequest) -> ForecastResult:
        if not self.models:
            raise RuntimeError("LightGBM adapter must be fitted before prediction")
        if request.track.value not in self.supported_tracks:
            raise ValueError(f"Unsupported LightGBM track: {request.track}")
        features = request_feature_frame(request).reindex(columns=self.feature_columns)
        normalized = np.stack([model.predict(features) for model in self.models]).mean(axis=0)
        raw_features = request_feature_frame(request)
        anchor = float(raw_features["target_last_finite"].iloc[0])
        scale = float(raw_features["target_diff_scale"].iloc[0])
        if not np.isfinite(anchor) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Target history has no valid causal anchor/scale")
        forecasts = anchor + scale * normalized
        dates = pd.date_range(request.origin, periods=request.horizon, freq="D")
        return ForecastResult(
            target_dates=dates,
            point_forecast=forecasts.astype(float),
            metadata={"seeds": [model.random_state for model in self.models]},
        )
