from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Track(StrEnum):
    U = "U"
    M = "M"
    X = "X"
    O = "O"


class TaskStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCESS = "success"
    NOT_APPLICABLE = "not_applicable"
    EMPTY_INPUT = "empty_input"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"


class WindowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str
    series_id: str
    origin: date
    context_start: date
    context_end: date
    target_start: date
    target_end: date
    horizon: int = Field(gt=0)
    track: Track
    split: str
    mask_hash: str
    eligible: bool
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "WindowRecord":
        if self.context_end >= self.target_start:
            raise ValueError("Context must end before the first target date")
        if (self.target_end - self.target_start).days + 1 != self.horizon:
            raise ValueError("Target date range must equal horizon")
        if self.eligible and self.exclusion_reason:
            raise ValueError("Eligible windows cannot have an exclusion reason")
        return self


class ForecastRequest(BaseModel):
    """Information available to an adapter at inference time.

    Deliberately contains no target truth field.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    window_id: str
    series_id: str
    origin: date
    horizon: int = Field(gt=0)
    track: Track
    history_target: pd.Series
    history_covariates: pd.DataFrame | None = None
    future_calendar: pd.DataFrame | None = None


class ForecastResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    target_dates: pd.DatetimeIndex
    point_forecast: np.ndarray
    quantiles: dict[float, np.ndarray] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shapes(self) -> "ForecastResult":
        size = len(self.target_dates)
        if self.point_forecast.shape != (size,):
            raise ValueError("Point forecast length must match target dates")
        for level, values in self.quantiles.items():
            if not 0 < level < 1:
                raise ValueError(f"Invalid quantile level: {level}")
            if values.shape != (size,):
                raise ValueError(f"Quantile {level} length must match target dates")
        return self


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    protocol_hash: str
    model_id: str
    weight_hash: str | None
    window_id: str
    target_date: date | None
    point_forecast: float | None
    quantiles: dict[str, float] = Field(default_factory=dict)
    seed: int | None
    status: TaskStatus
    runtime_seconds: float | None = None
    peak_memory_mb: float | None = None
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
