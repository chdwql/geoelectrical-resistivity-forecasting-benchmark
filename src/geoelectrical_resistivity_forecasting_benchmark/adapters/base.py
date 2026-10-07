from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..contracts import ForecastRequest, ForecastResult


class ForecastAdapter(ABC):
    model_id: str
    supported_tracks: frozenset[str]

    def fit(self, train_manifest: Any, config: dict[str, Any]) -> None:
        """Fit target-domain models. Frozen and deterministic models are no-ops."""

    @abstractmethod
    def predict(self, request: ForecastRequest) -> ForecastResult:
        raise NotImplementedError
