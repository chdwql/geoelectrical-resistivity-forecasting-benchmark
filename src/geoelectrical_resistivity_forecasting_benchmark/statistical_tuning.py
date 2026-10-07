from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .config import load_yaml
from .runner import run_local_models
from .scoring import score_run
from .validation_ledger import register_validation_runs


def statistical_model_specs(
    config_path: str | Path = "configs/statistical_models.yaml",
    *,
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    selected = families or {"ets", "arima", "fourier_ridge"}
    unknown = selected - {"ets", "arima", "fourier_ridge"}
    if unknown:
        raise ValueError(f"Unknown statistical families: {sorted(unknown)}")
    specs: list[dict[str, Any]] = []
    if "ets" in selected:
        effective: set[tuple[bool, bool]] = set()
        for trend, damped in product(
            config["ets"]["candidate_trend"], config["ets"]["candidate_damped_trend"]
        ):
            effective.add((bool(trend), bool(damped) if trend else False))
        for trend, damped in sorted(effective):
            specs.append(
                {
                    "base_model_id": "ets",
                    "model_id": f"ets__trend{int(trend)}_damped{int(damped)}",
                    "params": {"trend": trend, "damped_trend": damped},
                }
            )
    if "arima" in selected:
        for order_values in config["arima"]["candidate_orders"]:
            order = tuple(int(value) for value in order_values)
            specs.append(
                {
                    "base_model_id": "arima",
                    "model_id": "arima__" + "_".join(str(value) for value in order),
                    "params": {"order": order},
                }
            )
    if "fourier_ridge" in selected:
        for harmonics, alpha in product(
            config["fourier_ridge"]["annual_harmonics"], config["fourier_ridge"]["alpha"]
        ):
            specs.append(
                {
                    "base_model_id": "fourier_ridge",
                    "model_id": f"fourier_ridge__h{int(harmonics)}_a{float(alpha):g}",
                    "params": {"harmonics": int(harmonics), "alpha": float(alpha)},
                }
            )
    return specs


def run_statistical_validation(
    *,
    horizon: int = 90,
    families: set[str] | None = None,
    stations: list[str] | None = None,
    limit: int | None = None,
    candidates: set[str] | None = None,
) -> Path:
    specs = statistical_model_specs(families=families)
    if candidates is not None:
        available = {str(spec["model_id"]) for spec in specs}
        unknown = candidates - available
        if unknown:
            raise ValueError(f"Unknown candidate model IDs: {sorted(unknown)}")
        specs = [spec for spec in specs if str(spec["model_id"]) in candidates]
    destination = run_local_models(
        split="validation",
        horizon=horizon,
        model_ids=[str(spec["model_id"]) for spec in specs],
        model_specs=specs,
        stations=stations,
        limit=limit,
    )
    score_run(destination)
    register_validation_runs([destination])
    return destination
