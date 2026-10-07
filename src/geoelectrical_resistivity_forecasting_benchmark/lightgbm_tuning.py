from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .config import load_yaml
from .scoring import score_run
from .target_domain_runner import run_lightgbm
from .validation_ledger import register_validation_runs


def lightgbm_model_specs(
    config_path: str | Path = "configs/target_domain_models.yaml",
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)["lightgbm"]
    specs: list[dict[str, Any]] = []
    for leaves, rate, estimators in product(
        config["candidate_num_leaves"],
        config["candidate_learning_rate"],
        config["candidate_n_estimators"],
    ):
        specs.append(
            {
                "model_id": f"lightgbm__l{int(leaves)}_r{float(rate):g}_n{int(estimators)}",
                "parameters": {
                    "num_leaves": int(leaves),
                    "learning_rate": float(rate),
                    "n_estimators": int(estimators),
                },
            }
        )
    return specs


def run_lightgbm_validation(*, track: str, horizon: int = 90) -> Path:
    destination = run_lightgbm(
        split="validation",
        horizon=horizon,
        track=track,
        model_specs=lightgbm_model_specs(),
    )
    score_run(destination)
    register_validation_runs([destination])
    return destination
