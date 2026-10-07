from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .runner import run_local_models
from .scoring import score_run
from .validation_ledger import register_validation_runs


Q1_CANDIDATES = [
    {"window": window, "threshold": threshold, "minimum_history": 14 if window == 30 else 30}
    for window in (30, 90)
    for threshold in (6.0, 8.0, 10.0)
]


def run_q1_validation(
    *,
    output_path: str | Path = "artifacts/validation/q1_selection.csv",
) -> pd.DataFrame:
    """Evaluate the fixed Q1 grid on validation U-90 with two simple anchors."""

    rows: list[dict[str, object]] = []
    run_paths: list[Path] = []
    for candidate in Q1_CANDIDATES:
        run_path = run_local_models(
            split="validation",
            horizon=90,
            model_ids=["persistence", "drift"],
            input_variant="Q1",
            context_days=730,
            q1_window=int(candidate["window"]),
            q1_threshold=float(candidate["threshold"]),
            q1_minimum_history=int(candidate["minimum_history"]),
            output_root="artifacts/validation/q1_runs",
        )
        _scores, summary = score_run(run_path)
        run_paths.append(run_path)
        rows.append(
            {
                **candidate,
                "mean_anchor_station_equal_mase": float(summary["station_equal_mase"].mean()),
                "persistence_station_equal_mase": float(
                    summary.loc[summary["model_id"].eq("persistence"), "station_equal_mase"].iloc[0]
                ),
                "drift_station_equal_mase": float(
                    summary.loc[summary["model_id"].eq("drift"), "station_equal_mase"].iloc[0]
                ),
                "run_directory": str(run_path.relative_to(PROJECT_ROOT)),
            }
        )
    register_validation_runs(run_paths)
    result = pd.DataFrame(rows).sort_values(
        ["mean_anchor_station_equal_mase", "threshold", "window"],
        ascending=[True, False, False],
    )
    result["selected"] = False
    result.loc[result.index[0], "selected"] = True
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    result.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, destination)
    return result


def run_context_validation(
    *,
    output_path: str | Path = "artifacts/validation/context_365_summary.csv",
) -> Path:
    """Exercise the preregistered 365-day context sensitivity on validation U-90."""

    run_path = run_local_models(
        split="validation",
        horizon=90,
        model_ids=["persistence", "drift", "seasonal_365"],
        input_variant="Q0",
        context_days=365,
        output_root="artifacts/validation/context_runs",
    )
    _scores, summary = score_run(run_path)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    summary.assign(run_directory=str(run_path.relative_to(PROJECT_ROOT))).to_csv(
        temporary, index=False, encoding="utf-8-sig"
    )
    os.replace(temporary, destination)
    register_validation_runs([run_path])
    return run_path
