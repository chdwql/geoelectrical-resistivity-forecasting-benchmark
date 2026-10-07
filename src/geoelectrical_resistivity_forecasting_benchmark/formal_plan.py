from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, load_yaml, stable_hash
from .protocol_guard import require_model_freeze_for_test


TRACKS = ("U", "M", "X")
HORIZONS = (1, 7, 30, 90, 180)
SPLITS = ("test_a", "test_b")


def model_supports(model: dict[str, object], track: str, horizon: int) -> bool:
    if track not in model.get("tracks", []):
        return False
    constraints = model.get("track_constraints", {})
    allowed = constraints.get(track) if isinstance(constraints, dict) else None
    return allowed is None or horizon in allowed


def build_formal_task_plan(
    *,
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    output_path: str | Path = "artifacts/formal/task_plan.jsonl",
) -> tuple[Path, list[dict[str, object]]]:
    for split in SPLITS:
        require_model_freeze_for_test(split)
    windows_file = Path(windows_path)
    if not windows_file.is_absolute():
        windows_file = PROJECT_ROOT / windows_file
    windows = pd.read_parquet(windows_file)
    registry = load_yaml("configs/model_registry.yaml")
    rows: list[dict[str, object]] = []
    for model in registry["models"]:
        for split in SPLITS:
            for track in TRACKS:
                for horizon in HORIZONS:
                    eligible_windows = int(
                        (
                            windows["split"].eq(split)
                            & windows["track"].eq(track)
                            & windows["horizon"].eq(horizon)
                            & windows["eligible"]
                        ).sum()
                    )
                    supported = model_supports(model, track, horizon)
                    identity = {
                        "model_id": model["id"],
                        "split": split,
                        "track": track,
                        "horizon": horizon,
                        "input_variant": "Q0",
                        "context_days": 730,
                    }
                    rows.append(
                        {
                            "task_id": stable_hash(identity)[:24],
                            **identity,
                            "eligible_windows": eligible_windows,
                            "status": "planned" if supported else "N/A",
                            "failure_reason": None if supported else "unsupported_model_track_or_horizon",
                            "output_directory": (
                                f"artifacts/formal/runs/{model['id']}/{split}/{track.lower()}{horizon}"
                                if supported
                                else None
                            ),
                        }
                    )
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination, rows
