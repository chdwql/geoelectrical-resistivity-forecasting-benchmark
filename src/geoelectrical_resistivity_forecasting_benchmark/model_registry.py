from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_yaml


def probe_registry(path: str | Path = "configs/model_registry.yaml") -> list[dict[str, Any]]:
    registry = load_yaml(path)
    result: list[dict[str, Any]] = []
    for model in registry["models"]:
        record = dict(model)
        local_path = record.get("local_path")
        if local_path:
            resolved = PROJECT_ROOT / local_path
            record["weights_available"] = resolved.is_dir()
            record["weight_files"] = len(list(resolved.glob("*"))) if resolved.is_dir() else 0
        else:
            record["weights_available"] = False
            record["weight_files"] = 0
        result.append(record)
    return result
