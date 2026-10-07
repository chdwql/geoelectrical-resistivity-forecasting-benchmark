from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, file_sha256, tree_hash


MODEL_FREEZE_PATH = PROJECT_ROOT / "docs/frozen/MODEL_PROTOCOL_V1.json"
TEST_SPLITS = {"test_a", "test_b"}


def require_model_freeze_for_test(
    split: str,
    *,
    freeze_path: str | Path = MODEL_FREEZE_PATH,
) -> None:
    """Refuse access to formal test windows until the G4 model protocol is frozen."""

    if split not in TEST_SPLITS:
        return
    path = Path(freeze_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise RuntimeError(
            f"Formal split {split!r} is locked until G4 creates "
            "docs/frozen/MODEL_PROTOCOL_V1.json"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "frozen" or not record.get("model_protocol_hash"):
        raise RuntimeError("MODEL_PROTOCOL_V1.json is not a valid frozen model protocol")
    data_freeze = json.loads(
        (PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json").read_text(encoding="utf-8")
    )
    if record.get("data_protocol_hash") != data_freeze.get("protocol_hash"):
        raise RuntimeError("Model freeze does not reference the active frozen data protocol")
    for relative_path, expected_hash in record.get("config_hashes", {}).items():
        candidate = PROJECT_ROOT / relative_path
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise RuntimeError(f"Frozen model configuration changed: {relative_path}")
    ledger_path = PROJECT_ROOT / "artifacts/validation/trials.jsonl"
    if not ledger_path.is_file() or file_sha256(ledger_path) != record.get("validation_ledger_hash"):
        raise RuntimeError("Frozen validation ledger changed")
    # Imported lazily to keep ordinary validation runs independent of freeze construction.
    from .freezing import isolated_environment_paths, model_code_paths

    if tree_hash(model_code_paths()) != record.get("model_code_hash"):
        raise RuntimeError("Frozen model code changed")
    if tree_hash(isolated_environment_paths()) != record.get("isolated_environment_hash"):
        raise RuntimeError("Frozen isolated model environment changed")
