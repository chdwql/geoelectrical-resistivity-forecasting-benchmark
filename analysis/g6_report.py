from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FREEZE = PROJECT_ROOT / "docs/frozen/DATA_PROTOCOL_V1.json"
MODEL_FREEZE = PROJECT_ROOT / "docs/frozen/MODEL_PROTOCOL_V1.json"
TASK_PLAN = PROJECT_ROOT / "artifacts/formal/task_plan.jsonl"
WINDOWS = PROJECT_ROOT / "artifacts/data_v1/windows.parquet"
FOUNDATION_MODELS = ("timesfm_3", "chronos_2", "moirai_2_small", "toto_2_313m")
H1_REFERENCES = ("seasonal_365", "fourier_ridge")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def manifest_identity(manifest: dict[str, object]) -> dict[str, object]:
    identity = manifest.get("identity")
    if isinstance(identity, dict):
        return identity
    request = manifest.get("request_manifest")
    if isinstance(request, dict) and isinstance(request.get("identity"), dict):
        return request["identity"]
    return {}


def _nullable_identity(identity: dict[str, object], key: str) -> object:
    return identity.get(key)


def discover_runs(
    *,
    data_protocol_hash: str,
    input_variant: str,
    context_days: int,
) -> pd.DataFrame:
    task_files = sorted((PROJECT_ROOT / "artifacts").rglob("tasks.parquet"))
    rows: list[dict[str, object]] = []
    for task_path in task_files:
        directory = task_path.parent
        manifest_path = directory / "manifest.json"
        predictions_path = directory / "predictions.parquet"
        scores_path = directory / "scores.parquet"
        if not (manifest_path.is_file() and predictions_path.is_file() and scores_path.is_file()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest_identity(manifest)
        if identity.get("protocol_hash") != data_protocol_hash:
            continue
        if identity.get("split") not in {"test_a", "test_b"}:
            continue
        if identity.get("input_variant", "Q0") != input_variant:
            continue
        if int(identity.get("context_days", 730)) != context_days:
            continue
        if _nullable_identity(identity, "stations") is not None:
            continue
        if _nullable_identity(identity, "limit") is not None:
            continue

        tasks = pd.read_parquet(task_path)
        for model_id, group in tasks.groupby("model_id", sort=True):
            if "track" in group.columns and group["track"].notna().any():
                tracks = group["track"].dropna().astype(str).unique().tolist()
                if len(tracks) != 1:
                    raise RuntimeError(f"Ambiguous tracks in {task_path}: {tracks}")
                track = tracks[0]
            else:
                track = str(identity.get("track") or "U")
            status_counts = group["status"].value_counts(dropna=False).to_dict()
            rows.append(
                {
                    "model_id": str(model_id),
                    "split": str(identity["split"]),
                    "horizon": int(identity["horizon"]),
                    "track": track,
                    "input_variant": input_variant,
                    "context_days": context_days,
                    "run_id": str(group["run_id"].iloc[0]),
                    "run_directory": str(directory.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "task_count": int(len(group)),
                    "success_count": int(status_counts.get("success", 0)),
                    "error_count": int(len(group) - status_counts.get("success", 0)),
                    "predictions_sha256": file_sha256(predictions_path),
                    "tasks_sha256": file_sha256(task_path),
                    "scores_sha256": file_sha256(scores_path),
                    "manifest_sha256": file_sha256(manifest_path),
                }
            )
    return pd.DataFrame(rows)


KEY_COLUMNS = ["model_id", "split", "horizon", "track"]


def validate_standard_index(index: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    planned = plan.loc[plan["status"].eq("planned")].copy()
    duplicates = index.loc[index.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS + ["run_directory"]]
    if not duplicates.empty:
        raise RuntimeError(f"Duplicate formal runs:\n{duplicates.to_string(index=False)}")
    expected = set(map(tuple, planned[KEY_COLUMNS].itertuples(index=False, name=None)))
    observed = set(map(tuple, index[KEY_COLUMNS].itertuples(index=False, name=None)))
    if expected != observed:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(f"Formal run index mismatch; missing={missing}, unexpected={unexpected}")
    coverage = plan.merge(index, how="left", on=KEY_COLUMNS, validate="one_to_one")
    runnable = coverage["status"].eq("planned")
    wrong_counts = coverage.loc[runnable & coverage["task_count"].ne(coverage["eligible_windows"])]
    if not wrong_counts.empty:
        raise RuntimeError(
            "Task denominator mismatch:\n"
            + wrong_counts[KEY_COLUMNS + ["eligible_windows", "task_count"]].to_string(index=False)
        )
    coverage["final_status"] = np.where(
        coverage["status"].eq("N/A"),
        "N/A",
        np.where(coverage["error_count"].fillna(0).gt(0), "completed_with_failures", "success"),
    )
    coverage["task_coverage"] = coverage["success_count"] / coverage["eligible_windows"]
    return coverage


def build_score_ledger(index: pd.DataFrame) -> pd.DataFrame:
    windows = pd.read_parquet(WINDOWS)[
        ["window_id", "series_id", "origin", "target_start", "target_end", "track"]
    ].rename(columns={"track": "window_track"})
    rows: list[pd.DataFrame] = []
    for record in index.itertuples(index=False):
        directory = PROJECT_ROOT / record.run_directory
        scores = pd.read_parquet(directory / "scores.parquet")
        scores = scores.loc[scores["model_id"].eq(record.model_id)].copy()
        if len(scores) != record.success_count:
            raise RuntimeError(
                f"Score/task mismatch for {record.run_directory}/{record.model_id}: "
                f"scores={len(scores)}, successes={record.success_count}"
            )
        scores = scores.merge(windows, on=["window_id", "series_id"], how="left", validate="one_to_one")
        if scores["origin"].isna().any():
            raise RuntimeError(f"Unknown window ids in {record.run_directory}")
        if not scores["window_track"].eq(record.track).all():
            raise RuntimeError(f"Track mismatch in {record.run_directory}/{record.model_id}")
        scores["track"] = record.track
        scores["run_directory"] = record.run_directory
        scores["origin"] = pd.to_datetime(scores["origin"])
        rows.append(scores)
    ledger = pd.concat(rows, ignore_index=True)
    key = ["model_id", "split", "horizon", "track", "series_id", "origin"]
    if ledger.duplicated(key).any():
        raise RuntimeError("Unified score ledger contains duplicate logical windows")
    return ledger.sort_values(key).reset_index(drop=True)


def hierarchical_effect(pairs: pd.DataFrame, value_column: str = "delta_mase") -> float:
    by_series = pairs.groupby(["station", "series_id"], as_index=False)[value_column].mean()
    by_station = by_series.groupby("station", as_index=False)[value_column].mean()
    return float(by_station[value_column].mean())


def moving_block_bootstrap(
    pairs: pd.DataFrame,
    *,
    draws: int,
    block_origins: int,
    seed: int,
    value_column: str = "delta_mase",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Precompute one origin-by-series matrix per station.  Selecting matrix rows
    # with repetition is algebraically equivalent to the former concat/groupby
    # implementation, but avoids rebuilding pandas objects for every draw.
    stations: dict[str, np.ndarray] = {}
    for station, group in pairs.groupby("station", sort=True):
        station_frame = group.assign(origin=pd.to_datetime(group["origin"]))
        matrix = station_frame.pivot(
            index="origin", columns="series_id", values=value_column
        ).sort_index()
        stations[str(station)] = matrix.to_numpy(dtype=float)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_stations: list[float] = []
        for matrix in stations.values():
            n = matrix.shape[0]
            sampled_indices: list[int] = []
            while len(sampled_indices) < n:
                start = int(rng.integers(0, n))
                for offset in range(block_origins):
                    sampled_indices.append((start + offset) % n)
                    if len(sampled_indices) == n:
                        break
            sampled = matrix[np.asarray(sampled_indices), :]
            series_means = np.nanmean(sampled, axis=0)
            sampled_stations.append(float(np.nanmean(series_means)))
        estimates[draw] = float(np.mean(sampled_stations))
    return estimates


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * values[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def paired_comparison(
    ledger: pd.DataFrame,
    *,
    candidate: str,
    reference: str,
    split: str,
    candidate_track: str = "U",
    reference_track: str = "U",
    draws: int,
    block_origins: int,
    seed: int,
) -> dict[str, object]:
    keys = ["series_id", "station", "split", "horizon", "origin"]
    left = ledger.loc[
        ledger["model_id"].eq(candidate)
        & ledger["split"].eq(split)
        & ledger["horizon"].eq(90)
        & ledger["track"].eq(candidate_track),
        keys + ["mase"],
    ].rename(columns={"mase": "candidate_mase"})
    right = ledger.loc[
        ledger["model_id"].eq(reference)
        & ledger["split"].eq(split)
        & ledger["horizon"].eq(90)
        & ledger["track"].eq(reference_track),
        keys + ["mase"],
    ].rename(columns={"mase": "reference_mase"})
    pairs = left.merge(right, on=keys, how="inner", validate="one_to_one")
    pairs["delta_mase"] = pairs["candidate_mase"] - pairs["reference_mase"]
    effect = hierarchical_effect(pairs)
    bootstrap = moving_block_bootstrap(
        pairs,
        draws=draws,
        block_origins=block_origins,
        seed=seed,
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    below = (np.count_nonzero(bootstrap <= 0) + 1) / (draws + 1)
    above = (np.count_nonzero(bootstrap >= 0) + 1) / (draws + 1)
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_track": candidate_track,
        "reference_track": reference_track,
        "split": split,
        "horizon": 90,
        "paired_windows": int(len(pairs)),
        "paired_series": int(pairs["series_id"].nunique()),
        "paired_stations": int(pairs["station"].nunique()),
        "effect_delta_mase": effect,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_two_sided": float(min(1.0, 2 * min(below, above))),
        "bootstrap_draws": draws,
        "block_origins": block_origins,
    }


def primary_comparisons(
    ledger: pd.DataFrame, *, draws: int, block_origins: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h1_rows: list[dict[str, object]] = []
    offset = 0
    for split in ("test_a", "test_b"):
        for model in FOUNDATION_MODELS:
            for reference in H1_REFERENCES:
                h1_rows.append(
                    paired_comparison(
                        ledger,
                        candidate=model,
                        reference=reference,
                        split=split,
                        draws=draws,
                        block_origins=block_origins,
                        seed=seed + offset,
                    )
                )
                offset += 1
    h1 = pd.DataFrame(h1_rows)
    h1["holm_p_value"] = holm_adjust(h1["p_value_two_sided"])
    h1["direction"] = np.where(h1["effect_delta_mase"].lt(0), "candidate_better", "reference_better")

    available = set(
        map(
            tuple,
            ledger.loc[ledger["horizon"].eq(90), ["model_id", "split", "track"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    h3_rows: list[dict[str, object]] = []
    for split in ("test_a", "test_b"):
        models = sorted(ledger.loc[ledger["horizon"].eq(90), "model_id"].unique())
        for model in models:
            if (model, split, "U") not in available:
                continue
            for track in ("M", "X"):
                if (model, split, track) not in available:
                    continue
                h3_rows.append(
                    paired_comparison(
                        ledger,
                        candidate=model,
                        reference=model,
                        split=split,
                        candidate_track=track,
                        reference_track="U",
                        draws=draws,
                        block_origins=block_origins,
                        seed=seed + offset,
                    )
                )
                offset += 1
    h3 = pd.DataFrame(h3_rows)
    h3["holm_p_value"] = holm_adjust(h3["p_value_two_sided"])
    h3["direction"] = np.where(h3["effect_delta_mase"].lt(0), "added_information_better", "U_better")
    return h1, h3


def build_report(output_root: Path, *, draws: int, block_origins: int, seed: int) -> dict[str, object]:
    data_freeze = json.loads(DATA_FREEZE.read_text(encoding="utf-8"))
    model_freeze = json.loads(MODEL_FREEZE.read_text(encoding="utf-8"))
    plan = pd.DataFrame(read_jsonl(TASK_PLAN))
    index = discover_runs(
        data_protocol_hash=str(data_freeze["protocol_hash"]),
        input_variant="Q0",
        context_days=730,
    )
    coverage = validate_standard_index(index, plan)
    ledger = build_score_ledger(index)
    h1, h3 = primary_comparisons(
        ledger, draws=draws, block_origins=block_origins, seed=seed
    )

    outputs = {
        "run_index.csv": index.sort_values(KEY_COLUMNS).reset_index(drop=True),
        "task_coverage.csv": coverage.sort_values(KEY_COLUMNS).reset_index(drop=True),
        "window_scores.parquet": ledger,
        "h1_primary_comparisons.csv": h1,
        "h3_information_gain.csv": h3,
    }
    for filename, frame in outputs.items():
        destination = output_root / filename
        if destination.suffix == ".parquet":
            atomic_parquet(frame, destination)
        else:
            atomic_csv(frame, destination)

    manifest = {
        "report_version": "g6-report-v1",
        "data_protocol_hash": data_freeze["protocol_hash"],
        "model_protocol_hash": model_freeze["model_protocol_hash"],
        "task_plan_sha256": file_sha256(TASK_PLAN),
        "settings": {"bootstrap_draws": draws, "block_origins": block_origins, "seed": seed},
        "counts": {
            "logical_specs": int(len(coverage)),
            "runnable_specs": int(coverage["status"].eq("planned").sum()),
            "n_a_specs": int(coverage["status"].eq("N/A").sum()),
            "planned_tasks": int(coverage.loc[coverage["status"].eq("planned"), "eligible_windows"].sum()),
            "successful_tasks": int(index["success_count"].sum()),
            "failed_tasks": int(index["error_count"].sum()),
            "score_rows": int(len(ledger)),
            "h1_comparisons": int(len(h1)),
            "h3_comparisons": int(len(h3)),
        },
        "outputs": {
            filename: {"sha256": file_sha256(output_root / filename), "rows": int(len(frame))}
            for filename, frame in outputs.items()
        },
        "notes": [
            "Negative delta MASE favors the candidate or added-information track.",
            "H1 compares each frozen foundation model with seasonal persistence and validation-selected Fourier ridge.",
            "H3 pairs tracks by series, origin, split and horizon; track-specific window ids are never paired directly.",
            "Circular moving blocks contain three monthly origins, matching the 90-day overlapping target horizon.",
        ],
    }
    atomic_json(manifest, output_root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the post-freeze G6 statistical report.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6"))
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--block-origins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest = build_report(
        output,
        draws=args.bootstrap_draws,
        block_origins=args.block_origins,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
