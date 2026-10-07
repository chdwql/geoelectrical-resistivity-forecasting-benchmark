from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from .aggregation import aggregate_source_files
from .config import PROJECT_ROOT, file_sha256, load_yaml, stable_hash


def source_paths(config: dict[str, Any]) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    template = config["source_glob"]
    for station in config["stations"]:
        for factor in config["factors"]:
            path = PROJECT_ROOT / template.format(station=station, factor=factor)
            result.append((station, factor, path))
    return result


def combined_source_paths(config: dict[str, Any], station: str, factor: str) -> list[Path]:
    current = PROJECT_ROOT / config["source_glob"].format(station=station, factor=factor)
    historical_glob = config.get("historical_glob", "data/raw/{station}/{factor}/*.parquet")
    historical = sorted(PROJECT_ROOT.glob(historical_glob.format(station=station, factor=factor)))
    return [*historical, *([current] if current.is_file() else [])]


def audit_csv(path: Path, station: str, factor: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            [{
                "station": station,
                "factor": factor,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "status": "missing",
            }]
        )
    connection = duckdb.connect()
    try:
        relation = connection.read_csv(str(path), all_varchar=True)
        relation.create_view("source")
        summary = connection.execute(
            """
            SELECT
                count(*) AS rows,
                min(try_cast(STARTDATE AS TIMESTAMP)) AS first_timestamp,
                max(try_cast(STARTDATE AS TIMESTAMP)) AS last_timestamp,
                count(DISTINCT trim(POINTID) || '/' || trim(ITEMID)) AS series_count,
                count(DISTINCT trim(SAMPLERATE)) AS sampling_rate_count,
                count(*) FILTER (WHERE try_cast(STARTDATE AS TIMESTAMP) IS NULL) AS invalid_timestamp_rows,
                count(*) FILTER (WHERE try_cast(OBSVALUE AS DOUBLE) IS NULL) AS invalid_value_rows
            FROM source
            """
        ).fetchdf()
        series = connection.execute(
            """
            SELECT DISTINCT trim(POINTID) AS point, trim(ITEMID) AS item,
                            trim(SAMPLERATE) AS sample_rate
            FROM source
            ORDER BY point, item, sample_rate
            """
        ).fetchdf()
    finally:
        connection.close()
    row = summary.iloc[0].to_dict()
    row.update(
        {
            "station": station,
            "factor": factor,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "status": "available",
            "series": json.dumps(series.to_dict(orient="records"), ensure_ascii=False),
        }
    )
    return pd.DataFrame([row])


def audit_sources(config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    frames = [audit_csv(path, station, factor) for station, factor, path in source_paths(config)]
    result = pd.concat(frames, ignore_index=True)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result


def validate_raw_acquisition(
    task_path: str | Path = "data/fetch_tasks_2016_2020.csv",
    raw_root: str | Path = "data/raw",
    output_path: str | Path = "artifacts/data_v1/raw_acquisition_audit.csv",
) -> pd.DataFrame:
    """Validate every planned Windows partition against logs, sidecars and hashes."""

    tasks_file = Path(task_path)
    if not tasks_file.is_absolute():
        tasks_file = PROJECT_ROOT / tasks_file
    raw = Path(raw_root)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    tasks = pd.read_csv(
        tasks_file,
        dtype={"task_id": str, "station": str, "point": str, "start": str, "end": str},
    )
    required_columns = {"STARTDATE", "STATIONID", "POINTID", "ITEMID", "SAMPLERATE", "OBSVALUE"}
    rows: list[dict[str, Any]] = []
    planned_ids = set(tasks["task_id"])
    discovered_ids: set[str] = set()
    for task in tasks.itertuples(index=False):
        errors: list[str] = []
        folder = raw / str(task.station) / str(task.factor)
        matches = list(folder.glob(f"*_{task.task_id}.parquet"))
        parquet = matches[0] if len(matches) == 1 else None
        if len(matches) != 1:
            errors.append(f"partition_count={len(matches)}")
        sidecar = parquet.with_suffix(".json") if parquet else None
        log_path = raw / "_logs" / f"{task.task_id}.json"
        side_record: dict[str, Any] = {}
        log_record: dict[str, Any] = {}
        if sidecar is None or not sidecar.is_file():
            errors.append("missing_sidecar")
        else:
            try:
                side_record = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"invalid_sidecar:{type(exc).__name__}")
        if not log_path.is_file():
            errors.append("missing_log")
        else:
            try:
                log_record = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"invalid_log:{type(exc).__name__}")

        metadata_rows: int | None = None
        actual_hash: str | None = None
        schema: list[str] = []
        if parquet is not None:
            discovered_ids.add(str(task.task_id))
            try:
                parquet_file = pq.ParquetFile(parquet)
                metadata_rows = parquet_file.metadata.num_rows
                schema = parquet_file.schema_arrow.names
                missing = required_columns - set(schema)
                if missing:
                    errors.append(f"missing_columns={','.join(sorted(missing))}")
                actual_hash = file_sha256(parquet)
            except Exception as exc:
                errors.append(f"invalid_parquet:{type(exc).__name__}:{exc}")

        expected_identity = {
            "task_id": str(task.task_id),
            "station": str(task.station),
            "point": str(task.point),
            "factor": str(task.factor),
            "sample_rate": str(task.sample_rate),
            "start": str(task.start),
            "end": str(task.end),
        }
        for name, expected in expected_identity.items():
            if str(side_record.get(name, "")) != expected:
                errors.append(f"sidecar_{name}_mismatch")
            if str(log_record.get(name, "")) != expected:
                errors.append(f"log_{name}_mismatch")
        side_status = str(side_record.get("status", "missing"))
        log_status = str(log_record.get("status", "missing"))
        if side_status not in {"success", "empty"} or log_status != side_status:
            errors.append("status_mismatch")
        if metadata_rows is not None:
            if int(side_record.get("rows", -1)) != metadata_rows:
                errors.append("row_count_mismatch")
            if (metadata_rows == 0) != (side_status == "empty"):
                errors.append("empty_status_mismatch")
        if actual_hash is not None and side_record.get("sha256") != actual_hash:
            errors.append("sha256_mismatch")
        first = pd.to_datetime(side_record.get("first_timestamp"), errors="coerce")
        last = pd.to_datetime(side_record.get("last_timestamp"), errors="coerce")
        if side_status == "success":
            start = pd.Timestamp(datetime.strptime(str(task.start), "%Y%m%d").date())
            end = pd.Timestamp(datetime.strptime(str(task.end), "%Y%m%d").date())
            if pd.isna(first) or pd.isna(last) or first.date() < start.date() or last.date() > end.date():
                errors.append("timestamp_bounds_mismatch")
        rows.append(
            {
                **expected_identity,
                "status": side_status,
                "rows": metadata_rows,
                "bytes": parquet.stat().st_size if parquet else None,
                "sha256": actual_hash,
                "schema": json.dumps(schema),
                "validation_status": "valid" if not errors else "invalid",
                "validation_errors": "|".join(errors),
                "path": str(parquet.relative_to(PROJECT_ROOT)) if parquet else None,
            }
        )

    for parquet in raw.glob("[0-9]*/**/*.parquet"):
        task_id = parquet.stem.rsplit("_", 1)[-1]
        if task_id not in planned_ids:
            discovered_ids.add(task_id)
            rows.append(
                {
                    "task_id": task_id,
                    "validation_status": "invalid",
                    "validation_errors": "unplanned_partition",
                    "path": str(parquet.relative_to(PROJECT_ROOT)),
                }
            )
    result = pd.DataFrame(rows)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result


def build_daily_dataset(config_path: str | Path, output_root: str | Path | None = None) -> list[Path]:
    config = load_yaml(config_path)
    root = PROJECT_ROOT / (output_root or config["derived_root"])
    outputs: list[Path] = []
    source_manifest: list[dict[str, Any]] = []
    for station in config["stations"]:
        for factor in config["factors"]:
            paths = combined_source_paths(config, station, factor)
            if not paths:
                continue
            factor_config = config["factors"][factor]
            daily = aggregate_source_files(
                paths,
                factor=factor,
                aggregation=factor_config["aggregation"],
                minimum_coverage=float(factor_config["minimum_daily_coverage"]),
                excluded_item_ids=config.get("excluded_item_ids", []) if factor == "resistivity" else (),
            )
            destination = root / "daily_q0" / f"station={station}" / f"factor={factor}" / "part.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".parquet.tmp")
            daily.to_parquet(temporary, index=False)
            os.replace(temporary, destination)
            outputs.append(destination)
            source_manifest.append(
                {
                    "station": station,
                    "factor": factor,
                    "sources": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
                    "daily_rows": len(daily),
                }
            )
    manifest = {
        "version": config["version"],
        "config_hash": stable_hash(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "partitions": [str(path.relative_to(PROJECT_ROOT)) for path in outputs],
        "source_partitions": source_manifest,
    }
    manifest_path = root / "daily_q0_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs
