from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT, file_sha256, stable_hash


FACTOR_QUERIES: dict[str, tuple[str, str]] = {
    "resistivity": ("电阻率", "小时值"),
    "water_level": ("水位", "分钟值"),
    "temperature": ("温度", "分钟值"),
    "rainfall": ("降水", "分钟值"),
}


def _fetching_module():
    try:
        return importlib.import_module("addereq.fetching")
    except ImportError as exc:
        raise RuntimeError(
            "addereq is unavailable in the WSL ingestion environment; run bootstrap.sh --ingestion first"
        ) from exc


def probe_oracle(alias: str = "cn") -> dict[str, Any]:
    fetching = _fetching_module()
    connection = fetching.conn_to_Oracle(alias)
    if connection is None:
        raise RuntimeError(
            "Oracle connection returned None. Check Linux Instant Client and ~/.adder/default.conf; "
            "the benchmark will not fall back to the six-year protocol automatically."
        )
    try:
        metadata, items, methods = fetching.timeseries_info(connection)
        return {
            "alias": alias,
            "cache_namespace": connection.info.get("addereq_cache_namespace"),
            "station_rows": len(metadata),
            "item_rows": len(items),
            "method_rows": len(methods),
        }
    finally:
        connection.close()


def fetch_partition(
    *,
    station_id: str,
    point: str,
    factor: str,
    start: str,
    end: str,
    sample_rate: str | None = None,
    alias: str = "cn",
    output_root: str | Path = "data/raw",
    connection: Any | None = None,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if factor not in FACTOR_QUERIES:
        raise ValueError(f"Unsupported factor: {factor}")
    fetching = _fetching_module()
    destination = PROJECT_ROOT / output_root / station_id / factor / f"{start}_{end}_{point}.parquet"
    sidecar = destination.with_suffix(".json")
    if destination.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("sha256") == file_sha256(destination):
            return {**record, "status": "skipped_existing"}

    owns_connection = connection is None
    if connection is None:
        connection = fetching.conn_to_Oracle(alias)
        if connection is None:
            raise RuntimeError("Oracle connection failed; no output was written")
    try:
        if metadata is None:
            metadata, _, _ = fetching.timeseries_info(connection)
        metadata = metadata.rename(columns={column: str(column).upper() for column in metadata.columns})
        station_rows = metadata.loc[metadata["STATIONID"].astype(str).str.strip().eq(station_id)]
        if station_rows.empty:
            raise ValueError(f"Station not found in metadata: {station_id}")
        station_name = str(station_rows.iloc[0]["STATIONNAME"])
        method, default_sample_rate = FACTOR_QUERIES[factor]
        query_sample_rate = sample_rate or default_sample_rate
        frame = fetching.fetching_data(
            connection,
            start,
            end,
            method,
            station_name,
            query_sample_rate,
            "预处理库",
            itemname="全部",
            gzip_flag=False,
            my_split_lines=True,
            pointid=point,
        )
    finally:
        if owns_connection:
            connection.close()

    if frame is None:
        frame = pd.DataFrame()
    else:
        frame = frame.copy()
        if frame.index.name and frame.index.name not in frame.columns:
            frame = frame.reset_index()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    record = {
        "task_id": stable_hash([alias, station_id, point, factor, start, end, sample_rate])[:24],
        "status": "success" if not frame.empty else "empty",
        "station": station_id,
        "point": point,
        "factor": factor,
        "start": start,
        "end": end,
        "sample_rate": sample_rate or FACTOR_QUERIES[factor][1],
        "rows": len(frame),
        "sha256": file_sha256(destination),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": str(destination.relative_to(PROJECT_ROOT)),
    }
    sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def build_fetch_tasks(
    audit_path: str | Path = "artifacts/data_v1/source_audit.csv",
    output_path: str | Path = "artifacts/data_v1/fetch_tasks_2016_2020.csv",
) -> pd.DataFrame:
    audit_file = PROJECT_ROOT / audit_path
    audit = pd.read_csv(audit_file, dtype={"station": str})
    tasks: list[dict[str, Any]] = []
    for row in audit.itertuples(index=False):
        if row.status != "available":
            continue
        series = json.loads(row.series)
        point_rates: dict[str, set[str]] = {}
        for entry in series:
            point_rates.setdefault(str(entry["point"]), set()).add(str(entry["sample_rate"]))
        for point, rates in sorted(point_rates.items()):
            if row.factor == "water_level" and "90" in rates:
                sample_rate = "日值"
                periods = [(f"{year}0101", f"{year}1231") for year in range(2016, 2021)]
            elif row.factor == "resistivity":
                sample_rate = "小时值"
                periods = [(f"{year}0101", f"{year}1231") for year in range(2016, 2021)]
            else:
                sample_rate = "分钟值"
                periods = []
                for month_start in pd.date_range("2016-01-01", "2020-12-01", freq="MS"):
                    month_end = month_start + pd.offsets.MonthEnd(0)
                    periods.append((month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")))
            for start, end in periods:
                identity = [str(row.station), point, row.factor, sample_rate, start, end]
                tasks.append(
                    {
                        "task_id": stable_hash(identity)[:24],
                        "station": str(row.station),
                        "point": point,
                        "factor": row.factor,
                        "sample_rate": sample_rate,
                        "start": start,
                        "end": end,
                        "status": "planned",
                    }
                )
    result = pd.DataFrame(tasks).sort_values(["station", "factor", "point", "start"])
    destination = PROJECT_ROOT / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return result


def run_fetch_tasks(
    task_path: str | Path,
    *,
    limit: int | None = None,
    alias: str = "cn",
) -> list[dict[str, Any]]:
    tasks = pd.read_csv(PROJECT_ROOT / task_path, dtype={"station": str, "point": str})
    tasks = tasks.loc[tasks["status"].isin(["planned", "error"])]
    if limit is not None:
        tasks = tasks.head(limit)
    fetching = _fetching_module()
    connection = fetching.conn_to_Oracle(alias)
    if connection is None:
        raise RuntimeError(
            "Oracle connection failed before task execution; no task was attempted. "
            "Run `fetch --probe` after installing Linux Instant Client."
        )
    metadata, _, _ = fetching.timeseries_info(connection)
    records: list[dict[str, Any]] = []
    log_root = PROJECT_ROOT / "artifacts/data_v1/fetch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        for task in tasks.itertuples(index=False):
            try:
                record = fetch_partition(
                    station_id=task.station,
                    point=task.point,
                    factor=task.factor,
                    sample_rate=task.sample_rate,
                    start=str(task.start),
                    end=str(task.end),
                    alias=alias,
                    connection=connection,
                    metadata=metadata,
                )
            except Exception as exc:
                record = {
                    "task_id": task.task_id,
                    "status": "error",
                    "station": task.station,
                    "point": task.point,
                    "factor": task.factor,
                    "start": str(task.start),
                    "end": str(task.end),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            records.append(record)
            log_path = log_root / f"{task.task_id}.json"
            temporary_log = log_path.with_suffix(".json.tmp")
            temporary_log.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary_log, log_path)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        connection.close()
    return records


def summarise_fetch_tasks(
    task_path: str | Path = "artifacts/data_v1/fetch_tasks_2016_2020.csv",
    log_root: str | Path = "artifacts/data_v1/fetch_logs",
) -> pd.DataFrame:
    tasks = pd.read_csv(PROJECT_ROOT / task_path, dtype={"station": str, "point": str})
    logs: dict[str, dict[str, Any]] = {}
    for path in (PROJECT_ROOT / log_root).glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        logs[str(record["task_id"])] = record
    tasks["observed_status"] = tasks["task_id"].map(
        lambda task_id: logs.get(str(task_id), {}).get("status", "not_attempted")
    )
    return (
        tasks.groupby(["station", "factor", "observed_status"], as_index=False)
        .size()
        .rename(columns={"size": "tasks"})
    )
