from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from .config import file_sha256, stable_hash


REQUIRED_COLUMNS = {
    "STARTDATE",
    "STATIONID",
    "POINTID",
    "ITEMID",
    "SAMPLERATE",
    "OBSVALUE",
}

EXPECTED_PER_DAY = {"1": 1440, "60": 24, "90": 1}


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).upper() for column in result.columns]
    missing = REQUIRED_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result["STARTDATE"] = pd.to_datetime(result["STARTDATE"], errors="coerce")
    result["OBSVALUE"] = pd.to_numeric(result["OBSVALUE"], errors="coerce")
    for column in ("STATIONID", "POINTID", "ITEMID", "SAMPLERATE"):
        result[column] = result[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    result["SAMPLERATE"] = result["SAMPLERATE"].replace({"01": "1", "02": "2"})
    return result.dropna(subset=["STARTDATE", "STATIONID", "POINTID", "ITEMID", "OBSVALUE"])


def aggregate_frame(
    frame: pd.DataFrame,
    *,
    factor: str,
    aggregation: str,
    minimum_coverage: float,
    excluded_item_ids: Iterable[str] = (),
    source_hash: str = "synthetic",
) -> pd.DataFrame:
    """Build auditable Q0 daily values from an in-memory source frame."""

    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'mean' or 'sum'")
    data = _normalise_frame(frame)
    excluded = {str(item) for item in excluded_item_ids}
    if excluded:
        data = data.loc[~data["ITEMID"].isin(excluded)].copy()
    data["DATE"] = data["STARTDATE"].dt.normalize()
    keys = ["STATIONID", "POINTID", "ITEMID", "STARTDATE"]
    timestamp = (
        data.groupby(keys, dropna=False)
        .agg(
            OBSVALUE=("OBSVALUE", "first"),
            VALUE_COUNT=("OBSVALUE", "nunique"),
            RECORD_COUNT=("OBSVALUE", "size"),
            SAMPLERATE=("SAMPLERATE", "first"),
            RATE_COUNT=("SAMPLERATE", "nunique"),
            DATE=("DATE", "first"),
        )
        .reset_index()
    )
    timestamp["CONFLICT"] = timestamp["VALUE_COUNT"] > 1
    timestamp["EXACT_DUPLICATE"] = (timestamp["RECORD_COUNT"] > 1) & ~timestamp["CONFLICT"]
    timestamp.loc[timestamp["CONFLICT"], "OBSVALUE"] = pd.NA
    timestamp["EXPECTED"] = timestamp["SAMPLERATE"].map(EXPECTED_PER_DAY)

    group_keys = ["STATIONID", "POINTID", "ITEMID", "DATE"]
    grouped = timestamp.groupby(group_keys, dropna=False)
    daily = grouped.agg(
        value_sum=("OBSVALUE", "sum"),
        value_mean=("OBSVALUE", "mean"),
        valid_count=("OBSVALUE", "count"),
        expected_count=("EXPECTED", "max"),
        conflict_count=("CONFLICT", "sum"),
        exact_duplicate_count=("EXACT_DUPLICATE", "sum"),
        sampling_rate_count=("SAMPLERATE", "nunique"),
    ).reset_index()
    daily["coverage"] = daily["valid_count"] / daily["expected_count"]
    daily["value"] = daily["value_sum"] if aggregation == "sum" else daily["value_mean"]
    low_coverage = daily["expected_count"].isna() | (daily["coverage"] < minimum_coverage)
    daily.loc[low_coverage, "value"] = pd.NA

    def flags(row: pd.Series) -> str:
        values: list[str] = []
        if row["conflict_count"]:
            values.append("conflicting_timestamp")
        if row["exact_duplicate_count"]:
            values.append("exact_duplicate")
        if row["sampling_rate_count"] > 1 or pd.isna(row["expected_count"]):
            values.append("unknown_or_mixed_sampling_rate")
        if pd.isna(row["expected_count"]) or row["coverage"] < minimum_coverage:
            values.append("low_coverage")
        return "|".join(values)

    daily["quality_flags"] = daily.apply(flags, axis=1)
    daily["factor"] = factor
    daily["source_hash"] = source_hash
    return daily[
        [
            "DATE",
            "STATIONID",
            "POINTID",
            "ITEMID",
            "factor",
            "value",
            "valid_count",
            "expected_count",
            "coverage",
            "quality_flags",
            "source_hash",
        ]
    ].rename(
        columns={"DATE": "date", "STATIONID": "station", "POINTID": "point", "ITEMID": "item"}
    )


def read_source_csv(paths: list[Path]) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        relation = connection.read_csv([str(path) for path in paths], all_varchar=True, union_by_name=True)
        return relation.df()
    finally:
        connection.close()


def aggregate_csv_files(
    paths: list[Path],
    *,
    factor: str,
    aggregation: str,
    minimum_coverage: float,
    excluded_item_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate one or more complete source partitions.

    DuckDB performs CSV parsing, while conflict resolution is deliberately shared
    with the tested in-memory implementation. The caller should partition large
    histories by station and factor to bound memory use.
    """

    if not paths:
        raise ValueError("At least one source path is required")
    source_hash = stable_hash({str(path): file_sha256(path) for path in paths})
    return aggregate_frame(
        read_source_csv(paths),
        factor=factor,
        aggregation=aggregation,
        minimum_coverage=minimum_coverage,
        excluded_item_ids=excluded_item_ids,
        source_hash=source_hash,
    )


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def aggregate_source_files(
    paths: list[Path],
    *,
    factor: str,
    aggregation: str,
    minimum_coverage: float,
    excluded_item_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate mixed CSV/Parquet partitions without loading raw rows into pandas.

    This is the production Q0 path for the combined 2016--2026 history. DuckDB
    performs normalization, duplicate/conflict resolution and daily aggregation;
    only the much smaller daily result is materialized in memory.
    """

    if not paths:
        raise ValueError("At least one source path is required")
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'mean' or 'sum'")
    csv_paths = [str(path) for path in paths if path.suffix.lower() == ".csv"]
    parquet_paths = [str(path) for path in paths if path.suffix.lower() == ".parquet"]
    unsupported = [path for path in paths if path.suffix.lower() not in {".csv", ".parquet"}]
    if unsupported:
        raise ValueError(f"Unsupported source formats: {unsupported}")

    source_hash = stable_hash({str(path): file_sha256(path) for path in paths})
    excluded = {_sql_text(str(item)) for item in excluded_item_ids}
    exclusion_sql = f"AND item NOT IN ({','.join(sorted(excluded))})" if excluded else ""
    value_expression = "value_sum" if aggregation == "sum" else "value_mean"

    connection = duckdb.connect()
    try:
        source_views: list[str] = []
        if csv_paths:
            connection.read_csv(csv_paths, all_varchar=True, union_by_name=True).create_view("csv_source")
            source_views.append("SELECT * FROM csv_source")
        if parquet_paths:
            connection.read_parquet(parquet_paths, union_by_name=True).create_view("parquet_source")
            source_views.append("SELECT * FROM parquet_source")
        # CSV and Parquet snapshots can expose the same named fields in a
        # different physical order. A positional UNION silently mapped legacy
        # DATE_INDEX values into OBSVALUE; the production contract is by name.
        source_sql = " UNION ALL BY NAME ".join(source_views)
        query = f"""
            WITH source AS (
                {source_sql}
            ), normalized AS (
                SELECT
                    try_cast(STARTDATE AS TIMESTAMP) AS observed_at,
                    regexp_replace(trim(cast(STATIONID AS VARCHAR)), '\\.0$', '') AS station,
                    regexp_replace(trim(cast(POINTID AS VARCHAR)), '\\.0$', '') AS point,
                    regexp_replace(trim(cast(ITEMID AS VARCHAR)), '\\.0$', '') AS item,
                    CASE regexp_replace(trim(cast(SAMPLERATE AS VARCHAR)), '\\.0$', '')
                        WHEN '01' THEN '1'
                        WHEN '02' THEN '2'
                        ELSE regexp_replace(trim(cast(SAMPLERATE AS VARCHAR)), '\\.0$', '')
                    END AS sample_rate,
                    try_cast(OBSVALUE AS DOUBLE) AS observed_value
                FROM source
            ), valid AS (
                SELECT *
                FROM normalized
                WHERE observed_at IS NOT NULL
                  AND station IS NOT NULL AND station <> ''
                  AND point IS NOT NULL AND point <> ''
                  AND item IS NOT NULL AND item <> ''
                  AND observed_value IS NOT NULL
                  {exclusion_sql}
            ), timestamp_level AS (
                SELECT
                    station, point, item, observed_at,
                    first(observed_value) AS first_value,
                    count(DISTINCT observed_value) AS value_count,
                    count(*) AS record_count,
                    first(sample_rate) AS sample_rate
                FROM valid
                GROUP BY station, point, item, observed_at
            ), timestamp_clean AS (
                SELECT
                    station, point, item, observed_at, sample_rate,
                    CASE WHEN value_count > 1 THEN NULL ELSE first_value END AS observed_value,
                    value_count > 1 AS conflict,
                    record_count > 1 AND value_count <= 1 AS exact_duplicate,
                    CASE sample_rate WHEN '1' THEN 1440 WHEN '60' THEN 24 WHEN '90' THEN 1 END AS expected
                FROM timestamp_level
            ), daily_base AS (
                SELECT
                    station, point, item, cast(observed_at AS DATE) AS date,
                    coalesce(sum(observed_value), 0) AS value_sum,
                    avg(observed_value) AS value_mean,
                    count(observed_value) AS valid_count,
                    max(expected) AS expected_count,
                    sum(cast(conflict AS INTEGER)) AS conflict_count,
                    sum(cast(exact_duplicate AS INTEGER)) AS exact_duplicate_count,
                    count(DISTINCT sample_rate) AS sampling_rate_count
                FROM timestamp_clean
                GROUP BY station, point, item, cast(observed_at AS DATE)
            ), scored AS (
                SELECT *, valid_count::DOUBLE / expected_count AS coverage
                FROM daily_base
            )
            SELECT
                date, station, point, item,
                {_sql_text(factor)} AS factor,
                CASE
                    WHEN expected_count IS NULL OR coverage < {float(minimum_coverage)} THEN NULL
                    ELSE {value_expression}
                END AS value,
                valid_count, expected_count, coverage,
                concat_ws('|',
                    CASE WHEN conflict_count > 0 THEN 'conflicting_timestamp' END,
                    CASE WHEN exact_duplicate_count > 0 THEN 'exact_duplicate' END,
                    CASE WHEN sampling_rate_count > 1 OR expected_count IS NULL
                         THEN 'unknown_or_mixed_sampling_rate' END,
                    CASE WHEN expected_count IS NULL OR coverage < {float(minimum_coverage)}
                         THEN 'low_coverage' END
                ) AS quality_flags,
                {_sql_text(source_hash)} AS source_hash
            FROM scored
            ORDER BY station, point, item, date
        """
        return connection.execute(query).fetchdf()
    finally:
        connection.close()
