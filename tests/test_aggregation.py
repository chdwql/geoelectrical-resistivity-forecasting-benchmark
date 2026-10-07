from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from geoelectrical_resistivity_forecasting_benchmark.aggregation import (
    aggregate_frame,
    aggregate_source_files,
)


def hourly_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-02-29", periods=24, freq="h")
    frame = pd.DataFrame(
        {
            "STARTDATE": timestamps,
            "STATIONID": "11003",
            "POINTID": "D",
            "ITEMID": "3211",
            "SAMPLERATE": "60",
            "OBSVALUE": np.arange(24, dtype=float),
        }
    )
    exact_duplicate = frame.iloc[[0]].copy()
    conflict = frame.iloc[[1]].copy()
    conflict["OBSVALUE"] = 999.0
    return pd.concat([frame, exact_duplicate, conflict], ignore_index=True)


def test_q0_deduplicates_and_marks_conflicts() -> None:
    daily = aggregate_frame(
        hourly_frame(),
        factor="resistivity",
        aggregation="mean",
        minimum_coverage=0.8,
    )
    row = daily.iloc[0]
    assert row["valid_count"] == 23
    assert row["expected_count"] == 24
    assert row["coverage"] == 23 / 24
    assert row["value"] == np.mean([0, *range(2, 24)])
    assert set(row["quality_flags"].split("|")) == {"conflicting_timestamp", "exact_duplicate"}


def test_low_coverage_never_creates_truth() -> None:
    daily = aggregate_frame(
        hourly_frame().iloc[:4],
        factor="resistivity",
        aggregation="mean",
        minimum_coverage=0.8,
    )
    assert pd.isna(daily.iloc[0]["value"])
    assert "low_coverage" in daily.iloc[0]["quality_flags"]


def test_rainfall_is_daily_sum_and_order_invariant() -> None:
    frame = hourly_frame().iloc[:24].copy()
    first = aggregate_frame(frame, factor="rainfall", aggregation="sum", minimum_coverage=0.8)
    second = aggregate_frame(
        frame.sample(frac=1, random_state=42),
        factor="rainfall",
        aggregation="sum",
        minimum_coverage=0.8,
    )
    assert first.iloc[0]["value"] == sum(range(24))
    pdt.assert_frame_equal(first, second)


def test_excluded_resistivity_items_are_removed() -> None:
    frame = hourly_frame()
    frame["ITEMID"] = "3215"
    daily = aggregate_frame(
        frame,
        factor="resistivity",
        aggregation="mean",
        minimum_coverage=0.8,
        excluded_item_ids=["3215"],
    )
    assert daily.empty


def test_legacy_zero_padded_sample_rate_is_normalised() -> None:
    frame = hourly_frame().iloc[:24].copy()
    frame["SAMPLERATE"] = "01"
    minutes = pd.date_range("2024-02-29", periods=24, freq="min")
    frame["STARTDATE"] = minutes
    daily = aggregate_frame(
        frame,
        factor="temperature",
        aggregation="mean",
        minimum_coverage=0.01,
    )
    assert daily.iloc[0]["expected_count"] == 1440
    assert daily.iloc[0]["coverage"] == 24 / 1440


def test_mixed_csv_parquet_production_path_matches_in_memory(tmp_path) -> None:
    frame = hourly_frame()
    csv_path = tmp_path / "first.csv"
    parquet_path = tmp_path / "second.parquet"
    csv_frame = frame.iloc[:13].assign(PROCESSINGFLAG=None, DATE_INDEX=range(13))
    parquet_frame = frame.iloc[13:].assign(PROCESSINGFLAG=None, DATE_INDEX=range(13, len(frame)))
    csv_frame[
        [
            "STARTDATE",
            "STATIONID",
            "POINTID",
            "ITEMID",
            "SAMPLERATE",
            "PROCESSINGFLAG",
            "DATE_INDEX",
            "OBSVALUE",
        ]
    ].to_csv(csv_path, index=False)
    parquet_frame[
        [
            "STARTDATE",
            "STATIONID",
            "POINTID",
            "ITEMID",
            "SAMPLERATE",
            "OBSVALUE",
            "PROCESSINGFLAG",
            "DATE_INDEX",
        ]
    ].to_parquet(parquet_path, index=False)

    expected = aggregate_frame(
        frame,
        factor="resistivity",
        aggregation="mean",
        minimum_coverage=0.8,
    ).drop(columns="source_hash")
    actual = aggregate_source_files(
        [csv_path, parquet_path],
        factor="resistivity",
        aggregation="mean",
        minimum_coverage=0.8,
    ).drop(columns="source_hash")
    actual["valid_count"] = actual["valid_count"].astype("int64")
    actual["expected_count"] = actual["expected_count"].astype("int64")
    pdt.assert_frame_equal(actual, expected, check_dtype=False)
