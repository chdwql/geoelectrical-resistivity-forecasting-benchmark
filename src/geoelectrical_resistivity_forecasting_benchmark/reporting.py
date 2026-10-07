from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


def summarise_derived(
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    windows_path: str | Path = "artifacts/data_v1/windows.parquet",
    output_root: str | Path = "artifacts/data_v1",
) -> dict[str, pd.DataFrame]:
    daily_paths = sorted(PROJECT_ROOT.glob(daily_glob))
    if not daily_paths:
        raise FileNotFoundError(f"No daily partitions matched {daily_glob}")
    daily = pd.concat([pd.read_parquet(path) for path in daily_paths], ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["series_id"] = daily[["station", "point", "item"]].astype(str).agg("/".join, axis=1)
    coverage = (
        daily.groupby(["station", "factor", "year"], as_index=False)
        .agg(
            series_count=("series_id", "nunique"),
            represented_days=("date", "nunique"),
            rows=("value", "size"),
            valid_rows=("value", "count"),
            mean_source_coverage=("coverage", "mean"),
        )
    )
    coverage["valid_row_fraction"] = coverage["valid_rows"] / coverage["rows"]
    catalog = (
        daily.groupby(["station", "point", "item", "factor", "series_id"], as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            represented_days=("date", "nunique"),
            valid_days=("value", "count"),
            mean_source_coverage=("coverage", "mean"),
        )
    )
    catalog["calendar_span_days"] = (
        pd.to_datetime(catalog["last_date"]) - pd.to_datetime(catalog["first_date"])
    ).dt.days + 1
    catalog["valid_day_fraction"] = catalog["valid_days"] / catalog["calendar_span_days"]

    windows = pd.read_parquet(PROJECT_ROOT / windows_path)
    window_summary = (
        windows.groupby(["split", "horizon", "track", "eligible", "exclusion_reason"], dropna=False)
        .size()
        .rename("windows")
        .reset_index()
    )
    root = PROJECT_ROOT / output_root
    root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(root / "coverage_by_year.csv", index=False, encoding="utf-8-sig")
    catalog.to_csv(root / "series_catalog.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(root / "window_summary.csv", index=False, encoding="utf-8-sig")
    return {"coverage": coverage, "catalog": catalog, "windows": window_summary}
