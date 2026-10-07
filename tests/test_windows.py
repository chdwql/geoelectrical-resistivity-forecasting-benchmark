from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from geoelectrical_resistivity_forecasting_benchmark.windows import generate_windows, origins_for_split


def daily_target() -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", "2026-08-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "station": "11003",
            "point": "D",
            "item": "3211",
            "factor": "resistivity",
            "value": np.arange(len(dates), dtype=float),
        }
    )


def test_origin_cadence_and_complete_target_boundary() -> None:
    weekly = origins_for_split(date(2026, 1, 1), date(2026, 1, 31), 7)
    assert all(origin.weekday() == 0 for origin in weekly)
    assert weekly[-1] + pd.Timedelta(days=6) <= pd.Timestamp("2026-01-31")
    monthly = origins_for_split(date(2024, 1, 1), date(2024, 12, 31), 90)
    assert all(origin.day == 1 for origin in monthly)
    assert monthly[-1] + pd.Timedelta(days=89) <= pd.Timestamp("2024-12-31")


def test_windows_never_cross_split_end() -> None:
    windows = generate_windows(
        daily_target(),
        splits={"test_b": (date(2026, 1, 1), date(2026, 8, 31))},
        horizons=[90, 180],
        tracks=["U"],
        context_days=730,
        minimum_target_coverage=0.9,
    )
    assert (pd.to_datetime(windows["target_end"]) <= pd.Timestamp("2026-08-31")).all()
    assert set(windows["track"]) == {"U"}
    assert windows["eligible"].all()


def test_tracks_have_independent_input_eligibility() -> None:
    target = daily_target()
    kwargs = dict(
        splits={"test_b": (date(2026, 1, 1), date(2026, 8, 31))},
        horizons=[90],
        tracks=["U", "M", "X", "O"],
        context_days=730,
        minimum_target_coverage=0.9,
    )
    windows = generate_windows(target, **kwargs)
    assert windows.loc[windows["track"].eq("U"), "eligible"].all()
    assert not windows.loc[windows["track"].ne("U"), "eligible"].any()
    assert windows.loc[windows["track"].eq("M"), "exclusion_reason"].str.contains(
        "insufficient_station_resistivity_panel"
    ).all()


def test_future_value_perturbation_does_not_change_manifest() -> None:
    base = daily_target()
    changed = base.copy()
    changed.loc[changed["date"] >= "2026-01-01", "value"] += 1_000_000
    kwargs = dict(
        splits={"test_b": (date(2026, 1, 1), date(2026, 8, 31))},
        horizons=[90],
        tracks=["U"],
        context_days=730,
        minimum_target_coverage=0.9,
    )
    left = generate_windows(base, **kwargs)
    right = generate_windows(changed, **kwargs)
    pd.testing.assert_frame_equal(left, right)


def test_insufficient_context_is_explicit() -> None:
    short = daily_target().loc[lambda frame: frame["date"] >= "2023-07-01"]
    windows = generate_windows(
        short,
        splits={"test_a": (date(2024, 1, 1), date(2024, 12, 31))},
        horizons=[90],
        tracks=["U"],
        context_days=730,
        minimum_target_coverage=0.9,
    )
    assert not windows.iloc[0]["eligible"]
    assert "insufficient_context_span" in windows.iloc[0]["exclusion_reason"]


def test_explicit_target_catalog_excludes_context_only_resistivity() -> None:
    primary = daily_target()
    context_only = primary.copy()
    context_only["item"] = "3212"
    combined = pd.concat([primary, context_only], ignore_index=True)
    windows = generate_windows(
        combined,
        splits={"test_b": (date(2026, 1, 1), date(2026, 8, 31))},
        horizons=[90],
        tracks=["U"],
        context_days=730,
        minimum_target_coverage=0.9,
        target_series_ids=["11003/D/3211"],
    )
    assert set(windows["series_id"]) == {"11003/D/3211"}
