from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd

try:
    from analysis.manuscript_figures import (
        MODEL_COLORS, MODEL_FAMILY, MODEL_LABELS, circle, clipped_interval, line, square, svg_open, text,
    )
except ModuleNotFoundError:  # Support direct execution from the project root.
    from manuscript_figures import (
        MODEL_COLORS, MODEL_FAMILY, MODEL_LABELS, circle, clipped_interval, line, square, svg_open, text,
    )


ROOT = Path(__file__).resolve().parents[1]
G6 = ROOT / "artifacts" / "g6"
OUT = Path(os.environ.get("GEO_BENCH_RESULTS_DIR", str(ROOT / "manuscript"))) / "supplementary_figures"
INK, MUTED = "#17202A", "#566573"


def rank_figure(ranks: pd.DataFrame, path: Path) -> None:
    x = ranks.loc[ranks.horizon.eq(90)].pivot(index="model_id", columns="split", values="rank").sort_values("test_a")
    w, h = 1120, 720
    s = svg_open(w, h, "Cross-period rank stability at 90 days", "U track · rank 1 is best · lines connect the same model")
    left, right, top, bottom = 310, 810, 115, 650
    yp = lambda rank: top + (float(rank) - 1) / (len(x) - 1) * (bottom - top)
    s += [text(left, 98, "Test A · 2024–2025", 17, anchor="middle", weight=700), text(right, 98, "Test B · 2026", 17, anchor="middle", weight=700)]
    for model, row in x.iterrows():
        ya, yb, color = yp(row.test_a), yp(row.test_b), MODEL_COLORS[model]
        s += [line(left, round(ya), right, round(yb), color, 2), circle(left, ya, 4.5, color), square(right, yb, 4, color), text(left - 14, round(ya + 5), f"{MODEL_LABELS[model]}  {row.test_a:.0f}", 13, anchor="end", fill=MUTED), text(right + 14, round(yb + 5), f"{row.test_b:.0f}  {MODEL_LABELS[model]}", 13, fill=MUTED)]
    s += [text(560, 694, "Spearman ρ = 0.357; mean absolute rank change = 2.77", 14, anchor="middle", weight=700), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def sensitivity_figure(frame: pd.DataFrame, variant: str, title: str, path: Path) -> None:
    x = frame.loc[frame.variant.eq(variant)].copy()
    models = [m for m in MODEL_LABELS if m in set(x.model_id)]
    w, h = 1300, 850
    s = svg_open(w, h, title, "U track · 90 days · paired ΔMASE versus Q0/730 · A circle / B square · descriptive intervals")
    x0, y0, pw, row_h, xmin, xmax = 260, 115, 940, 48, -6, 12
    xp = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * pw
    for tick in (-6, -3, 0, 3, 6, 9, 12):
        xx = xp(tick); s += [line(round(xx), y0, round(xx), y0 + row_h * len(models), "#E5E7E9" if tick else MUTED, 1, "4 4" if tick == 0 else None), text(round(xx), y0 + row_h * len(models) + 25, str(tick), 12, anchor="middle", fill=MUTED)]
    for i, model in enumerate(models):
        y = y0 + 25 + i * row_h
        s.append(text(x0 - 15, round(y + 5), MODEL_LABELS[model], 13, anchor="end", fill=MUTED))
        for split, offset, shape in (("test_a", -7, "circle"), ("test_b", 7, "square")):
            row = x.loc[x.model_id.eq(model) & x.split.eq(split)].iloc[0]
            excludes_zero = row.ci_lower > 0 or row.ci_upper < 0
            clipped_interval(s, xp, row.ci_lower, row.effect_delta_mase, row.ci_upper, y + offset, xmin, xmax, MODEL_COLORS[model], excludes_zero, shape)
    s += [text(x0 + pw / 2, 800, "ΔMASE (negative favors the sensitivity variant)", 15, anchor="middle", weight=700), text(1195, 825, "Arrows indicate intervals outside −6 to 12", 12, anchor="end", fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def heat_color(value: float, low: float = 0.0, high: float = math.log10(50)) -> tuple[str, str]:
    ratio = min(max((math.log10(max(value, 1.0)) - low) / (high - low), 0), 1)
    start, end = (247, 251, 255), (8, 48, 107)
    rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(start, end, strict=True))
    return "#" + "".join(f"{part:02x}" for part in rgb), "white" if ratio > .55 else INK


def station_heatmap(frame: pd.DataFrame, path: Path) -> None:
    models = [m for m in MODEL_LABELS if m in set(frame.model_id)]
    stations = sorted(frame.station.astype(str).unique())
    w, h = 1570, 840
    s = svg_open(w, h, "Station-level 90-day forecasting difficulty", "U track · cell text is MASE · color uses log scale capped at 50 for display")
    cell_w, cell_h, y0 = 58, 44, 135
    for panel, split in enumerate(("test_a", "test_b")):
        x0 = 250 + panel * 650
        s.append(text(x0 + cell_w * len(stations) / 2, 105, "Test A · 2024–2025" if split == "test_a" else "Test B · 2026", 18, anchor="middle", weight=700))
        for j, station in enumerate(stations):
            s.append(text(x0 + j * cell_w + cell_w / 2, 126, station, 11, anchor="middle", fill=MUTED))
        subset = frame.loc[frame.split.eq(split)].copy(); subset["station"] = subset.station.astype(str)
        values = subset.set_index(["model_id", "station"])["mase"]
        for i, model in enumerate(models):
            y = y0 + i * cell_h
            if panel == 0: s.append(text(x0 - 14, y + 27, MODEL_LABELS[model], 13, anchor="end", fill=MUTED))
            for j, station in enumerate(stations):
                value = float(values.loc[(model, station)])
                fill, ink = heat_color(value)
                x = x0 + j * cell_w
                s.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{fill}"/>')
                label = f"{value:.1f}" if value < 100 else f"{value:.0f}"
                s.append(text(x + (cell_w - 2) / 2, y + 27, label, 11, anchor="middle", fill=ink))
    s += [text(775, 760, "Station identifier", 14, anchor="middle", weight=700), text(1490, 780, "Darker = larger MASE; values >50 share the darkest color", 12, anchor="end", fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def event_figure(frame: pd.DataFrame, path: Path) -> None:
    x = frame.loc[frame.horizon.eq(90) & frame.track.eq("U")].copy()
    models = ["timesfm_3", "chronos_2", "moirai_2_small", "toto_2_313m"]
    strata = ["no_target_event", "spike_candidate", "step_candidate", "step_and_spike"]
    labels = ["None", "Spike", "Step", "Step + spike"]
    w, h = 1260, 690
    s = svg_open(w, h, "Forecast difficulty by retrospective quality-event stratum", "U track · 90 days · unreviewed candidates · descriptive only · logarithmic y-axis")
    left, top, pw, ph, gap = 85, 125, 500, 430, 70
    for panel, split in enumerate(("test_a", "test_b")):
        x0 = left + panel * (pw + gap)
        xp = lambda index: x0 + index * pw / 3
        yp = lambda value: top + ph - (math.log10(float(value)) - math.log10(2)) / (math.log10(45) - math.log10(2)) * ph
        s.append(text(x0 + pw / 2, 105, "Test A · 2024–2025" if split == "test_a" else "Test B · 2026", 17, anchor="middle", weight=700))
        for tick in (2, 5, 10, 20, 40):
            yy = yp(tick); s += [line(x0, round(yy), x0 + pw, round(yy), "#E5E7E9", 1), text(x0 - 9, round(yy + 5), str(tick), 12, anchor="end", fill=MUTED)]
        for i, label in enumerate(labels): s.append(text(round(xp(i)), top + ph + 25, label, 12, anchor="middle", fill=MUTED))
        for model in models:
            group = x.loc[x.split.eq(split) & x.model_id.eq(model)].set_index("event_stratum")
            pts = [(xp(i), yp(group.loc[stratum, "mase"])) for i, stratum in enumerate(strata)]
            s.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a,b in pts)}" fill="none" stroke="{MODEL_COLORS[model]}" stroke-width="2"/>')
            for xx, yy in pts: s.append(circle(xx, yy, 4.5, MODEL_COLORS[model]))
    s += ['<text x="24" y="350" text-anchor="middle" font-size="15" font-weight="700" fill="#17202A" transform="rotate(-90 24 350)">Station-equal MASE</text>', text(620, 625, "Target-window event stratum", 15, anchor="middle", weight=700)]
    for i, model in enumerate(models):
        yy = 652 + (i % 2) * 24; xx = 280 + (i // 2) * 340
        s += [line(xx, yy - 4, xx + 24, yy - 4, MODEL_COLORS[model], 3), circle(xx + 12, yy - 4, 4, MODEL_COLORS[model]), text(xx + 34, yy, MODEL_LABELS[model], 13, fill=MUTED)]
    s.append("</svg>"); path.write_text("\n".join(s) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "ranks": G6 / "remaining" / "cross_year_ranks.csv",
        "sensitivity": G6 / "remaining" / "sensitivity_effects.csv",
        "station": G6 / "remaining" / "station_u90.csv",
        "events": G6 / "extended" / "quality_event_strata.csv",
    }
    rank_figure(pd.read_csv(sources["ranks"]), OUT / "figure_s1_rank_stability.svg")
    sensitivity = pd.read_csv(sources["sensitivity"])
    sensitivity_figure(sensitivity, "Q1", "Q1 causal robust-input sensitivity", OUT / "figure_s2_q1_sensitivity.svg")
    sensitivity_figure(sensitivity, "context365", "365-day context sensitivity", OUT / "figure_s3_context_sensitivity.svg")
    station_heatmap(pd.read_csv(sources["station"]), OUT / "figure_s4_station_heatmap.svg")
    event_figure(pd.read_csv(sources["events"]), OUT / "figure_s5_quality_events.svg")
    files = sorted(OUT.glob("figure_*.svg"))
    manifest = {
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.values()},
        "notes": ["All supplementary figures are generated from hashed G6 outputs.", "Sensitivity and quality-event figures are descriptive, not additional confirmatory families."],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
