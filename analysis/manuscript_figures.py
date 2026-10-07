from __future__ import annotations

import hashlib
import json
import math
import os
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GEO_BENCH_RESULTS_DIR", str(ROOT / "manuscript"))) / "figures"

INK = "#17202A"
MUTED = "#566573"
BLUE = "#2471A3"
TEAL = "#148F77"
ORANGE = "#D68910"
RED = "#B03A2E"
LIGHT_BLUE = "#EAF2F8"
LIGHT_TEAL = "#E8F8F5"
LIGHT_ORANGE = "#FEF5E7"
LIGHT_GRAY = "#F4F6F7"

MODEL_LABELS = {
    "persistence": "Persistence", "drift": "Linear drift", "seasonal_365": "Seasonal-365",
    "ets": "ETS", "arima": "ARIMA", "fourier_ridge": "Fourier ridge",
    "lightgbm": "LightGBM", "dlinear": "DLinear", "patchtst": "PatchTST",
    "timesfm_3": "TimesFM 3", "chronos_2": "Chronos-2", "moirai_2_small": "Moirai-2", "toto_2_313m": "Toto-2",
}
MODEL_COLORS = {
    "persistence": "#7F8C8D", "drift": "#34495E", "seasonal_365": "#AAB7B8",
    "ets": "#85C1E9", "arima": "#2E86C1", "fourier_ridge": "#154360",
    "lightgbm": "#58D68D", "dlinear": "#239B56", "patchtst": "#145A32",
    "timesfm_3": "#E67E22", "chronos_2": "#C0392B", "moirai_2_small": "#8E44AD", "toto_2_313m": "#D4AC0D",
}
MODEL_FAMILY = {
    **{k: "Simple" for k in ("persistence", "drift", "seasonal_365")},
    **{k: "Statistical" for k in ("ets", "arima", "fourier_ridge")},
    **{k: "Target-domain" for k in ("lightgbm", "dlinear", "patchtst")},
    **{k: "Foundation" for k in ("timesfm_3", "chronos_2", "moirai_2_small", "toto_2_313m")},
}


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = "none", radius: int = 10) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def text(x: int, y: int, value: str, size: int = 20, *, anchor: str = "start", weight: int = 400, fill: str = INK) -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'


def line(x1: int, y1: int, x2: int, y2: int, color: str = MUTED, width: int = 2, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dashed}/>'


def svg_open(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    result = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,"Noto Sans",sans-serif}</style>', text(60, 45, title, 25, weight=700)]
    if subtitle:
        result.append(text(60, 72, subtitle, 15, fill=MUTED))
    return result


def circle(cx: float, cy: float, radius: float, color: str, *, filled: bool = True, width: int = 2) -> str:
    fill = color if filled else "white"
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="{width}"/>'


def square(cx: float, cy: float, radius: float, color: str, *, filled: bool = True, width: int = 2) -> str:
    fill = color if filled else "white"
    return f'<rect x="{cx-radius:.1f}" y="{cy-radius:.1f}" width="{2*radius:.1f}" height="{2*radius:.1f}" fill="{fill}" stroke="{color}" stroke-width="{width}"/>'


def clipped_interval(s: list[str], xmap, low: float, effect: float, high: float, y: float, xmin: float, xmax: float, color: str, significant: bool, shape: str = "circle") -> None:
    lo, hi = max(low, xmin), min(high, xmax)
    s.append(line(round(xmap(lo)), round(y), round(xmap(hi)), round(y), color, 2))
    if low < xmin:
        s.append(f'<path d="M {xmap(xmin)+7:.1f} {y-5:.1f} L {xmap(xmin):.1f} {y:.1f} L {xmap(xmin)+7:.1f} {y+5:.1f}" fill="none" stroke="{color}" stroke-width="2"/>')
    if high > xmax:
        s.append(f'<path d="M {xmap(xmax)-7:.1f} {y-5:.1f} L {xmap(xmax):.1f} {y:.1f} L {xmap(xmax)-7:.1f} {y+5:.1f}" fill="none" stroke="{color}" stroke-width="2"/>')
    point = min(max(effect, xmin), xmax)
    s.append(square(xmap(point), y, 4.5, color, filled=significant) if shape == "square" else circle(xmap(point), y, 4.5, color, filled=significant))


def protocol_figure(path: Path) -> None:
    width, height = 1600, 920
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Noto Sans",sans-serif}.small{font-size:16px;fill:#566573}.label{font-size:18px;font-weight:700;fill:#17202A}</style>',
        text(80, 58, "Apparent-resistivity forecasting benchmark protocol", 30, weight=700),
        text(80, 88, "Fixed temporal splits, information budgets, and station-level evaluation", 18, fill=MUTED),
    ]

    # Panel A: temporal protocol.
    s += [text(80, 145, "A  Temporal design", 23, weight=700), text(80, 174, "Configuration choices stop before either test period.", 16, fill=MUTED)]
    x0, y0, total_w, bar_h = 80, 208, 1440, 62
    segments = [
        ("Train", "2016–2020", 5.0, BLUE, "Initial fitting"),
        ("Validation", "2021–2023", 3.0, TEAL, "All selection"),
        ("Test A", "2024–2025", 2.0, ORANGE, "Independent report"),
        ("Test B", "2026–31 Aug", 8 / 12, RED, "Temporal replication"),
    ]
    scale = total_w / sum(item[2] for item in segments)
    cursor = x0
    centers: list[tuple[float, str, str, str]] = []
    for name, years, duration, color, note in segments:
        w = duration * scale
        s.append(rect(round(cursor), y0, round(w), bar_h, color, radius=0))
        centers.append((cursor + w / 2, name, years, note))
        cursor += w
    for center, name, years, note in centers:
        s += [text(round(center), y0 + 27, name, 18, anchor="middle", weight=700, fill="white"), text(round(center), y0 + 50, years, 15, anchor="middle", fill="white"), text(round(center), y0 + 88, note, 15, anchor="middle", fill=MUTED)]
    s += [text(80, 330, "Forecast origins", 17, weight=700), text(250, 330, "1/7 d: Mondays   ·   30/90 d: month starts   ·   180 d: 1 Jan and 1 Jul", 17, fill=MUTED)]
    s += [text(80, 362, "Primary endpoint", 17, weight=700), text(250, 362, "90 days; all models receive at most 730 days of causal history", 17, fill=MUTED)]

    # Panel B: information tracks.
    s += [text(80, 430, "B  Controlled information tracks", 23, weight=700), text(80, 459, "A model receives only the information native to its registered track.", 16, fill=MUTED)]
    cards = [
        (80, "U", "Target history", ["One apparent-resistivity channel", "Calendar features only"], LIGHT_BLUE, BLUE),
        (455, "M", "Same-station resistivity", ["U plus eligible resistivity channels", "No future observations"], LIGHT_TEAL, TEAL),
        (830, "X", "Historical environment", ["M plus water level, temperature, rainfall", "No future observed weather"], LIGHT_ORANGE, ORANGE),
        (1205, "O", "Oracle (not evaluated)", ["X plus future measured environment", "Defined; no results in this study"], LIGHT_GRAY, MUTED),
    ]
    for x, code, heading, body, fill, color in cards:
        s.append(rect(x, 490, 315, 150, fill, color, 12))
        s.append(rect(x + 18, 508, 48, 48, color, radius=24))
        s += [text(x + 42, 540, code, 22, anchor="middle", weight=700, fill="white"), text(x + 82, 535, heading, 18, weight=700)]
        for i, item in enumerate(body):
            s += [text(x + 24, 584 + i * 26, "•", 18, weight=700, fill=color), text(x + 45, 584 + i * 26, item, 14, fill=MUTED)]
    s += [line(395, 565, 438, 565, BLUE, 3), text(416, 555, "+", 20, anchor="middle", weight=700, fill=BLUE), line(770, 565, 813, 565, TEAL, 3), text(791, 555, "+", 20, anchor="middle", weight=700, fill=TEAL)]

    # Panel C: comparison and aggregation.
    s += [text(80, 705, "C  Fair comparison and inference", 23, weight=700)]
    steps = [
        (80, "1", "Predict", "Frozen models and fixed configurations"),
        (390, "2", "Pair", "Common successful windows only"),
        (700, "3", "Aggregate", "day → window → channel → station"),
        (1010, "4", "Infer", "station-equal effects + block bootstrap"),
        (1320, "5", "Report", "A and B separately; failures retained"),
    ]
    for index, (x, number, heading, body) in enumerate(steps):
        if index:
            s.append(f'<path d="M {x-48} 790 L {x-20} 790" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"/>')
        s.append(rect(x, 752, 220, 112, LIGHT_GRAY, "#D5D8DC", 10))
        s.append(rect(x + 16, 769, 34, 34, INK, radius=17))
        s += [text(x + 33, 793, number, 17, anchor="middle", weight=700, fill="white"), text(x + 62, 791, heading, 17, weight=700)]
        words = body.split(" ")
        midpoint = max(1, len(words) // 2)
        first = " ".join(words[:midpoint])
        second = " ".join(words[midpoint:])
        s += [text(x + 18, 827, first, 14, fill=MUTED), text(x + 18, 849, second, 14, fill=MUTED)]
    s.insert(3, '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#566573"/></marker></defs>')
    s += [text(1520, 898, "Task: forecast observations, not earthquakes", 15, anchor="end", weight=700, fill=RED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def skill_figure(frame: pd.DataFrame, path: Path) -> None:
    w, h = 1500, 790
    s = svg_open(w, h, "Forecast skill across horizons", "U track · station-equal MASE · lower is better · logarithmic y-axis")
    left, top, panel_w, panel_h, gap = 85, 115, 555, 455, 80
    horizons, ymin, ymax = [1, 7, 30, 90, 180], 0.7, 45
    markers = {m: i % 3 for i, m in enumerate(MODEL_LABELS)}
    for panel, split in enumerate(("test_a", "test_b")):
        x0 = left + panel * (panel_w + gap)
        xp = lambda horizon: x0 + horizons.index(int(horizon)) * panel_w / 4
        yp = lambda value: top + panel_h - (math.log10(float(value)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * panel_h
        s.append(text(x0 + panel_w / 2, 101, "Test A · 2024–2025" if split == "test_a" else "Test B · 2026 to 31 Aug", 18, anchor="middle", weight=700))
        for tick in (1, 2, 5, 10, 20, 40):
            y = yp(tick); s += [line(x0, round(y), x0 + panel_w, round(y), "#E5E7E9", 1), text(x0 - 10, round(y + 5), str(tick), 13, anchor="end", fill=MUTED)]
        s += [line(x0, top, x0, top + panel_h, INK, 1), line(x0, top + panel_h, x0 + panel_w, top + panel_h, INK, 1)]
        for horizon in horizons:
            x = xp(horizon); s.append(text(round(x), top + panel_h + 26, str(horizon), 13, anchor="middle", fill=MUTED))
        for model, group in frame.loc[frame.split.eq(split)].groupby("model_id"):
            group = group.set_index("horizon")
            pts = [(xp(hz), yp(group.loc[hz, "station_equal_mase"])) for hz in horizons if hz in group.index]
            color = MODEL_COLORS[model]
            s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in pts:
                s.append(circle(x, y, 3.7, color))
    s += [text(680, 624, "Forecast horizon (days)", 15, anchor="middle", weight=700), '<text x="24" y="345" text-anchor="middle" font-size="15" font-weight="700" fill="#17202A" transform="rotate(-90 24 345)">Station-equal MASE</text>']
    # Compact family-grouped legend.
    legend_y = 660
    for family_index, family in enumerate(("Simple", "Statistical", "Target-domain", "Foundation")):
        x0 = 70 + family_index * 365
        s.append(text(x0, legend_y, family, 15, weight=700))
        models = [m for m in MODEL_LABELS if MODEL_FAMILY[m] == family]
        for i, model in enumerate(models):
            y = legend_y + 28 + i * 24
            s += [line(x0, y - 4, x0 + 25, y - 4, MODEL_COLORS[model], 3), circle(x0 + 12, y - 4, 3.5, MODEL_COLORS[model]), text(x0 + 34, y, MODEL_LABELS[model], 13, fill=MUTED)]
    s.append("</svg>"); path.write_text("\n".join(s) + "\n", encoding="utf-8")


def h1_figure(frame: pd.DataFrame, path: Path) -> None:
    w, h = 1380, 720
    s = svg_open(w, h, "Foundation-model effects at the primary endpoint", "H1 · U track · 90 days · ΔMASE = candidate − reference · filled marker: Holm p < 0.05")
    models = ["timesfm_3", "chronos_2", "moirai_2_small", "toto_2_313m"]
    xmin, xmax = -6, 6
    for row_i, split in enumerate(("test_a", "test_b")):
        for col_i, reference in enumerate(("seasonal_365", "fourier_ridge")):
            x0, y0, pw, ph = 230 + col_i * 565, 125 + row_i * 270, 480, 185
            xp = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * pw
            s.append(text(x0 + pw / 2, y0 - 20, ("Test A" if split == "test_a" else "Test B") + " · vs " + MODEL_LABELS[reference], 17, anchor="middle", weight=700))
            for tick in (-6, -3, 0, 3, 6):
                x = xp(tick); s += [line(round(x), y0, round(x), y0 + ph, "#E5E7E9" if tick else MUTED, 1, "4 4" if tick == 0 else None), text(round(x), y0 + ph + 22, str(tick), 12, anchor="middle", fill=MUTED)]
            group = frame.loc[frame.split.eq(split) & frame.reference.eq(reference)].set_index("candidate")
            for i, model in enumerate(models):
                y = y0 + 25 + i * 40; row = group.loc[model]
                if col_i == 0: s.append(text(x0 - 14, round(y + 5), MODEL_LABELS[model], 13, anchor="end", fill=MUTED))
                clipped_interval(s, xp, row.ci_lower, row.effect_delta_mase, row.ci_upper, y, xmin, xmax, MODEL_COLORS[model], row.holm_p_value < .05)
    s += [text(760, 690, "ΔMASE (negative favors the foundation model)", 15, anchor="middle", weight=700), text(1280, 690, "Arrows mark intervals beyond ±6", 13, anchor="end", fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def h3_figure(frame: pd.DataFrame, path: Path) -> None:
    w, h = 1450, 660
    s = svg_open(w, h, "Does additional station information improve forecasts?", "H3 · 90 days · paired with U · filled marker: Holm p < 0.05 · A circle / B square")
    models = ["chronos_2", "dlinear", "lightgbm", "patchtst", "timesfm_3", "toto_2_313m"]
    xmin, xmax = -2, 8
    for col_i, track in enumerate(("M", "X")):
        x0, y0, pw, ph = 250 + col_i * 600, 125, 500, 400
        xp = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * pw
        s.append(text(x0 + pw / 2, 105, ("M − U: same-station resistivity" if track == "M" else "X − U: historical environment"), 18, anchor="middle", weight=700))
        for tick in (-2, 0, 2, 4, 6, 8):
            x = xp(tick); s += [line(round(x), y0, round(x), y0 + ph, "#E5E7E9" if tick else MUTED, 1, "4 4" if tick == 0 else None), text(round(x), y0 + ph + 25, str(tick), 12, anchor="middle", fill=MUTED)]
        for i, model in enumerate(models):
            y = y0 + 36 + i * 62
            if col_i == 0: s.append(text(x0 - 15, round(y + 5), MODEL_LABELS[model], 14, anchor="end", fill=MUTED))
            for split, offset, shape in (("test_a", -8, "circle"), ("test_b", 8, "square")):
                rows = frame.loc[frame.candidate.eq(model) & frame.candidate_track.eq(track) & frame.split.eq(split)]
                if rows.empty: continue
                row = rows.iloc[0]
                clipped_interval(s, xp, row.ci_lower, row.effect_delta_mase, row.ci_upper, y + offset, xmin, xmax, MODEL_COLORS[model], row.holm_p_value < .05, shape)
    s += [text(780, 590, "ΔMASE (negative favors added information)", 15, anchor="middle", weight=700), text(1370, 620, "Arrows mark intervals beyond display range", 13, anchor="end", fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def probability_figure(frame: pd.DataFrame, path: Path) -> None:
    x = frame.loc[frame.horizon.eq(90) & frame.track.eq("U")].copy()
    w, h = 1160, 650
    s = svg_open(w, h, "Probabilistic calibration and sharpness", "U track · 90 days · lower width is sharper · dashed line is nominal 80% coverage")
    x0, y0, pw, ph = 95, 115, 760, 430
    xmin, xmax, ymin, ymax = 0.15, max(.6, float(x.width_80.max()) * 1.12), .68, .88
    xp = lambda value: x0 + (float(value) - xmin) / (xmax - xmin) * pw
    yp = lambda value: y0 + ph - (float(value) - ymin) / (ymax - ymin) * ph
    for tick in (.70, .75, .80, .85):
        y = yp(tick); s += [line(x0, round(y), x0 + pw, round(y), MUTED if tick == .8 else "#E5E7E9", 1, "5 4" if tick == .8 else None), text(x0 - 10, round(y + 5), f"{tick:.2f}", 13, anchor="end", fill=MUTED)]
    for tick in (.2, .3, .4, .5, .6):
        xx = xp(tick); s += [line(round(xx), y0, round(xx), y0 + ph, "#F0F1F2", 1), text(round(xx), y0 + ph + 25, f"{tick:.1f}", 13, anchor="middle", fill=MUTED)]
    s += [line(x0, y0, x0, y0 + ph, INK, 1), line(x0, y0 + ph, x0 + pw, y0 + ph, INK, 1)]
    label_offsets = {("chronos_2", "test_a"): (-92, 28), ("chronos_2", "test_b"): (-82, -10), ("timesfm_3", "test_a"): (12, -18), ("timesfm_3", "test_b"): (12, -2), ("moirai_2_small", "test_a"): (12, 18), ("moirai_2_small", "test_b"): (12, 18), ("toto_2_313m", "test_a"): (12, -10), ("toto_2_313m", "test_b"): (12, 18)}
    for row in x.itertuples(index=False):
        px, py, color = xp(row.width_80), yp(row.coverage_80), MODEL_COLORS[row.model_id]
        shape = circle(px, py, 6, color) if row.split == "test_a" else square(px, py, 5, color)
        dx, dy = label_offsets[(row.model_id, row.split)]
        s += [shape, text(round(px + dx), round(py + dy), MODEL_LABELS[row.model_id] + (" A" if row.split == "test_a" else " B"), 12, fill=color)]
    s += [text(x0 + pw / 2, 596, "Mean 80% interval width", 15, anchor="middle", weight=700), '<text x="25" y="335" text-anchor="middle" font-size="15" font-weight="700" fill="#17202A" transform="rotate(-90 25 335)">Empirical coverage</text>', text(920, 180, "Model", 15, weight=700)]
    for i, model in enumerate(("chronos_2", "timesfm_3", "moirai_2_small", "toto_2_313m")):
        yy = 215 + i * 42; s += [circle(920, yy - 5, 5, MODEL_COLORS[model]), text(938, yy, MODEL_LABELS[model], 13, fill=MUTED)]
    s += [circle(920, 410, 5, INK), text(938, 415, "Test A", 13, fill=MUTED), square(920, 447, 5, INK), text(938, 452, "Test B", 13, fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def cost_figure(costs: pd.DataFrame, skill: pd.DataFrame, path: Path) -> None:
    x = costs.loc[costs.horizon.eq(90) & costs.track.eq("U")].merge(skill.loc[skill.horizon.eq(90)], on=["model_id", "split", "horizon"])
    x = x.loc[x.median_inference_seconds.gt(0)].copy()
    w, h = 1340, 710
    s = svg_open(w, h, "Accuracy versus recorded inference cost", "U track · 90 days · both axes logarithmic · task time excludes model loading and environment startup")
    x0, y0, pw, ph = 100, 115, 820, 470
    xmin, xmax, ymin, ymax = 0.0005, 0.5, .7, 45
    xp = lambda value: x0 + (math.log10(float(value)) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * pw
    yp = lambda value: y0 + ph - (math.log10(float(value)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * ph
    for tick, label in ((.001, ".001"), (.01, ".01"), (.1, ".1")):
        xx = xp(tick); s += [line(round(xx), y0, round(xx), y0 + ph, "#E5E7E9", 1), text(round(xx), y0 + ph + 25, label, 13, anchor="middle", fill=MUTED)]
    for tick in (1, 2, 5, 10, 20, 40):
        yy = yp(tick); s += [line(x0, round(yy), x0 + pw, round(yy), "#E5E7E9", 1), text(x0 - 10, round(yy + 5), str(tick), 13, anchor="end", fill=MUTED)]
    s += [line(x0, y0, x0, y0 + ph, INK, 1), line(x0, y0 + ph, x0 + pw, y0 + ph, INK, 1)]
    for model, group in x.groupby("model_id"):
        group = group.set_index("split")
        if {"test_a", "test_b"}.issubset(group.index):
            a, b = group.loc["test_a"], group.loc["test_b"]
            s.append(line(round(xp(a.median_inference_seconds)), round(yp(a.station_equal_mase)), round(xp(b.median_inference_seconds)), round(yp(b.station_equal_mase)), MODEL_COLORS[model], 1))
        for split, row in group.iterrows():
            s.append(circle(xp(row.median_inference_seconds), yp(row.station_equal_mase), 5.5, MODEL_COLORS[model]) if split == "test_a" else square(xp(row.median_inference_seconds), yp(row.station_equal_mase), 4.8, MODEL_COLORS[model]))
    s += [text(x0 + pw / 2, 635, "Median recorded seconds per window", 15, anchor="middle", weight=700), '<text x="25" y="350" text-anchor="middle" font-size="15" font-weight="700" fill="#17202A" transform="rotate(-90 25 350)">Station-equal MASE</text>']
    for family_i, family in enumerate(("Simple", "Statistical", "Target-domain", "Foundation")):
        yy = 120 + family_i * 132; s.append(text(980, yy, family, 15, weight=700))
        for i, model in enumerate([m for m in MODEL_LABELS if MODEL_FAMILY[m] == family]):
            y = yy + 27 + i * 25; s += [circle(986, y - 4, 4, MODEL_COLORS[model]), text(1002, y, MODEL_LABELS[model], 13, fill=MUTED)]
    s += [circle(1195, 620, 5, INK), text(1210, 625, "Test A", 12, fill=MUTED), square(1195, 649, 4.5, INK), text(1210, 654, "Test B", 12, fill=MUTED), "</svg>"]
    path.write_text("\n".join(s) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "figure_1_protocol.svg"
    protocol_figure(protocol)
    g6 = ROOT / "artifacts" / "g6"
    sources = {
        "skill": g6 / "figures" / "skill_horizon_u.csv",
        "h1": g6 / "h1_primary_comparisons.csv",
        "h3": g6 / "h3_information_gain.csv",
        "probability": g6 / "extended" / "probability_summary.csv",
        "cost": g6 / "extended" / "cost_summary.csv",
    }
    skill = pd.read_csv(sources["skill"])
    unknown_models = set(skill["model_id"]) - set(MODEL_LABELS)
    if unknown_models:
        raise ValueError(f"Figure model metadata missing for: {sorted(unknown_models)}")
    skill_figure(skill, OUT / "figure_2_skill_horizon.svg")
    h1_figure(pd.read_csv(sources["h1"]), OUT / "figure_3_h1_effects.svg")
    h3_figure(pd.read_csv(sources["h3"]), OUT / "figure_4_information_gain.svg")
    probability_figure(pd.read_csv(sources["probability"]), OUT / "figure_5_probability.svg")
    cost_figure(pd.read_csv(sources["cost"]), skill, OUT / "figure_6_accuracy_cost.svg")
    figures = sorted(OUT.glob("figure_*.svg"))
    manifest = {
        "files": {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in figures},
        "sources": {str(item.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(item.read_bytes()).hexdigest() for item in sources.values()},
        "notes": [
            "Figure 1 is a protocol schematic and contains no manually transcribed model scores.",
            "Dates and track definitions follow the frozen research protocol.",
            "Figures 2–6 are generated directly from hashed G6 CSV outputs.",
            "Figure 3 displays confidence intervals within ±6 ΔMASE and Figure 4 within −2 to 8; arrows indicate intervals extending beyond those ranges, while source tables retain full values.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
