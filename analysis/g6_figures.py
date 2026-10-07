from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G6 = ROOT / "artifacts" / "g6"
OUT = G6 / "figures"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442", "#8C564B", "#9467BD", "#2CA02C", "#7F7F7F", "#17BECF"]


def station_equal_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    x = ledger.loc[ledger["track"].eq("U")].copy()
    series = x.groupby(["model_id", "split", "horizon", "station", "series_id"], as_index=False)["mase"].mean()
    station = series.groupby(["model_id", "split", "horizon", "station"], as_index=False)["mase"].mean()
    return station.groupby(["model_id", "split", "horizon"], as_index=False).agg(
        station_equal_mase=("mase", "mean"), stations=("station", "nunique")
    )


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Noto Sans",sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.ci{stroke-width:2}.pt{stroke:white;stroke-width:1}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{escape(title)}</text>',
    ]


def skill_horizon_svg(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1180, 650
    left, top, panel_w, panel_h = 75, 65, 455, 500
    horizons = [1, 7, 30, 90, 180]
    ymin, ymax = 0.7, 40.0
    ymap = lambda value: top + panel_h - (np.log10(float(value)) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin)) * panel_h
    models = sorted(summary["model_id"].unique())
    colors = dict(zip(models, COLORS, strict=True))
    s = svg_start(width, height, "U-track station-equal MASE across forecast horizons")
    for panel, split in enumerate(["test_a", "test_b"]):
        x0 = left + panel * 535
        for tick in [1, 2, 5, 10, 20, 40]:
            y = ymap(tick)
            s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}"/>')
            s.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{tick:.0f}</text>')
        s += [f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{top+panel_h}"/>', f'<line class="axis" x1="{x0}" y1="{top+panel_h}" x2="{x0+panel_w}" y2="{top+panel_h}"/>']
        xs = {h: x0 + i * panel_w / 4 for i, h in enumerate(horizons)}
        for h in horizons:
            s.append(f'<text x="{xs[h]:.1f}" y="{top+panel_h+22}" text-anchor="middle" font-size="11">{h}</text>')
        s.append(f'<text x="{x0+panel_w/2}" y="{top-12}" text-anchor="middle" font-size="15" font-weight="bold">{split.replace("_", " ").title()}</text>')
        for model in models:
            g = summary.loc[summary.model_id.eq(model) & summary.split.eq(split)].set_index("horizon")
            points = [(xs[h], ymap(g.loc[h, "station_equal_mase"])) for h in horizons if h in g.index]
            if len(points) > 1:
                s.append(f'<polyline fill="none" stroke="{colors[model]}" stroke-width="1.8" points="' + " ".join(f"{x:.1f},{y:.1f}" for x,y in points) + '"/>')
            for x, y in points:
                s.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colors[model]}"/>')
    s.append('<text x="535" y="625" text-anchor="middle" font-size="13">Forecast horizon (days)</text>')
    s.append('<text x="18" y="320" text-anchor="middle" font-size="13" transform="rotate(-90 18 320)">Station-equal MASE (log scale)</text>')
    lx, ly = 1080, 75
    for i, model in enumerate(models):
        y = ly + i * 35
        s.append(f'<line x1="{lx-30}" y1="{y}" x2="{lx-8}" y2="{y}" stroke="{colors[model]}" stroke-width="3"/>')
        s.append(f'<text x="{lx}" y="{y+4}" font-size="11">{escape(model)}</text>')
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def forest_svg(frame: pd.DataFrame, path: Path, title: str, label_columns: list[str]) -> None:
    frame = frame.copy().reset_index(drop=True)
    width, row_h = 1200, 28
    height, left, right, top = 80 + row_h * len(frame), 410, 40, 50
    low = float(frame["ci_lower"].min()); high = float(frame["ci_upper"].max())
    pad = max((high-low)*0.06, 0.2); low -= pad; high += pad
    xmap = lambda v: left + (float(v)-low)/(high-low)*(width-left-right)
    s = svg_start(width, height, title)
    zero = xmap(0)
    s.append(f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{height-35}" stroke="#777" stroke-dasharray="4 4"/>')
    for i, row in frame.iterrows():
        y = top + 18 + i * row_h
        label = " | ".join(str(row[c]) for c in label_columns)
        sig = "holm_p_value" in frame.columns and float(row["holm_p_value"]) < 0.05
        color = "#0072B2" if float(row["effect_delta_mase"]) < 0 else "#D55E00"
        s.append(f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-size="11" font-weight="{"bold" if sig else "normal"}">{escape(label)}</text>')
        s.append(f'<line class="ci" x1="{xmap(row.ci_lower):.1f}" y1="{y}" x2="{xmap(row.ci_upper):.1f}" y2="{y}" stroke="{color}"/>')
        s.append(f'<circle class="pt" cx="{xmap(row.effect_delta_mase):.1f}" cy="{y}" r="{5 if sig else 4}" fill="{color}"/>')
    for v in np.linspace(low, high, 6):
        x = xmap(v); s.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle" font-size="10">{v:.2f}</text>')
    s.append(f'<text x="{(left+width-right)/2}" y="{height-28}" text-anchor="middle" font-size="12">ΔMASE (negative favors candidate / added information)</text>')
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(G6 / "window_scores.parquet")
    summary = station_equal_summary(ledger)
    summary.to_csv(OUT / "skill_horizon_u.csv", index=False)
    skill_horizon_svg(summary, OUT / "skill_horizon_u.svg")
    h1 = pd.read_csv(G6 / "h1_primary_comparisons.csv")
    h3 = pd.read_csv(G6 / "h3_information_gain.csv")
    forest_svg(h1, OUT / "h1_forest.svg", "H1 paired effects with 95% block-bootstrap intervals", ["split", "candidate", "reference"])
    forest_svg(h3, OUT / "h3_forest.svg", "H3 information-track effects with 95% block-bootstrap intervals", ["split", "candidate", "candidate_track"])
    files = sorted(p for p in OUT.iterdir() if p.name != "manifest.json")
    manifest = {"source_manifest_sha256": hashlib.sha256((G6 / "manifest.json").read_bytes()).hexdigest(), "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
