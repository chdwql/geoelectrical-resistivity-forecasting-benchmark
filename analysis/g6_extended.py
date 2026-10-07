from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis.g6_figures import svg_start
except ModuleNotFoundError:  # Support direct execution from the project root.
    from g6_figures import svg_start

ROOT = Path(__file__).resolve().parents[1]
G6 = ROOT / "artifacts" / "g6"
OUT = G6 / "extended"


def hierarchical_metrics(frame: pd.DataFrame, metrics: list[str], keys: list[str]) -> pd.DataFrame:
    series = frame.groupby(keys + ["station", "series_id"], as_index=False)[metrics].mean()
    station = series.groupby(keys + ["station"], as_index=False)[metrics].mean()
    result = station.groupby(keys, as_index=False).agg(**{m: (m, "mean") for m in metrics}, stations=("station", "nunique"))
    return result


def probability_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_quantile_loss", "wis", "coverage_80", "width_80", "quantile_crossing_rate"]
    x = ledger.loc[ledger["wis"].notna()].copy()
    result = hierarchical_metrics(x, metrics, ["model_id", "split", "horizon", "track"])
    result["coverage_80_error"] = (result["coverage_80"] - 0.8).abs()
    return result


def collect_costs(run_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in run_index.itertuples(index=False):
        directory = ROOT / record.run_directory
        tasks = pd.read_parquet(directory / "tasks.parquet")
        tasks = tasks.loc[tasks["model_id"].eq(record.model_id) & tasks["status"].eq("success")]
        runtime = pd.to_numeric(tasks["runtime_seconds"], errors="coerce").dropna()
        memory = pd.to_numeric(tasks.get("peak_memory_mb"), errors="coerce").dropna() if "peak_memory_mb" in tasks else pd.Series(dtype=float)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        fit = manifest.get("fit_runtime_seconds")
        if isinstance(fit, dict):
            fit = fit.get(record.model_id)
        if fit is None and isinstance(manifest.get("training"), list):
            training_seconds = [entry.get("seconds") for entry in manifest["training"]]
            if training_seconds and all(value is not None for value in training_seconds):
                fit = sum(float(value) for value in training_seconds)
        rows.append({
            "model_id": record.model_id, "split": record.split, "horizon": record.horizon, "track": record.track,
            "successful_tasks": len(tasks), "median_inference_seconds": float(runtime.median()) if len(runtime) else np.nan,
            "total_inference_seconds": float(runtime.sum()) if len(runtime) else np.nan,
            "peak_memory_mb": float(memory.max()) if len(memory) else np.nan,
            "fit_runtime_seconds": float(fit) if fit is not None else np.nan,
            "timing_scope": "inference_only_excludes_model_load",
        })
    return pd.DataFrame(rows)


def event_strata(ledger: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy(); events["event_date"] = pd.to_datetime(events["event_date"])
    event_map = {k: g for k, g in events.groupby("series_id")}
    labels = []
    for row in ledger.itertuples(index=False):
        group = event_map.get(row.series_id)
        if group is None:
            labels.append("no_target_event"); continue
        hit = group.loc[group["event_date"].between(pd.Timestamp(row.target_start), pd.Timestamp(row.target_end))]
        if hit.empty: labels.append("no_target_event")
        elif hit["event_type"].eq("step_candidate").any() and hit["event_type"].eq("spike_candidate").any(): labels.append("step_and_spike")
        elif hit["event_type"].eq("step_candidate").any(): labels.append("step_candidate")
        else: labels.append("spike_candidate")
    x = ledger.copy(); x["event_stratum"] = labels
    result = hierarchical_metrics(x, ["mase", "mae"], ["model_id", "split", "horizon", "track", "event_stratum"])
    counts = x.groupby(["model_id", "split", "horizon", "track", "event_stratum"], as_index=False).agg(windows=("window_id", "size"), series=("series_id", "nunique"))
    return result.merge(counts, on=["model_id", "split", "horizon", "track", "event_stratum"], validate="one_to_one")


def probability_svg(summary: pd.DataFrame, path: Path) -> None:
    x = summary.loc[summary.horizon.eq(90) & summary.track.eq("U")].copy()
    width, height, left, top, pw, ph = 1050, 580, 80, 65, 780, 430
    xmax = max(float(x.width_80.max()) * 1.08, 0.1)
    s = svg_start(width, height, "U-90 probabilistic calibration and sharpness")
    for cov in [0.4, 0.6, 0.8, 1.0]:
        y = top + ph - cov * ph; s.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{cov:.1f}</text>')
    ideal = top + ph - .8 * ph; s.append(f'<line x1="{left}" y1="{ideal:.1f}" x2="{left+pw}" y2="{ideal:.1f}" stroke="#111" stroke-dasharray="5 4"/>')
    models = sorted(x.model_id.unique()); colors = dict(zip(models, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B"], strict=False))
    for row in x.itertuples(index=False):
        px = left + float(row.width_80)/xmax*pw; py = top + ph - float(row.coverage_80)*ph
        shape = 6 if row.split == "test_a" else 4
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{shape}" fill="{colors[row.model_id]}" fill-opacity="0.8"><title>{row.model_id} {row.split}: coverage={row.coverage_80:.3f}, width={row.width_80:.3f}</title></circle>')
    s += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>', f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>', f'<text x="{left+pw/2}" y="550" text-anchor="middle" font-size="13">Mean 80% interval width (lower is sharper)</text>', '<text x="18" y="280" text-anchor="middle" font-size="13" transform="rotate(-90 18 280)">Empirical 80% coverage</text>']
    for i, model in enumerate(models):
        y=80+i*27; s.append(f'<circle cx="900" cy="{y}" r="5" fill="{colors[model]}"/><text x="913" y="{y+4}" font-size="11">{model}</text>')
    s.append('<text x="880" y="470" font-size="11">large circle: test A</text><text x="880" y="490" font-size="11">small circle: test B</text></svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def cost_svg(costs: pd.DataFrame, skill: pd.DataFrame, path: Path) -> None:
    x = costs.loc[costs.horizon.eq(90) & costs.track.eq("U")].merge(skill.loc[skill.horizon.eq(90)], on=["model_id","split","horizon"])
    x = x.loc[x.median_inference_seconds.gt(0)].copy()
    width,height,left,top,pw,ph=1050,580,80,65,780,430
    xmin,xmax=float(x.median_inference_seconds.min())*.7,float(x.median_inference_seconds.max())*1.4
    ymin,ymax=.7,40.0
    xp=lambda v:left+(np.log10(float(v))-np.log10(xmin))/(np.log10(xmax)-np.log10(xmin))*pw
    yp=lambda v:top+ph-(np.log10(float(v))-np.log10(ymin))/(np.log10(ymax)-np.log10(ymin))*ph
    s=svg_start(width,height,"U-90 accuracy versus recorded inference cost")
    models=sorted(x.model_id.unique()); colors=dict(zip(models,["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000","#F0E442","#8C564B","#9467BD","#2CA02C","#7F7F7F","#17BECF"],strict=True))
    for row in x.itertuples(index=False):
        s.append(f'<circle cx="{xp(row.median_inference_seconds):.1f}" cy="{yp(row.station_equal_mase):.1f}" r="{6 if row.split=="test_a" else 4}" fill="{colors[row.model_id]}" fill-opacity="0.8"><title>{row.model_id} {row.split}: {row.median_inference_seconds:.4g}s, MASE {row.station_equal_mase:.3f}</title></circle>')
    s += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>',f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>','<text x="470" y="550" text-anchor="middle" font-size="13">Median recorded inference seconds/window (log scale; excludes load)</text>','<text x="18" y="280" text-anchor="middle" font-size="13" transform="rotate(-90 18 280)">Station-equal MASE (log scale)</text>']
    for i,model in enumerate(models):
        y=65+i*28;s.append(f'<circle cx="900" cy="{y}" r="5" fill="{colors[model]}"/><text x="913" y="{y+4}" font-size="11">{model}</text>')
    s.append('</svg>');path.write_text("\n".join(s),encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger=pd.read_parquet(G6/"window_scores.parquet"); run_index=pd.read_csv(G6/"run_index.csv")
    probability=probability_summary(ledger); costs=collect_costs(run_index)
    events=pd.read_csv(ROOT/"artifacts/data_v1/quality_event_candidates.csv")
    strata=event_strata(ledger,events)
    skill=pd.read_csv(G6/"figures/skill_horizon_u.csv")
    probability.to_csv(OUT/"probability_summary.csv",index=False); costs.to_csv(OUT/"cost_summary.csv",index=False); strata.to_csv(OUT/"quality_event_strata.csv",index=False)
    probability_svg(probability,OUT/"probability_calibration_u90.svg"); cost_svg(costs,skill,OUT/"accuracy_cost_u90.svg")
    files=sorted(p for p in OUT.iterdir() if p.name!="manifest.json")
    manifest={"source_g6_manifest_sha256":hashlib.sha256((G6/"manifest.json").read_bytes()).hexdigest(),"quality_event_source_sha256":hashlib.sha256((ROOT/"artifacts/data_v1/quality_event_candidates.csv").read_bytes()).hexdigest(),"files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},"notes":["Inference timing excludes model loading; target-domain fit time is reported separately.","Quality events are unreviewed retrospective candidates and are used only for descriptive stratification."]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"probability_rows":len(probability),"cost_rows":len(costs),"event_rows":len(strata),"files":len(files)},indent=2))


if __name__=="__main__": main()
