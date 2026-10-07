from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis.g6_figures import forest_svg, svg_start
    from analysis.g6_report import hierarchical_effect, moving_block_bootstrap
except ModuleNotFoundError:
    from g6_figures import forest_svg, svg_start
    from g6_report import hierarchical_effect, moving_block_bootstrap

ROOT = Path(__file__).resolve().parents[1]
G6 = ROOT / "artifacts" / "g6"
OUT = G6 / "remaining"
PROTOCOL = "f7c65c07da0e8ec606b61771da2b675b1793a9ae99c1fab300b5e79f5c02f473"


def station_u90(ledger: pd.DataFrame) -> pd.DataFrame:
    x = ledger.loc[ledger.horizon.eq(90) & ledger.track.eq("U")]
    return x.groupby(["model_id", "split", "station", "series_id"], as_index=False).mase.mean().groupby(["model_id", "split", "station"], as_index=False).mase.mean()


def rank_stability(skill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = skill.copy(); ranks["rank"] = ranks.groupby(["split", "horizon"])["station_equal_mase"].rank(method="average")
    wide = ranks.pivot(index=["model_id", "horizon"], columns="split", values="rank").reset_index()
    stats=[]
    for h,g in wide.groupby("horizon"):
        stats.append({"horizon":h,"spearman_rank_correlation":g.test_a.corr(g.test_b,method="spearman"),"models":len(g),"mean_absolute_rank_change":(g.test_a-g.test_b).abs().mean()})
    return ranks, pd.DataFrame(stats)


def manifest_identity(manifest: dict) -> dict:
    return manifest.get("identity") or manifest.get("request_manifest", {}).get("identity", {})


def sensitivity_ledger() -> pd.DataFrame:
    windows=pd.read_parquet(ROOT/"artifacts/data_v1/windows.parquet")[["window_id","series_id","origin"]]
    paths=list((ROOT/"artifacts/formal/sensitivity").rglob("scores.parquet"))+list((ROOT/"artifacts/benchmark").glob("*/scores.parquet"))
    rows=[]; seen=set(); source_files=[]
    for p in paths:
        mp=p.parent/"manifest.json"
        if not mp.exists(): continue
        m=json.loads(mp.read_text(encoding="utf-8")); i=manifest_identity(m)
        if i.get("protocol_hash")!=PROTOCOL or i.get("split") not in ("test_a","test_b") or i.get("horizon")!=90 or (i.get("track") or "U")!="U": continue
        variant = "Q1" if i.get("input_variant")=="Q1" and i.get("context_days")==730 else "context365" if i.get("input_variant")=="Q0" and i.get("context_days")==365 else None
        if variant is None: continue
        source_files.extend([p, mp])
        scores=pd.read_parquet(p).merge(windows,on=["window_id","series_id"],how="left",validate="many_to_one")
        scores["origin"] = pd.to_datetime(scores["origin"])
        for model,g in scores.groupby("model_id"):
            key=(variant,str(model),i["split"])
            if key in seen: raise RuntimeError(f"duplicate sensitivity result: {key}")
            seen.add(key); g=g.copy(); g["variant"]=variant; rows.append(g)
    result=pd.concat(rows,ignore_index=True)
    expected={(v,m,s) for v in ("Q1","context365") for m in result.model_id.unique() for s in ("test_a","test_b")}
    observed=set(map(tuple,result[["variant","model_id","split"]].drop_duplicates().itertuples(index=False,name=None)))
    if expected!=observed or len(expected)!=52: raise RuntimeError(f"sensitivity matrix mismatch expected_count={len(expected)} observed_count={len(observed)} models={sorted(result.model_id.unique())} missing={sorted(expected-observed)} unexpected={sorted(observed-expected)}")
    result.attrs["source_files"] = source_files
    return result


def sensitivity_effects(standard: pd.DataFrame, sensitivity: pd.DataFrame, draws: int=5000) -> pd.DataFrame:
    rows=[]; offset=0; keys=["series_id","station","split","horizon","origin"]
    base=standard.loc[standard.horizon.eq(90)&standard.track.eq("U"),keys+["model_id","mase"]]
    for (variant,model,split),g in sensitivity.groupby(["variant","model_id","split"],sort=True):
        left=g[keys+["mase"]].rename(columns={"mase":"sensitivity_mase"})
        right=base.loc[base.model_id.eq(model)&base.split.eq(split),keys+["mase"]].rename(columns={"mase":"standard_mase"})
        pairs=left.merge(right,on=keys,how="inner",validate="one_to_one");pairs["delta_mase"]=pairs.sensitivity_mase-pairs.standard_mase
        boot=moving_block_bootstrap(pairs,draws=draws,block_origins=3,seed=20260916+offset);offset+=1
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append({"variant":variant,"model_id":model,"split":split,"paired_windows":len(pairs),"paired_stations":pairs.station.nunique(),"effect_delta_mase":hierarchical_effect(pairs),"ci_lower":lo,"ci_upper":hi,"bootstrap_draws":draws})
    return pd.DataFrame(rows)


def rank_svg(ranks: pd.DataFrame,path:Path) -> None:
    x=ranks.loc[ranks.horizon.eq(90)].pivot(index="model_id",columns="split",values="rank").sort_values("test_a")
    w,h=850,520;s=svg_start(w,h,"U-90 rank changes between test periods")
    top,bottom=65,470; y=lambda r:top+(float(r)-1)/12*(bottom-top)
    s += ['<text x="180" y="50" text-anchor="middle">Test A</text>','<text x="670" y="50" text-anchor="middle">Test B</text>']
    for model,row in x.iterrows():
        ya,yb=y(row.test_a),y(row.test_b);s.append(f'<line x1="220" y1="{ya:.1f}" x2="630" y2="{yb:.1f}" stroke="#888"/><circle cx="220" cy="{ya:.1f}" r="4" fill="#0072B2"/><circle cx="630" cy="{yb:.1f}" r="4" fill="#D55E00"/><text x="210" y="{ya+4:.1f}" text-anchor="end" font-size="11">{model} ({row.test_a:.0f})</text><text x="640" y="{yb+4:.1f}" font-size="11">{model} ({row.test_b:.0f})</text>')
    s.append('</svg>');path.write_text("\n".join(s),encoding="utf-8")


def comparison_coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    frames=[]
    for family,filename in [("H1","h1_primary_comparisons.csv"),("H3","h3_information_gain.csv")]:
        table=pd.read_csv(G6/filename)
        for row in table.itertuples(index=False):
            candidate=len(ledger.loc[ledger.model_id.eq(row.candidate)&ledger.split.eq(row.split)&ledger.horizon.eq(row.horizon)&ledger.track.eq(row.candidate_track)])
            reference=len(ledger.loc[ledger.model_id.eq(row.reference)&ledger.split.eq(row.split)&ledger.horizon.eq(row.horizon)&ledger.track.eq(row.reference_track)])
            frames.append({"family":family,"candidate":row.candidate,"reference":row.reference,"candidate_track":row.candidate_track,"reference_track":row.reference_track,"split":row.split,"horizon":row.horizon,"candidate_windows":candidate,"reference_windows":reference,"paired_windows":row.paired_windows,"paired_fraction_of_smaller":row.paired_windows/min(candidate,reference)})
    return pd.DataFrame(frames)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);ledger=pd.read_parquet(G6/"window_scores.parquet");skill=pd.read_csv(G6/"figures/skill_horizon_u.csv")
    station=station_u90(ledger);ranks,rank_stats=rank_stability(skill);sens=sensitivity_ledger();effects=sensitivity_effects(ledger,sens)
    coverage=comparison_coverage(ledger);fixed_coverage=pd.read_csv(G6/"task_coverage.csv",keep_default_na=False,na_values=[""])[["model_id","split","horizon","track","status","eligible_windows","success_count","error_count","final_status","task_coverage"]]
    for name,frame in {"station_u90.csv":station,"cross_year_ranks.csv":ranks,"rank_stability.csv":rank_stats,"sensitivity_effects.csv":effects,"common_window_coverage.csv":coverage,"fixed_task_coverage.csv":fixed_coverage}.items(): frame.to_csv(OUT/name,index=False)
    rank_svg(ranks,OUT/"cross_year_rank_u90.svg")
    for variant in ("Q1","context365"):
        forest_svg(effects.loc[effects.variant.eq(variant)].sort_values(["split","model_id"]),OUT/f"{variant.lower()}_sensitivity_forest.svg",f"{variant} versus Q0/730 paired U-90 effects",["split","model_id"])
    files=sorted(p for p in OUT.iterdir() if p.name!="manifest.json");manifest={"source_g6_manifest_sha256":hashlib.sha256((G6/"manifest.json").read_bytes()).hexdigest(),"settings":{"bootstrap_draws":5000,"block_origins":3,"seed_base":20260916},"sensitivity_inputs":{str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sens.attrs["source_files"]},"files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"station_rows":len(station),"rank_rows":len(ranks),"sensitivity_comparisons":len(effects),"files":len(files)},indent=2))


if __name__=="__main__":main()
