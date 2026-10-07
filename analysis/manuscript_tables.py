from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

try:
    from analysis.manuscript_figures import MODEL_LABELS
except ModuleNotFoundError:  # Support direct execution from the project root.
    from manuscript_figures import MODEL_LABELS


ROOT = Path(__file__).resolve().parents[1]
G6 = ROOT / "artifacts" / "g6"
RESULTS = Path(os.environ.get("GEO_BENCH_RESULTS_DIR", str(ROOT / "manuscript"))).resolve()
OUT = RESULTS / "tables"
SUPP = RESULTS / "supplementary_tables"


def read_task_coverage(path: Path) -> pd.DataFrame:
    """Preserve the explicit unsupported state, without coercing numeric columns."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def markdown(frame: pd.DataFrame) -> str:
    values = frame.fillna("N/A").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows]) + "\n"


def table1() -> pd.DataFrame:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol_v1.yaml").read_text(encoding="utf-8"))
    series = yaml.safe_load((ROOT / "configs" / "series_v1.yaml").read_text(encoding="utf-8"))
    catalog = pd.read_csv(ROOT / "artifacts" / "data_v1" / "series_catalog.csv")
    coverage = pd.read_csv(G6 / "task_coverage.csv")
    runnable = coverage.loc[coverage.status.eq("planned")]
    task_count = int(runnable.task_count.sum())
    success_count = int(runnable.success_count.sum())
    if (len(runnable), task_count, success_count) != (252, 161_192, 161_148):
        raise ValueError(
            "Frozen task denominator mismatch: "
            f"specifications={len(runnable)}, tasks={task_count}, successes={success_count}"
        )
    rows = [
        ("Observation period", f"{catalog.first_date.min()} to {catalog.last_date.max()}"),
        ("Stations", str(catalog.station.nunique())),
        ("Evaluation targets", f"{len(series['target_series'])} predefined; 19 with testable windows"),
        ("Context-only resistivity series", str(len(series["context_only_resistivity_series"]))),
        ("Auxiliary series", str(int(catalog.factor.ne("resistivity").sum()))),
        ("Forecast horizons", ", ".join(map(str, protocol["horizons"])) + " days"),
        ("Primary endpoint", f"{protocol['primary_horizon']} days"),
        ("Information tracks", "U target; M same-station resistivity; X historical environment; O defined but not evaluated"),
        ("Maximum primary context", f"{protocol['context_days']} days"),
        ("Test periods", "A: 2024–2025; B: 2026 through 31 August"),
        ("Logical specifications", str(len(coverage))),
        ("Runnable specifications", str(len(runnable))),
        ("Supported model–window tasks", f"{task_count:,}"),
        ("Successful tasks", f"{success_count:,}"),
    ]
    return pd.DataFrame(rows, columns=["Characteristic", "Frozen benchmark value"])


def effect_table(path: Path, information: bool = False) -> pd.DataFrame:
    x = pd.read_csv(path)
    keep = ["split", "candidate", "reference", "candidate_track", "reference_track", "paired_windows", "paired_stations", "effect_delta_mase", "ci_lower", "ci_upper", "holm_p_value"]
    x = x[keep].copy()
    x["candidate"] = x.candidate.map(MODEL_LABELS).fillna(x.candidate)
    x["reference"] = x.reference.map(MODEL_LABELS).fillna(x.reference)
    x["split"] = x.split.map({"test_a": "Test A", "test_b": "Test B"})
    for col in ("effect_delta_mase", "ci_lower", "ci_upper"):
        x[col] = x[col].map(lambda value: f"{value:.3f}")
    x["holm_p_value"] = x.holm_p_value.map(lambda value: f"{value:.3f}")
    if information:
        x["comparison"] = x.candidate_track + "−" + x.reference_track
        return x[["split", "candidate", "comparison", "paired_windows", "paired_stations", "effect_delta_mase", "ci_lower", "ci_upper", "holm_p_value"]]
    return x[["split", "candidate", "reference", "paired_windows", "paired_stations", "effect_delta_mase", "ci_lower", "ci_upper", "holm_p_value"]]


def table4() -> pd.DataFrame:
    probability = pd.read_csv(G6 / "extended" / "probability_summary.csv")
    probability = probability.loc[probability.horizon.eq(90) & probability.track.eq("U")]
    costs = pd.read_csv(G6 / "extended" / "cost_summary.csv")
    costs = costs.loc[costs.horizon.eq(90) & costs.track.eq("U")]
    x = probability.merge(costs, on=["model_id", "split", "horizon", "track"], validate="one_to_one")
    x["model"] = x.model_id.map(MODEL_LABELS)
    x["period"] = x.split.map({"test_a": "Test A", "test_b": "Test B"})
    columns = ["model", "period", "wis", "coverage_80", "width_80", "quantile_crossing_rate", "median_inference_seconds", "peak_memory_mb"]
    x = x[columns].sort_values(["period", "wis"])
    formats = {"wis": 4, "coverage_80": 3, "width_80": 3, "quantile_crossing_rate": 5, "median_inference_seconds": 4, "peak_memory_mb": 0}
    for column, digits in formats.items():
        x[column] = x[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.{digits}f}")
    return x


def model_registry() -> pd.DataFrame:
    registry = yaml.safe_load((ROOT / "configs" / "model_registry.yaml").read_text(encoding="utf-8"))
    rows = []
    for model in registry["models"]:
        license_value = model.get("license") or model.get("source_license") or "/".join(filter(None, [model.get("code_license"), model.get("weight_license")])) or "local implementation"
        rows.append({
            "model_id": model["id"], "model": MODEL_LABELS[model["id"]], "family": model["family"], "training": model["training"],
            "tracks": "/".join(model["tracks"]), "probabilistic": model["probabilistic"], "version": model.get("runtime_version") or model.get("runtime_torch") or "N/A",
            "source_commit": model.get("source_commit") or model.get("source_reference_commit") or "N/A", "weight_hash": model.get("weight_hash", "N/A"), "license": license_value,
        })
    return pd.DataFrame(rows)


def per_station_all_horizons() -> pd.DataFrame:
    public_aggregate = G6 / "remaining" / "station_all_horizons.csv"
    if public_aggregate.exists():
        return pd.read_csv(public_aggregate, float_precision="round_trip")
    x = pd.read_parquet(G6 / "window_scores.parquet")
    x = x.loc[x.track.eq("U")]
    series = x.groupby(["model_id", "split", "horizon", "station", "series_id"], as_index=False).mase.mean()
    return series.groupby(["model_id", "split", "horizon", "station"], as_index=False).mase.mean()


def write_table(frame: pd.DataFrame, stem: str, directory: Path, *, write_markdown: bool) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    paths = [csv_path]
    if write_markdown:
        md_path = directory / f"{stem}.md"
        md_path.write_text(markdown(frame), encoding="utf-8")
        paths.append(md_path)
    return paths


def main() -> None:
    generated: list[Path] = []
    generated += write_table(table1(), "table_1_benchmark", OUT, write_markdown=True)
    generated += write_table(effect_table(G6 / "h1_primary_comparisons.csv"), "table_2_h1", OUT, write_markdown=True)
    generated += write_table(effect_table(G6 / "h3_information_gain.csv", information=True), "table_3_h3", OUT, write_markdown=True)
    generated += write_table(table4(), "table_4_probability_cost", OUT, write_markdown=True)
    generated += write_table(model_registry(), "table_s1_model_registry", SUPP, write_markdown=False)
    coverage = read_task_coverage(G6 / "task_coverage.csv")
    generated += write_table(coverage, "table_s2_task_coverage", SUPP, write_markdown=False)
    generated += write_table(per_station_all_horizons(), "table_s3_station_scores", SUPP, write_markdown=False)
    generated += write_table(pd.read_csv(G6 / "remaining" / "sensitivity_effects.csv"), "table_s4_sensitivity", SUPP, write_markdown=False)
    generated += write_table(pd.read_csv(G6 / "extended" / "cost_summary.csv"), "table_s5_cost", SUPP, write_markdown=False)
    manifest = {"files": {str(p.relative_to(ROOT if p.is_relative_to(ROOT) else RESULTS)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"generated_files": len(generated), "manifest": str((OUT / 'manifest.json').relative_to(ROOT if OUT.is_relative_to(ROOT) else RESULTS))}, indent=2))


if __name__ == "__main__":
    main()
