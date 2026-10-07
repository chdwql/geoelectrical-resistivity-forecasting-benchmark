from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from .config import PROJECT_ROOT, load_yaml
from .doctor import run_checks, serialise_checks
from .deep_io import prepare_deep_requests
from .freezing import write_data_protocol_freeze, write_model_protocol_freeze
from .foundation_io import prepare_foundation_requests, prepare_timesfm_requests
from .formal_plan import build_formal_task_plan
from .ingestion import (
    build_fetch_tasks,
    fetch_partition,
    probe_oracle,
    run_fetch_tasks,
    summarise_fetch_tasks,
)
from .lightgbm_tuning import run_lightgbm_validation
from .model_registry import probe_registry
from .protocol_guard import require_model_freeze_for_test
from .quality_events import build_event_ledger
from .reporting import summarise_derived
from .robustness_tuning import run_context_validation, run_q1_validation
from .runner import run_local_models
from .scoring import score_run, score_runs
from .source_data import audit_sources, build_daily_dataset, validate_raw_acquisition
from .statistical_tuning import run_statistical_validation
from .target_domain_runner import run_lightgbm
from .validation_ledger import register_validation_runs
from .windows import generate_windows


app = typer.Typer(help="Leakage-resistant apparent-resistivity forecasting benchmark.")
data_app = typer.Typer(help="Restricted source data and auditable derived datasets.")
model_app = typer.Typer(help="Model registry and adapter capability checks.")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    """Check WSL, uv, GPU and the optional Oracle ingestion chain."""

    checks = run_checks()
    if json_output:
        typer.echo(json.dumps(serialise_checks(checks), ensure_ascii=False, indent=2))
        return
    for check in checks:
        marker = "OK" if check.ok else "MISSING"
        typer.echo(f"[{marker:7}] {check.name}: {check.detail} ({check.required_for})")


@app.command()
def test() -> None:
    """Run the synthetic contract and leakage tests."""

    raise typer.Exit(subprocess.call(["uv", "run", "--frozen", "pytest"]))


@app.command("verify-freeze")
def verify_freeze(split: str = typer.Option(...)) -> None:
    """Verify every frozen hash before a formal test process starts."""

    try:
        require_model_freeze_for_test(split)
    except RuntimeError as exc:
        raise typer.ClickException(str(exc)) from exc
    typer.echo(f"Model protocol verified for {split}")


@data_app.command("audit")
def data_audit(
    config: Path = Path("configs/data_v1.yaml"),
    output: Path = Path("artifacts/data_v1/source_audit.csv"),
) -> None:
    result = audit_sources(config, output)
    typer.echo(f"Audited {len(result)} station/factor files -> {output}")


@data_app.command("build-daily")
def data_build_daily(
    config: Path = Path("configs/data_v1.yaml"),
    output_root: Path | None = None,
) -> None:
    outputs = build_daily_dataset(config, output_root)
    typer.echo(f"Built {len(outputs)} Q0 daily partitions")


@data_app.command("audit-raw")
def data_audit_raw(
    tasks: Path = Path("data/fetch_tasks_2016_2020.csv"),
    raw_root: Path = Path("data/raw"),
    output: Path = Path("artifacts/data_v1/raw_acquisition_audit.csv"),
) -> None:
    result = validate_raw_acquisition(tasks, raw_root, output)
    invalid = int(result["validation_status"].ne("valid").sum())
    typer.echo(
        f"Validated {len(result)} raw partitions; invalid={invalid}; "
        f"rows={int(result['rows'].fillna(0).sum())} -> {output}"
    )
    if invalid:
        raise typer.Exit(2)


@data_app.command("build-windows")
def data_build_windows(
    daily_glob: str = "artifacts/data_v1/daily_q0/**/*.parquet",
    protocol: Path = Path("configs/protocol_v1.yaml"),
    series_config: Path = Path("configs/series_v1.yaml"),
    output: Path = Path("artifacts/data_v1/windows.parquet"),
) -> None:
    paths = sorted(PROJECT_ROOT.glob(daily_glob))
    if not paths:
        raise typer.BadParameter(f"No daily partitions matched: {daily_glob}")
    daily = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    config = load_yaml(protocol)
    series = load_yaml(series_config)
    splits = {
        name: (date.fromisoformat(str(bounds[0])), date.fromisoformat(str(bounds[1])))
        for name, bounds in config["splits"].items()
    }
    windows = generate_windows(
        daily,
        splits=splits,
        horizons=[int(value) for value in config["horizons"]],
        tracks=list(config["tracks"]),
        context_days=int(config["context_days"]),
        minimum_target_coverage=float(config["minimum_target_coverage"]),
        minimum_context_coverage=float(config["minimum_context_coverage"]),
        minimum_auxiliary_context_coverage=float(config["minimum_auxiliary_context_coverage"]),
        minimum_oracle_future_coverage=float(config["minimum_oracle_future_coverage"]),
        target_series_ids=[str(value) for value in series["target_series"]],
    )
    destination = PROJECT_ROOT / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(destination, index=False)
    typer.echo(f"Built {len(windows)} windows -> {output}")


@data_app.command("summarize")
def data_summarize() -> None:
    results = summarise_derived()
    typer.echo(
        f"Summarized {len(results['catalog'])} series, "
        f"{len(results['coverage'])} station/factor/year groups and "
        f"{int(results['windows']['windows'].sum())} windows"
    )


@data_app.command("build-events")
def data_build_events(
    quality: Path = Path("configs/quality_v1.yaml"),
    series: Path = Path("configs/series_v1.yaml"),
    output: Path = Path("artifacts/data_v1/quality_event_candidates.csv"),
) -> None:
    result = build_event_ledger(quality_path=quality, series_path=series, output_path=output)
    typer.echo(f"Built {len(result)} retrospective quality-event candidates -> {output}")


@data_app.command("fetch")
def data_fetch(
    probe: bool = typer.Option(False, "--probe", help="Only test Oracle metadata access."),
    station: str | None = None,
    point: str | None = None,
    factor: str | None = None,
    start: str | None = None,
    end: str | None = None,
    sample_rate: str | None = typer.Option(None, "--sample-rate"),
    alias: str = "cn",
) -> None:
    if probe:
        try:
            result = probe_oracle(alias)
        except RuntimeError as exc:
            raise typer.ClickException(str(exc)) from exc
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    required = {"station": station, "point": point, "factor": factor, "start": start, "end": end}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise typer.BadParameter(f"Missing fetch options: {', '.join(missing)}")
    record = fetch_partition(
        station_id=station or "",
        point=point or "",
        factor=factor or "",
        start=start or "",
        end=end or "",
        sample_rate=sample_rate,
        alias=alias,
    )
    typer.echo(json.dumps(record, ensure_ascii=False, indent=2))


@data_app.command("plan-fetch")
def data_plan_fetch(
    audit: Path = Path("artifacts/data_v1/source_audit.csv"),
    output: Path = Path("artifacts/data_v1/fetch_tasks_2016_2020.csv"),
) -> None:
    tasks = build_fetch_tasks(audit, output)
    typer.echo(f"Planned {len(tasks)} immutable fetch partitions -> {output}")


@data_app.command("run-fetch-tasks")
def data_run_fetch_tasks(
    tasks: Path = Path("artifacts/data_v1/fetch_tasks_2016_2020.csv"),
    limit: int | None = typer.Option(None, min=1),
    alias: str = "cn",
) -> None:
    try:
        records = run_fetch_tasks(tasks, limit=limit, alias=alias)
    except RuntimeError as exc:
        raise typer.ClickException(str(exc)) from exc
    succeeded = sum(record.get("status") in {"success", "empty", "skipped_existing"} for record in records)
    typer.echo(f"Finished {len(records)} tasks; non-error={succeeded}; errors={len(records) - succeeded}")


@data_app.command("fetch-status")
def data_fetch_status(
    tasks: Path = Path("artifacts/data_v1/fetch_tasks_2016_2020.csv"),
) -> None:
    typer.echo(summarise_fetch_tasks(tasks).to_string(index=False))


@model_app.command("probe")
def model_probe(json_output: bool = typer.Option(False, "--json")) -> None:
    records = probe_registry()
    if json_output:
        typer.echo(json.dumps(records, ensure_ascii=False, indent=2))
        return
    for model in records:
        source = model.get("local_path") or model.get("upstream") or "implemented locally"
        typer.echo(f"{model['id']:20} {model['family']:18} {model['tracks']} {source}")


@model_app.command("run")
def model_run(
    split: str = typer.Option(...),
    horizon: int = typer.Option(..., min=1),
    models: str = typer.Option("persistence,drift,seasonal_365"),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    input_variant: str = typer.Option("Q0"),
    context_days: int = typer.Option(730),
    q1_window: int = typer.Option(90, min=2),
    q1_threshold: float = typer.Option(8.0, min=0.01),
    q1_minimum_history: int = typer.Option(30, min=2),
) -> None:
    destination = run_local_models(
        split=split,
        horizon=horizon,
        model_ids=[value.strip() for value in models.split(",") if value.strip()],
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        input_variant=input_variant,
        context_days=context_days,
        q1_window=q1_window,
        q1_threshold=q1_threshold,
        q1_minimum_history=q1_minimum_history,
    )
    typer.echo(f"Run written to {destination.relative_to(PROJECT_ROOT)}")


@model_app.command("prepare-timesfm")
def model_prepare_timesfm(
    split: str = typer.Option(...),
    horizon: int = typer.Option(..., min=1),
    track: str = typer.Option(...),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    input_variant: str = typer.Option("Q0"),
    context_days: int = typer.Option(730),
) -> None:
    destination = prepare_timesfm_requests(
        split=split,
        horizon=horizon,
        track=track,
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        input_variant=input_variant,
        context_days=context_days,
    )
    typer.echo(f"TimesFM request set -> {destination.relative_to(PROJECT_ROOT)}")


@model_app.command("prepare-foundation")
def model_prepare_foundation(
    model_id: str = typer.Option(...),
    split: str = typer.Option(...),
    horizon: int = typer.Option(..., min=1),
    track: str = typer.Option(...),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    input_variant: str = typer.Option("Q0"),
    context_days: int = typer.Option(730),
    q1_window: int = typer.Option(30, min=2),
    q1_threshold: float = typer.Option(10.0, min=0.01),
    q1_minimum_history: int = typer.Option(14, min=2),
) -> None:
    """Build a model-isolated Parquet/JSONL request set without target truth."""

    destination = prepare_foundation_requests(
        model_id=model_id,
        split=split,
        horizon=horizon,
        track=track,
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        input_variant=input_variant,
        context_days=context_days,
        q1_window=q1_window,
        q1_threshold=q1_threshold,
        q1_minimum_history=q1_minimum_history,
    )
    typer.echo(f"{model_id} request set -> {destination.relative_to(PROJECT_ROOT)}")


@model_app.command("run-lightgbm")
def model_run_lightgbm(
    split: str = typer.Option("validation"),
    horizon: int = typer.Option(..., min=1),
    track: str = typer.Option(...),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    max_train_windows_per_series: int | None = typer.Option(None, min=1),
    input_variant: str = typer.Option("Q0"),
    context_days: int = typer.Option(730),
) -> None:
    destination = run_lightgbm(
        split=split,
        horizon=horizon,
        track=track,
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        max_train_windows_per_series=max_train_windows_per_series,
        input_variant=input_variant,
        context_days=context_days,
    )
    typer.echo(f"LightGBM run written to {destination.relative_to(PROJECT_ROOT)}")


@model_app.command("prepare-deep")
def model_prepare_deep(
    model_id: str = typer.Option(...),
    split: str = typer.Option("validation"),
    horizon: int = typer.Option(..., min=1),
    track: str = typer.Option(...),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    max_train_windows_per_series: int | None = typer.Option(None, min=1),
    input_variant: str = typer.Option("Q0"),
    context_days: int = typer.Option(730),
) -> None:
    destination = prepare_deep_requests(
        model_id=model_id,
        split=split,
        horizon=horizon,
        track=track,
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        max_train_windows_per_series=max_train_windows_per_series,
        input_variant=input_variant,
        context_days=context_days,
    )
    typer.echo(f"Deep-model request set -> {destination.relative_to(PROJECT_ROOT)}")


@app.command("evaluate")
def evaluate(run_directory: Path) -> None:
    scores, summary = score_run(run_directory)
    typer.echo(f"Scored {len(scores)} successful model/windows")
    typer.echo(summary.to_string(index=False))


@app.command("evaluate-many")
def evaluate_many(run_directories: list[Path]) -> None:
    """Score multiple runs with one restricted-truth load."""

    summary = score_runs(run_directories)
    typer.echo(summary.to_string(index=False))


@app.command("register-validation")
def register_validation(run_directories: list[Path]) -> None:
    """Add scored validation-only runs to the immutable-input G4 trial ledger."""

    records = register_validation_runs(run_directories)
    typer.echo(f"Registered {len(records)} new model trials")


@app.command("tune-statistical")
def tune_statistical(
    horizon: int = typer.Option(90, min=1),
    families: str = typer.Option("ets,arima,fourier_ridge"),
    stations: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
    candidates: str | None = typer.Option(None),
) -> None:
    """Run, score and register statistical candidates on validation only."""

    destination = run_statistical_validation(
        horizon=horizon,
        families={value.strip() for value in families.split(",") if value.strip()},
        stations=[value.strip() for value in stations.split(",")] if stations else None,
        limit=limit,
        candidates={value.strip() for value in candidates.split(",") if value.strip()}
        if candidates
        else None,
    )
    typer.echo(f"Statistical validation run -> {destination.relative_to(PROJECT_ROOT)}")


@app.command("tune-lightgbm")
def tune_lightgbm(
    track: str = typer.Option(...),
    horizon: int = typer.Option(90, min=1),
) -> None:
    """Run, score and register the fixed LightGBM grid on validation only."""

    destination = run_lightgbm_validation(track=track, horizon=horizon)
    typer.echo(f"LightGBM validation run -> {destination.relative_to(PROJECT_ROOT)}")


@app.command("tune-q1")
def tune_q1() -> None:
    """Select the fixed causal robust-input grid on validation U-90 only."""

    result = run_q1_validation()
    typer.echo(result.to_string(index=False))


@app.command("validate-context")
def validate_context() -> None:
    """Exercise the preregistered 365-day context sensitivity on validation."""

    destination = run_context_validation()
    typer.echo(f"365-day validation run -> {destination.relative_to(PROJECT_ROOT)}")


@app.command("freeze")
def freeze(
    output: Path = Path("docs/frozen/DATA_PROTOCOL_V1.json"),
) -> None:
    try:
        record = write_data_protocol_freeze(output)
    except FileExistsError as exc:
        raise typer.ClickException(str(exc)) from exc
    typer.echo(f"Frozen data protocol {record['protocol_hash']} -> {output}")


@app.command("freeze-models")
def freeze_models(
    output: Path = Path("docs/frozen/MODEL_PROTOCOL_V1.json"),
) -> None:
    """Freeze G4 selections and unlock the formal test runners."""

    try:
        record = write_model_protocol_freeze(output)
    except (FileExistsError, FileNotFoundError, RuntimeError) as exc:
        raise typer.ClickException(str(exc)) from exc
    typer.echo(f"Frozen model protocol {record['model_protocol_hash']} -> {output}")


@app.command("plan-formal")
def plan_formal() -> None:
    """Create the fixed G5 task denominator after model-protocol freeze."""

    destination, rows = build_formal_task_plan()
    planned = [row for row in rows if row["status"] == "planned"]
    typer.echo(
        f"Formal plan -> {destination.relative_to(PROJECT_ROOT)}; "
        f"specs={len(rows)}, planned={len(planned)}, "
        f"model_windows={sum(int(row['eligible_windows']) for row in planned)}"
    )
