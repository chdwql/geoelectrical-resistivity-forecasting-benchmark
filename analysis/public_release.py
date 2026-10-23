from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "release" / "geoelectrical-resistivity-forecasting-benchmark-public.zip"
FIXED_ZIP_TIME = (2026, 9, 18, 0, 0, 0)

ROOT_FILES = ["LICENSE", "NOTICE", "README.md", "CITATION.cff", ".gitignore", ".gitattributes", ".python-version", "pyproject.toml", "uv.lock"]
PUBLIC_DIRS = ["src", "configs", "scripts", "environments", "tests"]
PUBLIC_FILES = [
    "analysis/__init__.py", "analysis/g6_extended.py", "analysis/g6_figures.py",
    "analysis/g6_remaining.py", "analysis/g6_report.py", "analysis/test_g6_report.py",
    "analysis/manuscript_figures.py", "analysis/manuscript_tables.py",
    "analysis/supplementary_figures.py", "analysis/public_release.py", "analysis/reproduce.py",
    "analysis/postreview_audit.py", "analysis/acta_figures.py",
    "examples/__init__.py", "examples/README.md", "examples/synthetic_end_to_end.py",
    "docs/REPRODUCTION.md", "docs/MODEL_ENVIRONMENTS.md",
    "docs/frozen/DATA_PROTOCOL_V1.json", "docs/frozen/MODEL_PROTOCOL_V1.json",
]
EXCLUDED_FILES = {"tests/test_manuscript_assembly.py", "tests/test_manuscript_text.py"}
SAFE_ARTIFACTS = [
    *[f"artifacts/postreview_20261023/{name}.csv" for name in
      ("season_matched_u90", "season_matched_rank_correlations", "probability_u90",
       "h1_synchronous_bootstrap", "h3_synchronous_bootstrap", "matched_effect_changes_descriptive")],
    "artifacts/data_v1/series_catalog.csv",
    "artifacts/g6/task_coverage.csv",
    "artifacts/g6/h1_primary_comparisons.csv",
    "artifacts/g6/h3_information_gain.csv",
    "artifacts/g6/figures/skill_horizon_u.csv",
    "artifacts/g6/extended/probability_summary.csv",
    "artifacts/g6/extended/cost_summary.csv",
    "artifacts/g6/extended/quality_event_strata.csv",
    "artifacts/g6/remaining/cross_year_ranks.csv",
    "artifacts/g6/remaining/sensitivity_effects.csv",
    "artifacts/g6/remaining/station_u90.csv",
]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".venv", "archive", "revoked"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".png"}
FORBIDDEN_PREFIXES = ("data/", "weights/", "artifacts/formal/", "artifacts/benchmark/", "artifacts/model", ".git/", "manuscript/", "release/", "docs/archive/", "docs/audits/", "docs/frozen/revoked/")
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?[A-Za-z0-9+/]{8,}"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def include_path(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative.as_posix() in EXCLUDED_FILES:
        return False
    if any(part in EXCLUDED_PARTS or part.startswith(".venv") for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def station_all_horizons_bytes() -> bytes:
    public_source = ROOT / "artifacts" / "g6" / "remaining" / "station_all_horizons.csv"
    if public_source.is_file():
        return public_source.read_bytes()
    x = pd.read_parquet(ROOT / "artifacts" / "g6" / "window_scores.parquet")
    x = x.loc[x.track.eq("U")]
    series = x.groupby(["model_id", "split", "horizon", "station", "series_id"], as_index=False).mase.mean()
    station = series.groupby(["model_id", "split", "horizon", "station"], as_index=False).mase.mean()
    buffer = io.StringIO()
    station.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def collect_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name in ROOT_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = path.read_bytes()
    for directory in PUBLIC_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if include_path(path):
                relative = path.relative_to(ROOT).as_posix()
                files[relative] = path.read_bytes()
    for name in PUBLIC_FILES:
        files[name] = (ROOT / name).read_bytes()
    for path in sorted((ROOT / "docs" / "model_cards").glob("*.md")):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in SAFE_ARTIFACTS:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = path.read_bytes()
    files["artifacts/g6/remaining/station_all_horizons.csv"] = station_all_horizons_bytes()
    # Every allowlisted file is text; preserve reproducible hashes across Git/OS line endings.
    files = {name: data.replace(b"\r\n", b"\n") for name, data in files.items()}
    for name, data in files.items():
        if name.startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"Forbidden release member: {name}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"Potential secret detected in release member: {name}")
    return dict(sorted(files.items()))


def build_archive(destination: Path = DEFAULT_ARCHIVE) -> dict[str, object]:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("The original-code license must be present")
    files = collect_files()
    manifest = {
        "release": "geoelectrical-resistivity-forecasting-benchmark-public",
        "version": "0.2.0",
        "scope": "reproduction code, frozen configuration, synthetic example and aggregate figure/table sources",
        "contains_restricted_observations": False,
        "contains_model_weights": False,
        "data_protocol_hash": "f7c65c07da0e8ec606b61771da2b675b1793a9ae99c1fab300b5e79f5c02f473",
        "model_protocol_hash": "20d8a8aa1f3955c6663a87942d57147d2ef95c2e29545f77f079ae856fdf0c69",
        "files": {name: digest(data) for name, data in files.items()},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in {**files, "PUBLIC_RELEASE_MANIFEST.json": manifest_bytes}.items():
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    external = {**manifest, "archive": destination.name, "archive_sha256": digest(destination.read_bytes())}
    (destination.parent / "PUBLIC_RELEASE_MANIFEST.json").write_text(json.dumps(external, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return external


def verify_archive(archive_path: Path, *, run_cleanroom: bool) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate archive members")
        if any(name.startswith(FORBIDDEN_PREFIXES) or ".." in PurePosixPath(name).parts for name in names):
            raise RuntimeError("Archive contains a forbidden or unsafe path")
        manifest = json.loads(archive.read("PUBLIC_RELEASE_MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"PUBLIC_RELEASE_MANIFEST.json"}:
            raise RuntimeError("Archive members differ from the allowlisted manifest")
        for name, expected in manifest["files"].items():
            if digest(archive.read(name)) != expected:
                raise RuntimeError(f"Archive hash mismatch: {name}")
        if not run_cleanroom:
            return {"members": len(names), "hashes_verified": len(manifest["files"]), "cleanroom": False}
        with tempfile.TemporaryDirectory(prefix="geo-bench-public-") as temporary:
            root = Path(temporary)
            archive.extractall(root)
            cleanroom_env = os.environ.copy()
            cleanroom_env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
            commands = [
                ["uv", "sync", "--frozen", "--dev"],
                ["uv", "run", "--frozen", "pytest"],
                ["uv", "run", "--frozen", "python", "analysis/reproduce.py"],
            ]
            for command in commands:
                subprocess.run(command, cwd=root, check=True, env=cleanroom_env)
            return {"members": len(names), "hashes_verified": len(manifest["files"]), "cleanroom": True, "commands": len(commands)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the restricted-data-safe public release archive.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cleanroom", action="store_true")
    args = parser.parse_args()
    result = verify_archive(args.archive, run_cleanroom=args.cleanroom) if args.verify else build_archive(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
