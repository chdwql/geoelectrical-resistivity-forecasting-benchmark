"""Recreate the synthetic example and paper figures/tables from public aggregates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    env = os.environ.copy()
    env["GEO_BENCH_RESULTS_DIR"] = str(output)
    for script in ("examples/synthetic_end_to_end.py", "analysis/manuscript_figures.py",
                   "analysis/supplementary_figures.py", "analysis/manuscript_tables.py"):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    expected = {"figures": ("figure_*.svg", 6), "supplementary_figures": ("figure_*.svg", 5),
                "tables": ("table_*.csv", 4), "supplementary_tables": ("table_*.csv", 5)}
    counts = {folder: len(list((output / folder).glob(pattern))) for folder, (pattern, _) in expected.items()}
    if any(counts[folder] != count for folder, (_, count) in expected.items()):
        raise RuntimeError(f"Reproduction output counts differ from the frozen paper: {counts}")
    subprocess.run([sys.executable, str(ROOT / 'analysis/acta_figures.py')], cwd=ROOT, env=env, check=True)
    additional = output / 'postreview'
    additional.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'artifacts/postreview_20261023').glob('*.csv'):
        shutil.copy2(source, additional / source.name)
    if len(list(additional.glob('*.csv'))) != 6:
        raise RuntimeError('Expected six public exploratory aggregate tables')
    print(json.dumps({"output": str(output), "counts": counts,
                      "scope": "figure/table reconstruction from published aggregate inputs; synthetic pipeline check"}, indent=2))


if __name__ == "__main__":
    main()
