# Geoelectrical Resistivity Forecasting Benchmark

Reproduction software for a benchmark of time-series foundation models on geophysical apparent-resistivity observations. The repository contains the methods and inputs needed to check the reported numerical tables and figures, plus code for the full restricted-data workflow.

## Reproduce figures, tables, and the synthetic example

In Ubuntu WSL2, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --frozen --dev
uv run --frozen python analysis/reproduce.py
uv run --frozen pytest
```

The reproduction command generates 11 SVG figures and 9 CSV tables in `results/`, with source/output hashes. It also runs a synthetic demonstration of data contracts, forecasts, and scoring. It reads only aggregate result inputs and synthetic observations; no database access or GPU is needed for these steps.

## Contents

- `src/`: data contracts, daily aggregation, causal windows, model requests, adapters, scoring, and execution.
- `configs/` and `docs/frozen/`: the exact protocol and model configurations.
- `environments/`: isolated foundation-model and target-domain model runners with uv locks.
- `analysis/`: numerical evaluation and figure/table reconstruction.
- `artifacts/`: allowlisted aggregate result inputs, not the private observation dataset.
- `examples/` and `tests/`: synthetic demonstration and numerical/contract tests.
- `docs/`: reproduction instructions and model/runtime specifications.

Full forecast and statistical replication needs the approved observation snapshot, private ledgers, and separately downloaded weights. Aggregate-based reconstruction checks the published tables and visualizations; it cannot independently verify all forecasts. See [reproduction scope and data access](docs/REPRODUCTION.md) and [model environments](docs/MODEL_ENVIRONMENTS.md).

## Build a distributable snapshot

```bash
uv run --frozen python analysis/public_release.py
uv run --frozen python analysis/public_release.py --verify --cleanroom
```

The archive uses an explicit file allowlist and a per-file SHA-256 manifest. Its clean-room check installs frozen dependencies, runs numerical tests, and reconstructs the figures/tables in an isolated directory. Manuscripts, submission files, research history, restricted observations, and model weights are excluded.

Original code is licensed under [Apache-2.0](LICENSE). [NOTICE](NOTICE) describes the third-party and restricted-data boundaries. Software citation metadata and the version are in [CITATION.cff](CITATION.cff).
