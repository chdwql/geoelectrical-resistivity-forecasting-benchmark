# Public synthetic end-to-end example

This example exercises the production request contracts, simple adapters, prediction-ledger schema, and point metrics without reading or reproducing any restricted observation.

From Ubuntu WSL2 at the project root, run:

```bash
uv run --frozen python examples/synthetic_end_to_end.py
```

The command deterministically creates `examples/output/` containing:

- `daily.csv`: four unrestricted synthetic apparent-resistivity series in the daily data contract;
- `windows.csv`: U-track 30- and 90-day windows with 730-day causal histories;
- `predictions.csv`: prediction-ledger rows for persistence, linear drift, and seasonal persistence;
- `scores.csv`: window-level MAE, RMSE, and MASE;
- `manifest.json`: seed, protocol identifier, task counts, and SHA-256 hashes.

The synthetic series deliberately include a short missing interval, a level shift, and a spike. These are artificial quality-control examples and have no earthquake interpretation. The example is a contract and reproducibility demonstration, not a miniature estimate of the real benchmark's model ranking.
