# Reproduction scope and data access

This repository supports checking results from a benchmark of daily apparent-resistivity observations at nine geophysical monitoring stations. It forecasts observations; it does not estimate earthquake occurrence, location, magnitude, or probability.

## Available without restricted observations

`uv run --frozen python analysis/reproduce.py` exercises the synthetic end-to-end example and regenerates six main figures, five supplementary figures, four main CSV tables, and five supplementary CSV tables in `results/`. The numerical inputs are the exact aggregate source tables in `artifacts/g6/` and the channel catalog in `artifacts/data_v1/series_catalog.csv`. The release manifest records every input hash. Generated figure and table manifests record source and output hashes.

The source aggregates preserve the fixed task denominator, unsupported N/A specifications, model failures, station summaries, paired effects, uncertainty intervals, probability summaries, and timing summaries. The supported denominator is 161,192 model-window tasks, of which 161,148 succeeded. There are 390 logical specifications, including 138 unsupported specifications. Reconstructing figures from these aggregates verifies published tabulation and plotting, not independent rerunning of forecasts or bootstrap inference from per-window predictions.

## Complete numerical replication with approved data access

Version 0.2.0 adds `analysis/postreview_audit.py`, six aggregate exploratory-analysis CSVs under `artifacts/postreview_20261023/`, and `analysis/acta_figures.py`. The reproduction entry point also exports the revised 11 figures as vector PDF/EPS/SVG in `results/acta_tex/` and copies the six additional CSV tables to `results/postreview/`. Table S6 is `season_matched_u90.csv`, S7 is `probability_u90.csv`, S8/S9 are the two synchronous bootstrap tables. The rank-correlation and matched-effect-change CSVs provide the remaining aggregate checks. These analyses were added after the original submission; they do not replace the frozen original comparisons.

Running `uv run --frozen python analysis/postreview_audit.py` requires the approved private window-score ledger, daily Q0 snapshot and window manifest. It refuses to overwrite an existing analysis directory. Public aggregates allow checking the reported values and figure reconstruction; they do not allow independently rerunning the bootstrap or matched-support selection. Window identifiers, station/channel/month support records and private prediction ledgers are not published.

The submission figures use Arial (8 pt labels, 9 pt panel headings). Install Arial through a licensed system installation for identical typography; fonts are not redistributed. On systems without Arial, Matplotlib substitutes an available font, affecting layout but not numerical results.

The production implementation in `src/`, statistical evaluation scripts in `analysis/g6_*.py`, frozen configuration in `configs/`, and model-specific runners and locks in `environments/` support the complete workflow. `docs/frozen/` binds the data and model protocols. Model identities, weight hashes, licenses, capability limits, context rules, split definitions, and training choices are retained in the registry and frozen records. Third-party weights must be obtained separately under their respective licenses.

Full replication of forecasts, window scores, bootstrap statistics, and quality-event summaries additionally needs the approved observation snapshot and derived private ledgers. These are excluded from this repository. Shandong Earthquake Agency has approved confidential sharing of the study observations with editors and reviewers; contact Qinglin Wang at chd_wql@qq.com to coordinate delivery with the editorial office. Public readers can reproduce the aggregate-based figures/tables and synthetic example. Access to observations for further scientific reuse is subject to the provider's conditions.

## Data contracts

Daily values: `date, station, point, item, factor, value, valid_count, expected_count, coverage, quality_flags, source_hash`.

Windows: `window_id, series_id, origin, context_start/end, target_start/end, horizon, track, split, mask_hash, eligible, exclusion_reason`.

Prediction ledger: `run_id, protocol_hash, model_id, weight_hash, window_id, target_date, point_forecast, quantiles, seed, status, runtime, peak_memory, failure_reason`.

These contracts are implemented in the production code. Q0 scoring truth is never imputed; Q1 affects causal inputs only. All tracks respect the origin's historical information boundary.

## Repository and archive

The repository contains executable methods, configurations, environment locks, numerical tests, technical usage documentation, and aggregate source inputs. Manuscript text, references, Word/PDF submission files, correspondence, research plans, decision logs, and intermediate review reports are kept outside the archive. No source observations, per-day real-data predictions, credentials, caches, or model weights are distributed.

The original software is licensed under Apache-2.0. Third-party dependencies and weights keep their original licenses; see `NOTICE`, `configs/model_registry.yaml`, and the model cards. The version is recorded in `CITATION.cff`. A software-archive DOI will be added only once an archive service actually assigns it. A private GitHub repository is preparation storage, not unrestricted public access.
