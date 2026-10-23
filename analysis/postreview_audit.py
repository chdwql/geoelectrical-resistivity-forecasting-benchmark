"""Post-review sensitivity analyses; never overwrite frozen manuscript results.

Run in WSL: uv run --frozen python -m analysis.postreview_audit
All additional inference is exploratory, conditional on these fixed stations.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.g6_report import file_sha256, hierarchical_effect, holm_adjust
from analysis.g6_extended import hierarchical_metrics
from geoelectrical_resistivity_forecasting_benchmark.runner import _load_daily

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/postreview_20261023'
SEED = 20261023

def synchronized_bootstrap(pairs, draws=5000, block=3, seed=SEED):
    """Resample calendar months jointly across all channels and fixed stations."""
    frame = pairs.copy()
    frame['origin'] = pd.to_datetime(frame.origin)
    matrix = frame.pivot(index='origin', columns=['station', 'series_id'], values='delta_mase')
    calendar = pd.date_range(matrix.index.min(), matrix.index.max(), freq='MS')
    matrix = matrix.reindex(calendar)
    values = matrix.to_numpy()
    station_indices = [np.flatnonzero(matrix.columns.get_level_values(0) == s)
                       for s in matrix.columns.get_level_values(0).unique()]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        starts = rng.integers(0, len(calendar), size=int(np.ceil(len(calendar) / block)))
        indices = ((starts[:, None] + np.arange(block)) % len(calendar)).ravel()[:len(calendar)]
        sample = values[indices]
        # Never silently change the set of channels/stations in a replicate.
        if not np.isfinite(sample).any(axis=0).all():
            estimates.append(np.nan)
            continue
        channel_means = np.nanmean(sample, axis=0)
        estimates.append(np.mean([channel_means[i].mean() for i in station_indices]))
    return np.asarray(estimates)

def paired(ledger, row):
    keys = ['station', 'series_id', 'origin']
    base = ledger[ledger.split.eq(row.split) & ledger.horizon.eq(90)]
    a = base[base.model_id.eq(row.candidate) & base.track.eq(row.candidate_track)]
    b = base[base.model_id.eq(row.reference) & base.track.eq(row.reference_track)]
    result = a[keys + ['mase']].merge(b[keys + ['mase']], on=keys, validate='one_to_one')
    result['delta_mase'] = result.mase_x - result.mase_y
    return result[np.isfinite(result.delta_mase)].copy()

def historical_scale(history):
    x = np.asarray(history, float)
    good = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    return np.abs(np.diff(x)[good]).mean() if good.any() else np.nan

def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite audit: {OUT}')
    ledger_path = ROOT / 'artifacts/g6/window_scores.parquet'
    ledger = pd.read_parquet(ledger_path)
    ledger['origin'] = pd.to_datetime(ledger.origin)
    outputs = {}
    # Strict common support: every model and year must have each series/month.
    u = ledger[ledger.track.eq('U') & ledger.horizon.eq(90) & ledger.origin.dt.month.le(6)].copy()
    u['year'], u['month'] = u.origin.dt.year, u.origin.dt.month
    valid = u[np.isfinite(u.mase)]
    n_models, n_years = valid.model_id.nunique(), valid.year.nunique()
    counts = valid.groupby(['station', 'series_id', 'month']).size()
    keys = counts[counts.eq(n_models * n_years)].reset_index()[['station', 'series_id', 'month']]
    matched = valid.merge(keys, on=['station', 'series_id', 'month'], validate='many_to_one')
    scores = hierarchical_metrics(matched, ['mase'], ['model_id', 'year'])
    scores['rank'] = scores.groupby('year').mase.rank()
    outputs['season_matched_u90.csv'] = scores
    outputs['season_matched_support.csv'] = matched[['model_id','year','station','series_id','month','window_id']]
    ranks = scores.pivot(index='model_id', columns='year', values='rank')
    outputs['season_matched_rank_correlations.csv'] = ranks.corr(method='spearman').reset_index()
    # Reconstruct actual Q0 scales, not MAE/MASE ratios (undefined for zero error).
    windows_path = ROOT / 'artifacts/data_v1/windows.parquet'
    windows = pd.read_parquet(windows_path).set_index('window_id')
    daily = _load_daily('artifacts/data_v1/daily_q0/**/*.parquet')
    lookup = {k: g.set_index('date').value for k, g in daily[daily.factor.eq('resistivity')].groupby('series_id')}
    prob = ledger[ledger.track.eq('U') & ledger.horizon.eq(90) & ledger.wis.notna()].copy()
    scales = {}
    for wid, sid in prob[['window_id','series_id']].drop_duplicates().itertuples(index=False, name=None):
        w = windows.loc[wid]
        scales[wid] = historical_scale(lookup[sid].reindex(pd.date_range(w.context_start, w.context_end)))
    prob['scale'] = prob.window_id.map(scales)
    if not (np.isfinite(prob.scale) & prob.scale.gt(0)).all():
        raise ValueError('Undefined probability scale; explicit denominator policy required')
    np.testing.assert_allclose(prob.mae / prob.scale, prob.mase, rtol=1e-10, atol=1e-10)
    metrics = ['wis','width_80','mean_quantile_loss']
    for metric in metrics:
        prob['scaled_' + metric] = prob[metric] / prob.scale
    outputs['probability_u90.csv'] = hierarchical_metrics(prob, metrics + ['scaled_' + m for m in metrics] + ['coverage_80'], ['model_id','split'])
    # Same H1/H3 comparison families, alternative dependence assumption.
    leave_out, station_effects = [], []
    for family, filename in [('h1','h1_primary_comparisons.csv'),('h3','h3_information_gain.csv')]:
        original = pd.read_csv(ROOT / 'artifacts/g6' / filename)
        rows = []
        for i, row in enumerate(original.itertuples(index=False)):
            pairs = paired(ledger, row)
            boot = synchronized_bootstrap(pairs, seed=SEED+i)
            valid_boot = boot[np.isfinite(boot)]
            if len(valid_boot) != len(boot):
                raise ValueError('Bootstrap replicate lost a channel; inspect missingness before inference')
            p = min(1., 2 * min((np.count_nonzero(boot <= 0)+1)/5001, (np.count_nonzero(boot >= 0)+1)/5001))
            identity = dict(family=family, candidate=row.candidate, reference=row.reference,
                            track=row.candidate_track, split=row.split)
            lo, hi = np.quantile(boot, [.025,.975])
            rows.append(dict(**identity, effect=hierarchical_effect(pairs), ci_lower=lo, ci_upper=hi,
                             p_value=p, original_holm_p=row.holm_p_value, windows=len(pairs), origins=pairs.origin.nunique()))
            for station in sorted(pairs.station.unique()):
                leave_out.append(dict(**identity, omitted_station=station, effect=hierarchical_effect(pairs[pairs.station.ne(station)])))
                station_effects.append(dict(**identity, station=station, effect=hierarchical_effect(pairs[pairs.station.eq(station)])))
        result = pd.DataFrame(rows)
        result['holm_p'] = holm_adjust(result.p_value)
        outputs[family + '_synchronous_bootstrap.csv'] = result
    outputs['leave_one_station_out.csv'] = pd.DataFrame(leave_out)
    outputs['station_effects.csv'] = pd.DataFrame(station_effects)
    # Direct descriptive effect changes on matched station/channel/month support.
    changes = []
    for model in ['timesfm_3','chronos_2','moirai_2_small','toto_2_313m']:
        for ref in ['seasonal_365','fourier_ridge']:
            pivot = matched[matched.model_id.isin([model,ref])].pivot(index=['station','series_id','month','year'], columns='model_id', values='mase')
            delta = (pivot[model]-pivot[ref]).unstack('year')
            for year in [2024,2025]:
                change = (delta[2026]-delta[year]).rename('delta_mase').reset_index()
                changes.append(dict(candidate=model, reference=ref, earlier_year=year,
                                    change_2026_minus_earlier=hierarchical_effect(change), pairs=len(change)))
    outputs['matched_effect_changes_descriptive.csv'] = pd.DataFrame(changes)
    OUT.mkdir()
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    sources = [ledger_path, windows_path, Path(__file__), ROOT/'analysis/g6_report.py', ROOT/'analysis/g6_extended.py', ROOT/'src/geoelectrical_resistivity_forecasting_benchmark/metrics.py']
    sources += sorted((ROOT/'artifacts/data_v1/daily_q0').rglob('*.parquet'))
    sources += [ROOT/'artifacts/g6'/n for n in ['h1_primary_comparisons.csv','h3_information_gain.csv','manifest.json']]
    manifest = dict(status='post-review exploratory; not preregistered', seed=SEED, draws=5000, block_months=3,
        matched_models=n_models, matched_years=n_years, matched_rows=len(matched),
        source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in sources},
        outputs={n:dict(rows=len(f),sha256=file_sha256(OUT/n)) for n,f in outputs.items()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
