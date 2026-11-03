"""Primary question: outcome-associated behavior at comparable click volume.

Outcome-defined 1:1 descriptive comparison, not causal matching or a population
failure-rate estimator. Row-level pairs must remain in ignored output folders.
"""
from bisect import bisect_left
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CONCENTRATION_FEATURES


def make_pairs(frame: pd.DataFrame, tolerance: float, seed: int = 42) -> pd.DataFrame:
    """Exact matches first, then nearest available clicks; no reuse per run.

    For positive counts, max(clicks)/min(clicks) <= 1+tolerance.
    Zero clicks can only match zero clicks. Greedy matches are not claimed optimal.
    """
    population = frame[frame.final_result.isin(['Fail', 'Pass', 'Distinction'])].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pairs = []
    for (module, presentation), group in population.groupby(['code_module', 'code_presentation']):
        failures = rng.permutation(group.index[group.final_result.eq('Fail')]).tolist()
        passing = rng.permutation(group.index[~group.final_result.eq('Fail')]).tolist()
        passing.sort(key=lambda i: population.at[i, 'total_clicks'])
        clicks = [float(population.at[i, 'total_clicks']) for i in passing]
        remaining = []
        for exact in (True, False):
            if not exact:
                failures = remaining
            for i in failures:
                value = float(population.at[i, 'total_clicks'])
                pos = bisect_left(clicks, value)
                options = [j for j in (pos-1, pos) if 0 <= j < len(clicks)]
                if exact:
                    options = [j for j in options if clicks[j] == value]
                else:
                    options = [j for j in options if (value == clicks[j] == 0) or
                               (min(value, clicks[j]) > 0 and max(value, clicks[j])/min(value, clicks[j]) <= 1+tolerance+1e-12)]
                if not options:
                    if exact:
                        remaining.append(i)
                    continue
                j = min(options, key=lambda j: abs(np.log1p(value)-np.log1p(clicks[j])))
                pi = passing.pop(j)
                other = clicks.pop(j)
                pairs.append({'code_module': module, 'code_presentation': presentation,
                              'fail_index': i, 'pass_index': pi,
                              'fail_student': int(population.at[i,'id_student']),
                              'pass_student': int(population.at[pi,'id_student']),
                              'fail_clicks': value, 'pass_clicks': other,
                              'relative_gap': max(value,other)/min(value,other)-1 if min(value,other)>0 else 0.0})
    return pd.DataFrame(pairs, columns=['code_module','code_presentation','fail_index','pass_index',
                                      'fail_student','pass_student','fail_clicks','pass_clicks','relative_gap'])


def _component_intervals(pairs: pd.DataFrame, differences: np.ndarray, seed: int, repeats: int) -> np.ndarray:
    # Connected pairs sharing any student are resampled together. This preserves
    # student dependence across presentations while retaining both sides of pairs.
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a,b in zip(pairs.fail_student,pairs.pass_student):
        parent[find(a)] = find(b)
    labels = pd.factorize(np.asarray([find(a) for a in pairs.fail_student]))[0]
    n = int(labels.max())+1
    sums = np.zeros((n,differences.shape[1]))
    np.add.at(sums, labels, differences)
    counts = np.bincount(labels)
    rng = np.random.default_rng(seed)
    values = np.empty((repeats,differences.shape[1]))
    for i in range(repeats):
        selected = rng.integers(0,n,n)
        values[i] = sums[selected].sum(axis=0)/counts[selected].sum()
    return np.quantile(values,[.025,.975],axis=0)


def compare_matched(frame: pd.DataFrame, output_dir: Path, seed: int = 42, repeats: int = 1000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population = frame[frame.final_result.isin(['Fail','Pass','Distinction'])].reset_index(drop=True)
    features = ['active_days','longest_inactive_run','daily_concentration','daily_entropy'] + CONCENTRATION_FEATURES + [f'week_{i}_share' for i in range(4)] + sorted(c for c in frame if c.startswith('resource_share__'))
    summaries, effects, local_effects = [], [], []
    for tolerance in (0.0,.05,.10):
        pairs = make_pairs(population,tolerance,seed)
        pairs.to_parquet(output_dir / f'matched_pairs_{int(tolerance*100):02}.parquet',index=False)
        if pairs.empty:
            summaries.append({'tolerance':tolerance,'pairs':0,'population_rows':len(population)})
            continue
        fail = population.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
        passing = population.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
        differences = fail[features].to_numpy()-passing[features].to_numpy()
        interval = _component_intervals(pairs,differences,seed,repeats)
        sd = np.sqrt((fail.log_clicks.var(ddof=0)+passing.log_clicks.var(ddof=0))/2)
        summaries.append({'tolerance':tolerance,'pairs':len(pairs),'retained_rows':2*len(pairs),
                          'population_rows':len(population),'fail_coverage':len(pairs)/max(1,population.final_result.eq('Fail').sum()),
                          'zero_click_pairs':int(pairs.fail_clicks.eq(0).sum()),
                          'max_relative_click_gap':float(pairs.relative_gap.max()),
                          'mean_relative_click_gap':float(pairs.relative_gap.mean()),
                          'log_click_smd':float((fail.log_clicks.mean()-passing.log_clicks.mean())/sd) if sd else 0.0})
        for j,feature in enumerate(features):
            effects.append({'tolerance':tolerance,'feature':feature,'pairs':len(pairs),
                            'fail_mean':float(fail[feature].mean()),'pass_mean':float(passing[feature].mean()),
                            'fail_minus_pass':float(differences[:,j].mean()),'ci_low':float(interval[0,j]),'ci_high':float(interval[1,j])})
        for (module,presentation),indices in pairs.groupby(['code_module','code_presentation']).indices.items():
            for j,feature in enumerate(features):
                local_effects.append({'tolerance':tolerance,'module':module,'presentation':presentation,
                                     'feature':feature,'pairs':len(indices),'fail_minus_pass':float(differences[indices,j].mean())})
    pd.DataFrame(summaries).to_csv(output_dir/'matched_balance.csv',index=False)
    pd.DataFrame(effects).to_csv(output_dir/'matched_effects.csv',index=False)
    pd.DataFrame(local_effects).to_csv(output_dir/'matched_presentation_effects.csv',index=False)
    (output_dir/'matched_protocol.json').write_text(json.dumps({'seed':seed,'bootstrap_repeats':repeats,
        'population':'Pass/Distinction vs Fail; days 0-27; Withdrawn excluded',
        'matching':'same module and presentation; exact first; then greedy nearest available; no reuse',
        'positive_click_ratio_limits':[1,1.05,1.10], 'zero_click_rule':'zero matches zero only',
        'interval':'connected-student-component bootstrap conditional on fixed pairs; pointwise exploratory 95% intervals',
        'limitations':'No causal inference; matched outcome balance is artificial; matching order not optimized; no multiple-testing correction'},indent=2),encoding='utf-8')
