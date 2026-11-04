"""Retrospective nonlinear grade-by-engagement response attributes at TMA +14.

Feature definitions never use the final outcome. Mark release time is unknown.
The prespecified blocks below are evaluated once, including a main-effects
control so nonlinear mark effects are not misreported as interaction gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import _fit_predict
from .tma_submission_timing import add_submission_distance, VERSIONS as PREVIOUS

BASE = PREVIOUS['S0_score_add_submission_distance']
MAIN = ['mark_squared', 'click_rise', 'click_drop', 'active_rise', 'active_drop']
DIRECTION = ['high_mark_click_drop', 'low_mark_click_rise']
PERSISTENCE = ['high_mark_active_drop', 'low_mark_active_rise',
               'high_mark_drop_weak_week2', 'low_mark_rise_sustained_week2']
VERSIONS = {'R0_existing': BASE, 'R1_main_effects': BASE + MAIN,
            'R2_direction': BASE + MAIN + DIRECTION,
            'R3_persistence': BASE + MAIN + DIRECTION + PERSISTENCE}


def attributes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mark = out.first_tma_score / 100
    if not mark.dropna().between(0, 1).all():
        raise ValueError('Mark out of range')
    gap = out.relative_click_gap
    active_gap = (out.post_active_days - out.pre_active_days) / 14
    out['mark_squared'] = mark ** 2
    out['click_rise'] = gap.clip(lower=0)
    out['click_drop'] = (-gap).clip(lower=0)
    out['active_rise'] = active_gap.clip(lower=0)
    out['active_drop'] = (-active_gap).clip(lower=0)
    out['high_mark_click_drop'] = mark ** 2 * out.click_drop
    out['low_mark_click_rise'] = (1 - mark) ** 2 * out.click_rise
    out['high_mark_active_drop'] = mark ** 2 * out.active_drop
    out['low_mark_active_rise'] = (1 - mark) ** 2 * out.active_rise
    out['high_mark_drop_weak_week2'] = out.high_mark_click_drop * (1 - out.post_week2_share)
    out['low_mark_rise_sustained_week2'] = out.low_mark_click_rise * out.post_week2_share
    return out


def evaluate(frame: pd.DataFrame, bootstrap: int = 1000):
    rows, intervals = [], []
    for seed in [42, 143, 144, 145, 146]:
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        assert set(train.id_student).isdisjoint(test.id_student)
        y = test.final_result.eq('Fail').astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, cols) for name, cols in VERSIONS.items()}
        for name, p in predictions.items():
            rows.append(dict(seed=seed, model=name, train_rows=len(train), test_rows=len(test),
                             average_precision=average_precision_score(y, p), brier=brier_score_loss(y, p)))
        if seed != 42:
            continue
        members = list(test.groupby('id_student').indices.values())
        rng = np.random.default_rng(32242)
        draws = [np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for name, comparator in [('R1_main_effects', 'R0_existing'), ('R2_direction', 'R1_main_effects'),
                                 ('R3_persistence', 'R2_direction'), ('R3_persistence', 'R1_main_effects'),
                                 ('R3_persistence', 'R0_existing')]:
            p, q = predictions[name], predictions[comparator]
            differences = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                           for ix in draws if len(np.unique(y[ix])) == 2]
            intervals.append(dict(model=name, comparator=comparator,
                                  delta_ap=average_precision_score(y, p)-average_precision_score(y, q),
                                  ci_low=np.quantile(differences, .025), ci_high=np.quantile(differences, .975)))
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def transfer(frame):
    train = frame[frame.code_presentation.str.startswith('2013')]
    test = frame[frame.code_presentation.str.startswith('2014') &
                 ~frame.id_student.isin(train.id_student) & frame.code_module.isin(train.code_module)]
    rows = []
    for name, columns in VERSIONS.items():
        columns = ['code_module' if col == 'module_presentation' else col for col in columns]
        p = _fit_predict(train, test, columns)
        y = test.final_result.eq('Fail').astype(int)
        rows.append(dict(model=name, train_rows=len(train), test_rows=len(test),
                         fail_prevalence=y.mean(), average_precision=average_precision_score(y, p),
                         brier=brier_score_loss(y, p)))
    return pd.DataFrame(rows)


def describe(frame):
    # Fixed illustrative thresholds, not tuned to outcome or used in models.
    scored = frame[frame.first_tma_score.notna()].copy()
    scored['grade_band'] = np.select([scored.first_tma_score.ge(80), scored.first_tma_score.lt(60)],
                                     ['high_80plus', 'low_below60'], default='middle_60to79')
    scored['change_band'] = np.select([scored.pre_clicks.eq(0) & scored.post_clicks.eq(0),
                                      scored.relative_click_gap.le(-.5), scored.relative_click_gap.ge(.5)],
                                     ['both_zero', 'post_at_most_one_third', 'post_at_least_triple'],
                                     default='moderate_change')
    result = scored.groupby(['grade_band', 'change_band']).agg(
        records=('id_student', 'size'), fail_records=('final_result', lambda x: x.eq('Fail').sum()),
        mean_pre_clicks=('pre_clicks', 'mean'), mean_post_clicks=('post_clicks', 'mean'),
        mean_post_active_days=('post_active_days', 'mean'), late_fraction=('submitted_after_due', 'mean'))
    result['fail_fraction'] = result.fail_records / result.records
    return result[result.records >= 30].reset_index()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(data_dir, node_dir, two_week_dir, output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    node_path, feature_path = node_dir/'node_features.parquet', two_week_dir/'features.parquet'
    protocol = json.loads((node_dir/'node_protocol.json').read_text(encoding='utf-8'))
    upstream = json.loads((two_week_dir/'protocol.json').read_text(encoding='utf-8'))
    assert upstream['node_features_sha256'] == sha(node_path)
    for name in ['assessments', 'studentAssessment']:
        assert protocol['sources_sha256'][name] == sha(data_dir/f'{name}.csv')
    frame = attributes(add_submission_distance(pd.read_parquet(feature_path), pd.read_parquet(node_path), data_dir))
    models, intervals = evaluate(frame)
    reference_path = Path('results/tma_submission_timing_20261103/models.csv')
    reference = pd.read_csv(reference_path).query("model == 'S0_score_add_submission_distance'").sort_values('seed')
    current = models.query("model == 'R0_existing'").sort_values('seed')
    assert np.allclose(current.average_precision, reference.average_precision, atol=1e-12, rtol=0)
    assert np.allclose(current.brier, reference.brier, atol=1e-12, rtol=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {'models': models, 'intervals': intervals, 'year_transfer': transfer(frame),
                        'illustrative_groups': describe(frame)}.items():
        table.to_csv(output_dir/f'{name}.csv', index=False)
    record = dict(models=VERSIONS, grade_weight='high=(score/100)^2; low=(1-score/100)^2',
                  change='signed symmetric (post-pre)/(post+pre); both zero -> 0; split into rise and drop',
                  persistence='active-day change /14; second-week fraction of post clicks',
                  missing='score-dependent attributes missing if score unavailable; training-only median imputation',
                  cohort=len(frame), fail_records=int(frame.final_result.eq('Fail').sum()),
                  code_sha256=sha(__file__), input_sha256={str(p): sha(p) for p in
                    [node_path, feature_path, data_dir/'assessments.csv', data_dir/'studentAssessment.csv',reference_path]},
                  limitations='retrospective marks with unknown release date; selected non-withdrawn cohort; repeated test use; clicks do not measure effort; no causal claims')
    (output_dir/'protocol.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['data-dir', 'node-dir', 'two-week-dir', 'output-dir']:
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.node_dir, args.two_week_dir, args.output_dir)
