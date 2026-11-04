"""Submission timing x post-TMA persistence within fixed click-volume clusters.

Outcome-defined pairs support association, not failure-rate estimation. Feature
blocks are fixed before inspecting results; compare with additive baselines.
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
from .assessment_node import KEY, _fit_predict
from .matched import _component_intervals
from .tma_submission_timing import add_submission_distance, VERSIONS as BASES
from .tma_score_probe import make_joint_pairs
from .tma_click_clusters import _same_cluster_pairs

COMBOS = ['late_days_x_inactive_run', 'late_both_weeks', 'ontime_both_weeks',
          'unsubmitted_x_inactive_run', 'early_days_x_active_days', 'late_x_week2_share']
COMPONENTS = ['submitted_by_cutoff', 'submitted_after_due', 'post_active_days',
              'post_both_weeks', 'post_longest_inactive_run', 'post_week2_share']
MODELS = {'B0_additive': BASES['M1_add_submission_distance'],
          'B1_combinations': BASES['M1_add_submission_distance'] + COMBOS,
          'S0_additive_score': BASES['S0_score_add_submission_distance'],
          'S1_combinations_score': BASES['S0_score_add_submission_distance'] + COMBOS}
MODELS.update({f'S_single_{col}': BASES['S0_score_add_submission_distance'] + [col] for col in COMBOS})


def attributes(frame):
    out = frame.copy()
    late = out.submitted_after_due
    ontime = out.submitted_by_cutoff - late
    no_submission = 1 - out.submitted_by_cutoff
    out['submission_status'] = np.select([no_submission.eq(1), late.eq(1)],
                                         ['not_submitted', 'late'], default='on_time')
    out['late_days_x_inactive_run'] = out.days_late.fillna(0) / 14 * out.post_longest_inactive_run / 14
    out['late_both_weeks'] = late * out.post_both_weeks
    out['ontime_both_weeks'] = ontime * out.post_both_weeks
    out['unsubmitted_x_inactive_run'] = no_submission * out.post_longest_inactive_run / 14
    out['early_days_x_active_days'] = out.days_early.fillna(0) / 14 * out.post_active_days / 14
    out['late_x_week2_share'] = late * out.post_week2_share
    return out


def paired_tables(frame, pairs, family, repeats=1000):
    a = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    b = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    assert a.final_result.eq('Fail').all() and b.final_result.ne('Fail').all()
    assert a[KEY[:2]].equals(b[KEY[:2]])
    sm = np.minimum(a.pre_clicks.to_numpy(), b.pre_clicks.to_numpy())
    lg = np.maximum(a.pre_clicks.to_numpy(), b.pre_clicks.to_numpy())
    pre20 = ((sm == 0) & (lg == 0)) | ((sm > 0) & (lg / np.maximum(sm, 1) <= 1.2 + 1e-12))
    same = a.volume_cluster.eq(b.volume_cluster).to_numpy()
    same_status = a.submission_status.eq(b.submission_status).to_numpy()
    rows, balances = [], []
    for label, name in [(-1, 'zero'), (0, 'low'), (1, 'high')]:
        for subset, keep in [('all', same), ('pre20', same & pre20),
                             ('pre20_same_status', same & pre20 & same_status)]:
            keep = keep & a.volume_cluster.eq(label).to_numpy()
            q = pairs.loc[keep].reset_index(drop=True)
            if len(q) < 10:
                continue
            x, y = a.loc[keep], b.loc[keep]
            cols = COMBOS + COMPONENTS
            difference = x[cols].to_numpy(float) - y[cols].to_numpy(float)
            ci = _component_intervals(q, difference, 42, repeats)
            pooled = np.sqrt((x[cols].var(ddof=0).to_numpy() + y[cols].var(ddof=0).to_numpy()) / 2)
            balances.append(dict(family=family, cluster=name, subset=subset, pairs=len(q),
                fail_pre_clicks=x.pre_clicks.mean(), pass_pre_clicks=y.pre_clicks.mean(),
                fail_post_clicks=x.post_clicks.mean(), pass_post_clicks=y.post_clicks.mean(),
                fail_score=x.first_tma_score.mean(), pass_score=y.first_tma_score.mean(),
                same_status_fraction=x.submission_status.eq(y.submission_status).mean()))
            for j, col in enumerate(cols):
                rows.append(dict(family=family, cluster=name, subset=subset, feature=col, pairs=len(q),
                    fail_mean=x[col].mean(), pass_mean=y[col].mean(), fail_minus_pass=difference[:,j].mean(),
                    ci_low=ci[0,j], ci_high=ci[1,j],
                    pooled_smd=difference[:,j].mean()/pooled[j] if pooled[j] else 0,
                    fail_greater_fraction=(difference[:,j]>0).mean(),
                    tie_fraction=(difference[:,j]==0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(balances)


def evaluate(frame):
    rows, intervals = [], []
    for seed in [42,143,144,145,146]:
        left,right = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(frame,groups=frame.id_student))
        train,test = frame.iloc[left],frame.iloc[right]
        assert set(train.id_student).isdisjoint(test.id_student)
        y = test.final_result.eq('Fail').astype(int).to_numpy()
        predictions = {name:_fit_predict(train,test,cols) for name,cols in MODELS.items()}
        for label, group in [('all',np.ones(len(test),bool)),('low',test.volume_cluster.eq(0).to_numpy()),
                             ('high',test.volume_cluster.eq(1).to_numpy())]:
            for name,p in predictions.items():
                rows.append(dict(seed=seed,cluster=label,model=name,records=int(group.sum()),
                    fail_prevalence=y[group].mean(),average_precision=average_precision_score(y[group],p[group]),
                    brier=brier_score_loss(y[group],p[group])))
            if seed != 42:
                continue
            selected = test.iloc[np.flatnonzero(group)]
            members=list(selected.groupby('id_student').indices.values())
            rng=np.random.default_rng(36442)
            draws=[np.concatenate([members[j] for j in rng.integers(len(members),size=len(members))]) for _ in range(500)]
            for name,base in [('B1_combinations','B0_additive'),('S1_combinations_score','S0_additive_score'),
                              ('S_single_unsubmitted_x_inactive_run','S0_additive_score'),
                              ('S1_combinations_score','S_single_unsubmitted_x_inactive_run')]:
                p,q,truth=predictions[name][group],predictions[base][group],y[group]
                diff=[average_precision_score(truth[ix],p[ix])-average_precision_score(truth[ix],q[ix])
                      for ix in draws if len(np.unique(truth[ix]))==2]
                intervals.append(dict(cluster=label,model=name,comparator=base,
                    delta_ap=average_precision_score(truth,p)-average_precision_score(truth,q),
                    ci_low=np.quantile(diff,.025),ci_high=np.quantile(diff,.975)))
    return pd.DataFrame(rows),pd.DataFrame(intervals)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer(frame):
    train=frame[frame.code_presentation.str.startswith('2013')]
    test=frame[frame.code_presentation.str.startswith('2014') & ~frame.id_student.isin(train.id_student)
               & frame.code_module.isin(train.code_module)]
    rows=[]
    for name,columns in MODELS.items():
        columns=['code_module' if c=='module_presentation' else c for c in columns]
        p=_fit_predict(train,test,columns)
        y=test.final_result.eq('Fail').astype(int)
        rows.append(dict(model=name,train_rows=len(train),test_rows=len(test),
                         average_precision=average_precision_score(y,p),brier=brier_score_loss(y,p)))
    return pd.DataFrame(rows)


def run(data_dir,node_dir,two_week_dir,cluster_dir,output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    node_path=node_dir/'node_features.parquet'; feature_path=two_week_dir/'features.parquet'
    p=json.loads((two_week_dir/'protocol.json').read_text(encoding='utf-8'))
    cp=json.loads((cluster_dir/'volume_cluster_protocol.json').read_text(encoding='utf-8'))
    np_=json.loads((node_dir/'node_protocol.json').read_text(encoding='utf-8'))
    assert p['node_features_sha256']==sha(node_path)
    assert cp['input_sha256']['features.parquet']==sha(feature_path)
    for name in ['assessments','studentAssessment']:
        assert np_['sources_sha256'][name]==sha(data_dir/f'{name}.csv')
    frame=attributes(add_submission_distance(pd.read_parquet(feature_path),pd.read_parquet(node_path),data_dir))
    labels=pd.read_parquet(cluster_dir/'sample_volume_clusters.parquet')
    frame=frame.merge(labels,on=KEY,how='left',validate='one_to_one',sort=False)
    assert frame.volume_cluster.notna().all()
    pairs=pd.read_parquet(two_week_dir/'post_pairs_05.parquet')
    # Validate identity and clicks before using positional pair indices.
    _same_cluster_pairs(frame,pairs,frame.volume_cluster.to_numpy())
    effects,balance=paired_tables(frame,pairs,'post_click_matched')
    scored=frame[frame.first_tma_score.notna()].reset_index(drop=True).assign(total_clicks=lambda x:x.post_clicks)
    joint=make_joint_pairs(scored,score_gap=5,click_tolerance=.05,seed=42)
    effects2,balance2=paired_tables(scored,joint,'score_post_click_matched')
    models,intervals=evaluate(frame)
    ref_path=Path('results/tma_submission_timing_20261103/models.csv')
    ref=pd.read_csv(ref_path)
    for new,old in [('B0_additive','M1_add_submission_distance'),('S0_additive_score','S0_score_add_submission_distance')]:
        a=models[(models.cluster=='all')&(models.model==new)].sort_values('seed')
        b=ref[ref.model==old].sort_values('seed')
        assert np.allclose(a.average_precision,b.average_precision,atol=1e-12,rtol=0)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in {'effects':pd.concat([effects,effects2]),'balance':pd.concat([balance,balance2]),
                       'models':models,'intervals':intervals,'year_transfer':transfer(frame)}.items():
        table.to_csv(output_dir/f'{name}.csv',index=False)
    sources=[node_path,feature_path,two_week_dir/'post_pairs_05.parquet',cluster_dir/'sample_volume_clusters.parquet',
             data_dir/'assessments.csv',data_dir/'studentAssessment.csv',ref_path]
    (output_dir/'protocol.json').write_text(json.dumps(dict(models=MODELS,combinations=COMBOS,
        input_sha256={str(x):sha(x) for x in sources},code_sha256=sha(__file__),
        cluster_model='fixed existing outcome-blind volume clusters; pooled model evaluated within cluster, no separate cluster training',
        pair_intervals='1000 fixed-pair connected-student bootstrap repeats, pointwise, no multiplicity adjustment',
        limitations='exploratory; repeated tests; selected non-withdrawn cohort; personal extension and grade release unknown; course-wide clicks; not causal'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['data-dir','node-dir','two-week-dir','cluster-dir','output-dir']:
        parser.add_argument('--'+arg,type=Path,required=True)
    a=parser.parse_args()
    run(a.data_dir,a.node_dir,a.two_week_dir,a.cluster_dir,a.output_dir)
