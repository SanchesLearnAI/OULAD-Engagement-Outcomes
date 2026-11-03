"""Read-only checks of GitHub snapshot 92416b9; stores aggregate evidence only."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import duckdb
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'audit_20261102'
OUT.mkdir(exist_ok=True)
KEY = ['code_module', 'code_presentation', 'id_student']
published = ROOT / 'results/local_uci_20261102'
local = ROOT / 'outputs/local_uci_corrected'
evidence = {}
models = json.loads((published / 'model_results.json').read_text())
evidence['local_published_models_equal'] = models == json.loads((local / 'model_results.json').read_text())
evidence['archive_sha256'] = hashlib.file_digest((ROOT / 'data/oulad.zip').open('rb'), 'sha256').hexdigest()
with zipfile.ZipFile(ROOT / 'data/oulad.zip') as z:
    evidence['archive_members'] = z.namelist()
    reg_name = next(n for n in z.namelist() if n.endswith('studentRegistration.csv'))
    reg = pd.read_csv(z.open(reg_name), na_values=['?'])
    info_name = next(n for n in z.namelist() if n.endswith('studentInfo.csv'))
    info = pd.read_csv(z.open(info_name))
    vle_name = next(n for n in z.namelist() if n.endswith('vle.csv'))
    vle = pd.read_csv(z.open(vle_name))
frame = pd.read_parquet(local / 'features.parquet')
frame = frame.merge(reg, on=KEY, validate='one_to_one')
evidence['labels_zero'] = frame.groupby('final_result').agg(n=('id_student','size'), zero=('zero_click','sum')).reset_index().to_dict('records')
evidence['registration'] = {
    'missing_registration': int(reg.date_registration.isna().sum()),
    'registered_after_day_0': int((reg.date_registration > 0).sum()),
    'registered_after_day_27': int((reg.date_registration > 27).sum()),
    'withdrawn_before_day_0': int((reg.date_unregistration < 0).sum()),
    'withdrawn_by_day_27': int((reg.date_unregistration <= 27).sum()),
    'zero_click_withdrawn_by_day_27': int(((frame.zero_click == 1) & (frame.date_unregistration <= 27)).sum()),
}
evidence['population'] = {'rows':len(frame), 'students':frame.id_student.nunique(), 'modules':frame.code_module.nunique(), 'module_presentations':len(frame.groupby(KEY[:2])), 'binary_rows':int(frame.final_result.isin(['Pass','Distinction','Fail']).sum())}
con = duckdb.connect()
con.execute("SET memory_limit='3GB'")
path = str(ROOT / 'data/raw/studentVle.csv').replace("'", "''")
con.execute(f"CREATE TEMP TABLE events AS SELECT * FROM read_csv_auto('{path}') WHERE date BETWEEN 0 AND 27")
evidence['raw_window'] = con.execute('SELECT count(*) n, sum(sum_click) clicks, count(DISTINCT (code_module,code_presentation,id_student,id_site,date,sum_click)) distinct_rows FROM events').df().to_dict('records')[0]
evidence['exact_dedup'] = con.execute('SELECT count(*) n, sum(sum_click) clicks FROM (SELECT DISTINCT * FROM events)').df().to_dict('records')[0]
raw = con.execute('SELECT code_module,code_presentation,id_student,sum(sum_click) raw_clicks FROM events GROUP BY 1,2,3').df()
dedup = con.execute('SELECT code_module,code_presentation,id_student,sum(sum_click) dedup_clicks FROM (SELECT DISTINCT * FROM events) GROUP BY 1,2,3').df()
joined = frame.merge(raw,on=KEY,how='left').merge(dedup,on=KEY,how='left').fillna({'raw_clicks':0,'dedup_clicks':0})
evidence['raw_feature_totals_equal'] = bool(np.allclose(joined.total_clicks,joined.raw_clicks))
joined['relative_click_reduction'] = (joined.raw_clicks-joined.dedup_clicks)/joined.raw_clicks.replace(0,np.nan)
evidence['dedup_effect'] = {'affected_rows':int((joined.raw_clicks!=joined.dedup_clicks).sum()),'relative_reduction_quantiles_nonzero':joined.relative_click_reduction.quantile([0,.25,.5,.75,.9,1]).to_dict()}
evidence['dedup_by_module'] = joined.groupby('code_module')[['raw_clicks','dedup_clicks']].sum().reset_index().to_dict('records')
population = frame[frame.final_result.isin(['Pass','Distinction','Fail'])].copy()
population['fail'] = population.final_result.eq('Fail')
balance=[]
for bins in (5,10,20):
    for (module,presentation),g in population.groupby(KEY[:2]):
        edges = np.unique(np.quantile(g.log_clicks,np.linspace(0,1,bins+1)))
        strata = pd.Series(0,index=g.index) if len(edges)<2 else pd.cut(g.log_clicks,edges,labels=False,include_lowest=True)
        for s,v in g.groupby(strata):
            a=v[v.fail].log_clicks; b=v[~v.fail].log_clicks
            if not len(a) or not len(b): continue
            sd=np.sqrt((a.var(ddof=0)+b.var(ddof=0))/2)
            balance.append({'bins':bins,'module':module,'presentation':presentation,'stratum':int(s),'n':len(v),'smd':float((a.mean()-b.mean())/sd) if sd else 0.0,'gap':float(a.mean()-b.mean()),'min_click':float(v.total_clicks.min()),'max_click':float(v.total_clicks.max())})
balance=pd.DataFrame(balance)
balance.to_csv(OUT/'click_balance.csv',index=False)
evidence['click_balance'] = [dict(bins=int(b),retained_rows=int(g.n.sum()),weighted_absolute_smd=float(np.average(g.smd.abs(),weights=g.n)),max_absolute_smd=float(g.smd.abs().max()),strata_above_01=int((g.smd.abs()>.1).sum()),strata=len(g),weighted_log_gap=float(np.average(g.gap,weights=g.n))) for b,g in balance.groupby('bins')]
metrics=pd.read_csv(published/'presentation_metrics.csv')
pivot=metrics.pivot(index=['module','presentation'],columns='model',values='average_precision')
pivot['delta_full_lr']=pivot.full_lr-pivot.baseline_lr
pivot[['baseline_lr','full_lr','delta_full_lr']].to_csv(OUT/'presentation_delta.csv')
evidence['presentation_generalization']={'positive':int((pivot.delta_full_lr>0).sum()),'total':len(pivot),'min':float(pivot.delta_full_lr.min()),'max':float(pivot.delta_full_lr.max()),'mean_unweighted':float(pivot.delta_full_lr.mean()),'bottom':pivot.sort_values('delta_full_lr').head(3)[['baseline_lr','full_lr','delta_full_lr']].reset_index().to_dict('records')}
pred=pd.read_parquet(local/'heldout_predictions.parquet')
y=pred.final_result.eq('Fail').astype(int)
checks=[]
for m in models:
    score=pred['score_'+m['model']]
    ap=average_precision_score(y,score)
    f1=f1_score(y,score>=m['threshold_from_training'])
    checks.append({'model':m['model'],'ap':ap,'f1':f1,'matches_published':bool(np.isclose(ap,m['average_precision']) and np.isclose(f1,m['f1']))})
evidence['metrics_recomputed']=checks
cluster=pd.read_parquet(local/'cluster_sample.parquet')
evidence['cluster_module_composition']=pd.crosstab(cluster.cluster,cluster.code_module,normalize='index').reset_index().to_dict('records')
evidence['cluster_click_medians']=cluster.groupby('cluster').total_clicks.median().to_dict()
evidence['cluster_early_withdrawal']=cluster.merge(reg,on=KEY).assign(early=lambda x:x.date_unregistration<=27).groupby('cluster').agg(n=('id_student','size'),early=('early','sum')).reset_index().to_dict('records')
evidence['resource_availability']=vle.groupby('code_module').activity_type.nunique().to_dict()
evidence['similar_click_selected']=pd.read_csv(published/'similar_click_comparison.csv').query("feature in ['active_days','longest_inactive_run','week_0_share','week_3_share','resource_share__forumng']").to_dict('records')
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)),encoding='utf-8')
print(json.dumps(evidence,indent=2,ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
