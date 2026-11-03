"""Fixed-split E1/E2/E3 ablations; aggregate outputs only."""
from pathlib import Path
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT
OUT=ROOT/'research/experiment_design_20261103'
K=['code_module','code_presentation','id_student']
S=['code_module','code_presentation']
SEEDS=[42,143,144,145,146]

def ref_features(train,test):
    train=train.copy();test=test.copy()
    # Reference distributions are learned from training students only.
    refs={k:np.sort(g.log_clicks.to_numpy()) for k,g in train.groupby('batch')}
    fallback=np.sort(train.log_clicks.to_numpy())
    for part in [train,test]:
        pct=np.empty(len(part));z=np.empty(len(part))
        for b, idx in part.groupby('batch').indices.items():
            r=refs.get(b,fallback);v=part.iloc[idx].log_clicks.to_numpy()
            pct[idx]=np.searchsorted(r,v,side='right')/len(r)
            sd=r.std();z[idx]=(v-r.mean())/sd if sd>0 else 0
        part['relative_percentile']=pct;part['relative_z']=z
    return train,test

def fit_score(a,b,cols,return_predictions=False):
    cat=[x for x in cols if x in ('batch','code_module')];num=[x for x in cols if x not in cat]
    transformer=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
    model=make_pipeline(transformer,LogisticRegression(max_iter=600))
    y=a.final_result.eq('Fail').astype(int)
    model.fit(a[cols],y)
    p=model.predict_proba(b[cols])[:,1]
    yy=b.final_result.eq('Fail').astype(int)
    result={'ap':float(average_precision_score(yy,p)),'brier':float(brier_score_loss(yy,p))}
    return (result,p) if return_predictions else result

def run(frame,sets,experiment):
    frame=frame[frame.final_result.isin(['Pass','Distinction','Fail'])].reset_index(drop=True).copy()
    frame['batch']=frame.code_module+'_'+frame.code_presentation
    frame['log_clicks']=np.log1p(frame.total_clicks)
    frame['log_clicks2']=frame.log_clicks**2
    rows=[]
    for seed in SEEDS:
        ix,jx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(frame,groups=frame.id_student))
        a,b=ref_features(frame.iloc[ix].reset_index(drop=True),frame.iloc[jx].reset_index(drop=True))
        assert set(a.id_student).isdisjoint(set(b.id_student))
        for name,cols in sets.items():
            v=fit_score(a,b,cols)
            rows.append(dict(experiment=experiment,seed=seed,model=name,n_train=len(a),n_test=len(b),fail_rate=b.final_result.eq('Fail').mean(),**v))
    return pd.DataFrame(rows)

def paired_validation(frame,sets,pairs,label):
    frame=frame[frame.final_result.isin(['Pass','Distinction','Fail'])].reset_index(drop=True).copy()
    frame['batch']=frame.code_module+'_'+frame.code_presentation
    frame['log_clicks']=np.log1p(frame.total_clicks);frame['log_clicks2']=frame.log_clicks**2
    ix,jx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(frame,groups=frame.id_student))
    a,b=ref_features(frame.iloc[ix].reset_index(drop=True),frame.iloc[jx].reset_index(drop=True))
    y=b.final_result.eq('Fail').astype(int).to_numpy();groups=b.id_student.to_numpy()
    scores={name:fit_score(a,b,cols,True)[1] for name,cols in sets.items()}
    members=list(b.groupby('id_student').indices.values())
    rng=np.random.default_rng(42342)
    draws=[]
    for _ in range(1000):
        selected=rng.integers(len(members),size=len(members))
        draws.append(np.concatenate([members[i] for i in selected]))
    result=[];per=[]
    for name,control in pairs:
        p,q=scores[name],scores[control]
        delta=average_precision_score(y,p)-average_precision_score(y,q)
        ds=[average_precision_score(y[j],p[j])-average_precision_score(y[j],q[j]) for j in draws if y[j].min()!=y[j].max()]
        result.append(dict(experiment=label,model=name,comparator=control,n_test=len(b),students=b.id_student.nunique(),delta_ap=float(delta),ci_low=float(np.quantile(ds,.025)),ci_high=float(np.quantile(ds,.975))))
        for module,inds in b.groupby('code_module').indices.items():
            if y[inds].min()==y[inds].max():continue
            per.append(dict(experiment=label,model=name,comparator=control,module=module,n=len(inds),delta_ap=float(average_precision_score(y[inds],p[inds])-average_precision_score(y[inds],q[inds]))))
    return result,per

def first_tma():
    with zipfile.ZipFile(REPO/'data/oulad.zip') as z:
        a=pd.read_csv(z.open('assessments.csv'))
        sub=pd.read_csv(z.open('studentAssessment.csv'))
    a['date']=pd.to_numeric(a.date,errors='coerce')
    a=a[a.assessment_type.eq('TMA') & a.weight.gt(0)].sort_values(['date','id_assessment']).drop_duplicates(S)
    assert not a.duplicated(S).any()
    return a[S+['id_assessment','date']],sub[['id_assessment','id_student','date_submitted','is_banked']]

def node_features(base):
    assessment,submission=first_tma()
    f=base.merge(assessment,on=S,validate='many_to_one').merge(submission,on=['id_assessment','id_student'],how='left',validate='one_to_one')
    f=f[~f.is_banked.eq(1)].copy()
    # Day-27 fixed cutoff. First TMA deadlines after 27 are explicitly 'not yet due'.
    f['not_due27']=f.date.gt(27).astype(int)
    f['submitted27']=f.date_submitted.le(27).astype(int)
    f['on_time27']=(f.date.le(27)&f.date_submitted.le(f.date)).astype(int)
    f['late27']=(f.date.le(27)&f.date_submitted.gt(f.date)&f.date_submitted.le(27)).astype(int)
    f['no_submit_due27']=(f.date.le(27)&~f.date_submitted.le(27)).astype(int)
    f['days_to_deadline27']=np.maximum(f.date-27,0)
    f['eligible_cutoff']=f.date_registration.le(27)&(f.date_unregistration.isna()|f.date_unregistration.gt(27))
    # Day-grain event sums; source duplicates are retained as in the project feature table.
    pieces=[]
    lookup=f[K+['date']].rename(columns={'date':'due'})
    for chunk in pd.read_csv(REPO/'data/raw/studentVle.csv',usecols=K+['date','sum_click'],chunksize=500000):
        part=chunk.merge(lookup,on=K,validate='many_to_one')
        part['relative']=part.date-part.due
        part=part[part.relative.between(-13,14) | part.date.between(0,27)]
        if len(part):pieces.append(part)
    events=pd.concat(pieces,ignore_index=True)
    events=events.drop(columns='due')
    specs={'pre14':(-13,0),'prelast3':(-2,0),'post14':(1,14),'postweek1':(1,7),'postweek2':(8,14)}
    for name,(lo,hi) in specs.items():
        for suffix,cut in [('',events),('_27',events[events.date.le(27)])]:
            g=cut[cut.relative.between(lo,hi)]
            click=g.groupby(K).sum_click.sum().rename(name+'_clicks'+suffix)
            active=g[g.sum_click.gt(0)].groupby(K).date.nunique().rename(name+'_active'+suffix)
            f=f.join(click,on=K).join(active,on=K)
            f[[name+'_clicks'+suffix,name+'_active'+suffix]]=f[[name+'_clicks'+suffix,name+'_active'+suffix]].fillna(0)
    f['prelast3_share']=(f.prelast3_clicks/f.pre14_clicks.replace(0,np.nan)).fillna(0)
    f['post_both_weeks']=(f.postweek1_active.gt(0)&f.postweek2_active.gt(0)).astype(int)
    f['post_vs_pre_log']=np.log1p(f.post14_clicks)-np.log1p(f.pre14_clicks)
    f['prelast3_share_27']=(f.prelast3_clicks_27/f.pre14_clicks_27.replace(0,np.nan)).fillna(0)
    f['post_both_weeks_27']=(f.postweek1_active_27.gt(0)&f.postweek2_active_27.gt(0)).astype(int)
    f['post_vs_pre_log_27']=np.log1p(f.post14_clicks_27)-np.log1p(f.pre14_clicks_27)
    # d+14 version: each student's cutoff differs, and only post-cutoff withdrawals are outcomes.
    f['eligible_node14']=f.date_registration.le(f.date)&(f.date_unregistration.isna()|f.date_unregistration.gt(f.date+14))
    f['submitted_node14']=f.date_submitted.le(f.date+14).astype(int)
    f['late_node14']=(f.date_submitted.gt(f.date)&f.date_submitted.le(f.date+14)).astype(int)
    return f

def main():
    OUT.mkdir(exist_ok=True)
    base=pd.read_parquet(REPO/'outputs/timing_concentration_20261102/features.parquet')
    functions={
        'content':['oucontent','resource','page','glossary'],
        'practice':['quiz','externalquiz'],
        'collaboration':['forumng','ouwiki','oucollaborate','ouelluminate'],
        'navigation':['homepage','subpage','sharedsubpage','folder'],
        'other':['url','htmlactivity','dataplus','dualpane','questionnaire','repeatactivity']}
    for group,types in functions.items():
        base['function_'+group]=base[[f'resource_share__{t}' for t in types if f'resource_share__{t}' in base]].sum(axis=1)
    t=['active_days','longest_inactive_run','daily_concentration','daily_entropy','week_0_share','week_1_share','week_2_share']
    rs=sorted(x for x in base if x.startswith('resource_share__'))[:-1]
    r=['resource_type_count','resource_entropy']+rs
    core=['log_clicks','log_clicks2','batch'];rel=['relative_percentile','relative_z']
    e1=run(base,{'A0':core,'A1':core+rel,'A2':core+t+r,'A3':core+t+r+rel},'E1')
    e1.to_csv(OUT/'e1_models.csv',index=False)
    node=node_features(base)
    fixed=node[node.eligible_cutoff].copy()
    nd=['not_due27','submitted27','on_time27','late27','no_submit_due27']
    nt=['pre14_active_27','prelast3_share_27','post14_active_27','post_both_weeks_27','post_vs_pre_log_27']
    e2fixed=run(fixed,{'D0':core+t+r,'D1':core+t+r+nd,'D2':core+t+r+nt,'D3':core+t+r+nd+nt},'E2_day27')
    e2fixed.to_csv(OUT/'e2_day27_models.csv',index=False)
    # Node cutoff comparison uses the same cohort and same observed d-13..d+14 span.
    nf=node[node.eligible_node14].copy()
    nf['total_clicks']=nf.pre14_clicks+nf.post14_clicks
    nbase=['log_clicks','log_clicks2','batch','pre14_clicks','post14_clicks']
    sub=['submitted_node14','late_node14'];track=['pre14_active','post14_active','prelast3_share','post_both_weeks','post_vs_pre_log']
    e2node=run(nf,{'B0':nbase,'B1':nbase+track,'B2':nbase+sub,'B3':nbase+sub+track},'E2_node14')
    e2node.to_csv(OUT/'e2_node14_models.csv',index=False)
    e3=run(base,{'C0':core+t,'C1':core+t+rs,'C2':core+t+['resource_type_count','resource_entropy'],'C3':core+t+r,
                 'C4':core+t+[f'function_{k}' for k in list(functions)[:-1]]},'E3')
    e3.to_csv(OUT/'e3_models.csv',index=False)
    checks=[];by_module=[]
    validation=[
        (base,{'A2':core+t+r,'A3':core+t+r+rel},[('A3','A2')],'E1'),
        (fixed,{'D0':core+t+r,'D1':core+t+r+nd,'D3':core+t+r+nd+nt},[('D1','D0'),('D3','D1')],'E2_day27'),
        (nf,{'B0':nbase,'B1':nbase+track,'B2':nbase+sub,'B3':nbase+sub+track},[('B1','B0'),('B3','B2')],'E2_node14'),
        (base,{'C1':core+t+rs,'C3':core+t+r},[('C3','C1')],'E3')]
    for frame,models,pairs,label in validation:
        rows,modules=paired_validation(frame,models,pairs,label)
        checks.extend(rows);by_module.extend(modules)
    pd.DataFrame(checks).to_csv(OUT/'paired_ap_intervals.csv',index=False)
    pd.DataFrame(by_module).to_csv(OUT/'module_ap_deltas.csv',index=False)
    transfer=[]
    population=nf[nf.final_result.isin(['Pass','Distinction','Fail'])].copy()
    population['log_clicks']=np.log1p(population.total_clicks)
    population['log_clicks2']=population.log_clicks**2
    train=population[population.code_presentation.str.startswith('2013')]
    test=population[population.code_presentation.str.startswith('2014') & ~population.id_student.isin(train.id_student)]
    for cohort,part in [('known_courses',test[test.code_module.isin(train.code_module)]),('new_CCC',test[test.code_module.eq('CCC')])]:
        if len(part)==0:continue
        for model,cols in {'B2':['code_module','log_clicks','log_clicks2','pre14_clicks','post14_clicks']+sub,
                           'B3':['code_module','log_clicks','log_clicks2','pre14_clicks','post14_clicks']+sub+track}.items():
            transfer.append(dict(cohort=cohort,model=model,n_train=len(train),n_test=len(part),fail_rate=part.final_result.eq('Fail').mean(),**fit_score(train,part,cols)))
    pd.DataFrame(transfer).to_csv(OUT/'e2_year_transfer.csv',index=False)
    for name,table,pairs in [('E1',e1,[('A1','A0'),('A3','A2')]),('E2_day27',e2fixed,[('D1','D0'),('D2','D0'),('D3','D1')]),('E2_node14',e2node,[('B1','B0'),('B2','B0'),('B3','B2')]),('E3',e3,[('C2','C0'),('C3','C1'),('C3','C0'),('C4','C0')])]:
        w=table.pivot(index='seed',columns='model',values='ap')
        for a,b in pairs:
            print(name,a+'-'+b,'AP deltas',np.round((w[a]-w[b]).to_numpy(),5).tolist())
    (OUT/'run_metadata.json').write_text(json.dumps({'seeds':SEEDS,'base_rows':len(base),'day27_node_rows':len(fixed),'node14_rows':len(nf),
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'features_sha256':hashlib.sha256((REPO/'outputs/timing_concentration_20261102/features.parquet').read_bytes()).hexdigest(),
       'rule':'Student-group disjoint 80/20; fixed logistic regularization; no hyperparameter tuning; train-only course reference ranks; all model comparisons within same experiment share population and split. E3 uses existing original 28-day features.'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
