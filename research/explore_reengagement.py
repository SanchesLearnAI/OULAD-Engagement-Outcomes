"""Read-only exploration of activity after the first weighted TMA deadline.

Writes aggregate tables only. Run from any directory with Python/pandas.
"""
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/reframe_20261103'
ZIP = ROOT / 'data/oulad.zip'
KEY = ['code_module', 'code_presentation', 'id_student']
with zipfile.ZipFile(ZIP) as z:
    info = pd.read_csv(z.open('studentInfo.csv'), na_values=['?'])
    reg = pd.read_csv(z.open('studentRegistration.csv'), na_values=['?'])
    assessments = pd.read_csv(z.open('assessments.csv'), na_values=['?'])
    submitted = pd.read_csv(z.open('studentAssessment.csv'), na_values=['?'])
    first = assessments[assessments.assessment_type.eq('TMA') & assessments.weight.gt(0)].copy()
    first = first[first.date.eq(first.groupby(KEY[:2]).date.transform('min'))]
    assert not first.duplicated(KEY[:2]).any()
    c = info.merge(reg, on=KEY,validate='one_to_one').merge(first[KEY[:2]+['id_assessment','date']],on=KEY[:2],validate='many_to_one')
    c = c.merge(submitted[['id_assessment','id_student','date_submitted','is_banked']],on=['id_assessment','id_student'],how='left',validate='one_to_one')
    c = c[c.date_registration.le(c.date) & (c.date_unregistration.isna() | c.date_unregistration.gt(c.date+14)) & ~c.is_banked.eq(1) & (c.date_submitted.isna() | c.date_submitted.gt(c.date))].copy()
    assert len(c)==5126
    c['outcome'] = c.final_result.replace({'Pass':'Passed','Distinction':'Passed','Fail':'Failed','Withdrawn':'Withdrew'})
    c['submitted14'] = c.date_submitted.le(c.date+14)
    # Collapse repeated source events at day grain. Keep their click contributions.
    parts=[]
    lookup=c[KEY+['date']].rename(columns={'date':'deadline'})
    for chunk in pd.read_csv(z.open('studentVle.csv'),usecols=KEY+['date','sum_click'],chunksize=500000):
        m=chunk.merge(lookup,on=KEY,how='inner',validate='many_to_one')
        m['relative_day']=m.date-m.deadline
        m=m[m.relative_day.between(-13,14)]
        if len(m):parts.append(m.groupby(KEY+['relative_day'],as_index=False).sum_click.sum())
daily=pd.concat(parts).groupby(KEY+['relative_day'],as_index=False).sum_click.sum()
assert daily.sum_click.ge(0).all()
post=daily[daily.relative_day.between(1,14) & daily.sum_click.gt(0)].copy()
post['half']=np.where(post.relative_day.le(7),'week1','week2')
summary=post.groupby(KEY).agg(clicks14=('sum_click','sum'),active_days14=('relative_day','nunique'))
half=post.groupby(KEY+['half']).relative_day.nunique().unstack(fill_value=0)
summary=summary.join(half).reset_index()
c=c.merge(summary,on=KEY,how='left',validate='one_to_one')
for col in ['clicks14','active_days14','week1','week2']:c[col]=c[col].fillna(0).astype(int)
c['active_both_weeks']=c.week1.gt(0) & c.week2.gt(0)
c['active_7plus_days']=c.active_days14.ge(7)
pre=daily[daily.relative_day.between(-13,0)].groupby(KEY).sum_click.sum().rename('clicks_pre')
c=c.join(pre,on=KEY)
c['clicks_pre']=c.clicks_pre.fillna(0).astype(int)
c['post_click_band']=pd.cut(c.clicks14,[-1,0,20,100,np.inf],labels=['0','1-20','21-100','101+'])
def outcomes(df,cols):
    t=df.groupby(cols+['outcome'],observed=True).size().unstack(fill_value=0)
    for name in ['Passed','Failed','Withdrew']:
        if name not in t:t[name]=0
    t=t[['Passed','Failed','Withdrew']]
    t['n']=t.sum(axis=1)
    for name in ['Passed','Failed','Withdrew']:t[name+'_pct']=(100*t[name]/t.n).round(1)
    return t.reset_index()
outcomes(c,['submitted14','active_both_weeks']).to_csv(OUT/'explore_joint_outcomes.csv',index=False)
outcomes(c,['submitted14','post_click_band','active_both_weeks']).to_csv(OUT/'explore_click_strata.csv',index=False)
outcomes(c,['submitted14','active_7plus_days']).to_csv(OUT/'explore_active_days.csv',index=False)
outcomes(c,['code_module','code_presentation','submitted14','active_both_weeks']).to_csv(OUT/'explore_by_presentation.csv',index=False)
print('cohort',len(c),'students',c.id_student.nunique(),'daily_rows',len(daily))
print(outcomes(c,['submitted14','active_both_weeks']).to_string(index=False))
print(outcomes(c,['submitted14','post_click_band','active_both_weeks']).to_string(index=False))
