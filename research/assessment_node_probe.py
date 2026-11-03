"""Read-only feasibility probe: first assessment status as of day 27.

No future submission or score is used as an observed day-27 feature.
Transferred assessment credits are excluded at pair level.
"""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oulad_analysis.matched import _component_intervals

OUT=Path(__file__).parent/'assessment_node_probe'
OUT.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'data/oulad.zip') as z:
    assessments=pd.read_csv(z.open('assessments.csv'),na_values=['?'])
    submissions=pd.read_csv(z.open('studentAssessment.csv'),na_values=['?'])
    courses=pd.read_csv(z.open('courses.csv'))
assert not submissions.duplicated(['id_assessment','id_student']).any()
nonexam=assessments[assessments.assessment_type.ne('Exam')]
first=nonexam[nonexam.date.eq(nonexam.groupby(['code_module','code_presentation']).date.transform('min'))].copy()
assert not first.duplicated(['code_module','code_presentation']).any(), 'First deadline has multiple assessments; revise eligibility.'
first.to_csv(OUT/'first_deadlines.csv',index=False)
f=pd.read_parquet(ROOT/'outputs/timing_concentration_20261102/features.parquet')
f=f[f.final_result.isin(['Pass','Distinction','Fail'])].reset_index(drop=True)
f=f.merge(first[['code_module','code_presentation','id_assessment','date']],on=['code_module','code_presentation'],how='left',validate='many_to_one')
f=f.merge(submissions[['id_assessment','id_student','date_submitted','is_banked']],on=['id_assessment','id_student'],how='left',validate='one_to_one')
assert len(f)==22437
f['eligible']=f.date.between(0,27) & ~f.is_banked.eq(1)
f['on_time']=f.date_submitted.le(f.date)
f['late_by27']=f.date_submitted.gt(f.date) & f.date_submitted.le(27)
f['no_submission_by27']=f.date_submitted.isna() | f.date_submitted.gt(27)
cols=['on_time','late_by27','no_submission_by27']
rows=[]
coverage=[]
for caliper in (0,5,10):
    pairs=pd.read_parquet(ROOT/f'outputs/timing_concentration_20261102/matched_pairs_{caliper:02}.parquet')
    a=f.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    b=f.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    assert a.id_student.eq(pairs.fail_student).all() and b.id_student.eq(pairs.pass_student).all()
    keep=a.eligible & b.eligible
    p=pairs.loc[keep].reset_index(drop=True)
    a=a.loc[keep].reset_index(drop=True)
    b=b.loc[keep].reset_index(drop=True)
    diff=a[cols].to_numpy(dtype=float)-b[cols].to_numpy(dtype=float)
    ci=_component_intervals(p,diff,42,1000)
    coverage.append({'caliper_percent':caliper,'original_pairs':len(pairs),'eligible_pairs':len(p),'presentations':len(p.groupby(['code_module','code_presentation']))})
    for j,name in enumerate(cols):
        rows.append({'caliper_percent':caliper,'metric':name,'pairs':len(p),'fail_rate':float(a[name].mean()),'pass_rate':float(b[name].mean()),'fail_minus_pass':float(diff[:,j].mean()),'ci_low':float(ci[0,j]),'ci_high':float(ci[1,j])})
pd.DataFrame(rows).to_csv(OUT/'matched_submission_status.csv',index=False)
summary={'course_duration_range':[int(courses.module_presentation_length.min()),int(courses.module_presentation_length.max())],
         'first_deadline_presentations_by_day27':int(first.date.le(27).sum()),'presentations':len(first),
         'coverage':coverage,'score_used':False,'banked_rule':'exclude pair if either record has transferred first-assessment credit',
         'as_of_day':27,'caution':'Exploratory paired status differences; no causal or independent predictive-gain claim; pointwise CI conditional on fixed matching.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
