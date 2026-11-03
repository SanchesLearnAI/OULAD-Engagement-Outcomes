"""Aggregate-only OULAD feasibility checks for a revised research question."""
from pathlib import Path
import json
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'reframe_20261103'
OUT.mkdir(exist_ok=True)
KEY = ['code_module', 'code_presentation']
with zipfile.ZipFile(ROOT / 'data/oulad.zip') as z:
    frames = {name: pd.read_csv(z.open(name + '.csv'), na_values=['?']) for name in
              ['courses', 'studentInfo', 'studentRegistration', 'assessments', 'studentAssessment', 'vle']}
info, reg, a, sa, vle = [frames[n] for n in ['studentInfo', 'studentRegistration', 'assessments', 'studentAssessment', 'vle']]
assert not info.duplicated(KEY + ['id_student']).any()
assert not sa.duplicated(['id_assessment', 'id_student']).any()
base = info.merge(reg, on=KEY + ['id_student'], validate='one_to_one')
nonexam = a[a.assessment_type.ne('Exam')].copy()
summary = {'rows': len(info), 'unique_students': int(info.id_student.nunique()),
    'outcomes': info.final_result.value_counts().to_dict(),
    'registration_missing': int(reg.date_registration.isna().sum()),
    'vle_week_from_missing': int(vle.week_from.isna().sum()),
    'vle_week_to_missing': int(vle.week_to.isna().sum()), 'vle_rows': len(vle),
    'nonexam_missing_date': int(nonexam.date.isna().sum()),
    'assessment_scores_missing': int(sa.score.isna().sum()),
    'banked_submission_rows': int(sa.is_banked.eq(1).sum()),
    'repeated_students': int(info.groupby('id_student').size().gt(1).sum()),
    'withdrawal_dates': {'before_start': int(reg.date_unregistration.lt(0).sum()),
        'through_day27': int(reg.date_unregistration.le(27).sum())}}
course = info.groupby(KEY).size().rename('enrollments').to_frame()
course = course.join(nonexam.groupby(KEY).agg(nonexam_count=('id_assessment','size'),
        positive_weight_count=('weight', lambda x: int(x.gt(0).sum())),
        first_deadline=('date','min')))
course = course.join(vle.groupby(KEY).activity_type.nunique().rename('resource_type_count'))
course = course.join(nonexam[nonexam.weight.gt(0)].groupby(KEY).date.min().rename('first_positive_deadline'))
course.to_csv(OUT / 'course_context.csv')
records = []
for definition, candidates in [('first_nonexam', nonexam), ('first_positive_weight', nonexam[nonexam.weight.gt(0)]), ('first_positive_TMA', nonexam[nonexam.weight.gt(0) & nonexam.assessment_type.eq('TMA')])]:
    first = candidates[candidates.date.eq(candidates.groupby(KEY).date.transform('min'))].copy()
    assert not first.duplicated(KEY).any(), 'Tied first assessments require joint status definition'
    f = base.merge(first[KEY + ['id_assessment','date','weight']], on=KEY, validate='many_to_one')
    f = f.merge(sa[['id_assessment','id_student','date_submitted','is_banked']], on=['id_assessment','id_student'], how='left',validate='one_to_one')
    # Status assessed at end of deadline day; exclude already withdrawn and transferred credit.
    f = f[f.date_registration.le(f.date) & (f.date_unregistration.isna() | f.date_unregistration.gt(f.date)) & ~f.is_banked.eq(1)].copy()
    f['missed'] = f.date_submitted.isna() | f.date_submitted.gt(f.date)
    f['eligible14'] = f.date_unregistration.isna() | f.date_unregistration.gt(f.date + 14)
    f['submitted_by14'] = f.date_submitted.le(f.date + 14)
    for keys, g in f.groupby(KEY):
        m = g[g.missed]
        at14 = m[m.eligible14]
        counts = at14.final_result.value_counts()
        records.append(dict(definition=definition, code_module=keys[0],code_presentation=keys[1],
            deadline=int(g.date.iloc[0]),weight=float(g.weight.iloc[0]),eligible_at_deadline=len(g),
            missed_deadline=len(m),withdrawn_by14=int((~m.eligible14).sum()),
            missed_still_registered14=len(at14),submitted_by14=int(at14.submitted_by14.sum()),
            final_pass=int(counts.get('Pass',0)+counts.get('Distinction',0)),
            final_fail=int(counts.get('Fail',0)),final_withdrawn=int(counts.get('Withdrawn',0))))
cohort = pd.DataFrame(records)
cohort.to_csv(OUT / 'first_deadline_cohorts.csv', index=False)
summary['cohorts'] = cohort.groupby('definition')[['eligible_at_deadline','missed_deadline','withdrawn_by14','missed_still_registered14','submitted_by14','final_pass','final_fail','final_withdrawn']].sum().to_dict('index')
summary['nonexam_weights_by_module'] = {k: sorted(g.weight.unique().tolist()) for k,g in nonexam.groupby('code_module')}
repeat = info.groupby(['id_student','code_module']).size()
summary['same_module_repeat_student_module_groups'] = int(repeat.gt(1).sum())
summary['same_module_repeat_students'] = int(repeat[repeat.gt(1)].reset_index().id_student.nunique())
summary['positive_tma_sensitivity'] = []
candidates = nonexam[nonexam.weight.gt(0) & nonexam.assessment_type.eq('TMA')]
first_tma = candidates[candidates.date.eq(candidates.groupby(KEY).date.transform('min'))]
f = base.merge(first_tma[KEY + ['id_assessment','date']], on=KEY, validate='many_to_one')
f = f.merge(sa[['id_assessment','id_student','date_submitted','is_banked']], on=['id_assessment','id_student'],how='left',validate='one_to_one')
for grace in [0,1,3,7]:
    start = f.date + grace
    missed = f.date_registration.le(start) & (f.date_unregistration.isna() | f.date_unregistration.gt(start)) & ~f.is_banked.eq(1) & (f.date_submitted.isna() | f.date_submitted.gt(start))
    remaining = missed & (f.date_unregistration.isna() | f.date_unregistration.gt(start + 14))
    summary['positive_tma_sensitivity'].append({'grace_days':grace,'no_submission_at_start':int(missed.sum()),'still_registered14_later':int(remaining.sum())})
assert int(cohort.query('definition == "first_positive_TMA"').missed_still_registered14.sum()) == 5126
assert (cohort.final_pass + cohort.final_fail + cohort.final_withdrawn).eq(cohort.missed_still_registered14).all()
joined = sa.merge(a, on='id_assessment',validate='many_to_one')
joined = joined[~joined.is_banked.eq(1)].copy()
joined['offset'] = joined.date_submitted - joined.date
timing = joined.groupby(KEY + ['assessment_type','id_assessment']).agg(
    deadline=('date','first'), n=('id_student','size'),
    first_submission=('date_submitted','min'), median_submission=('date_submitted','median'),
    on_time=('offset', lambda x: int(x.le(0).sum())),
    offset_q25=('offset', lambda x:x.quantile(.25)), offset_q75=('offset',lambda x:x.quantile(.75)))
timing.to_csv(OUT/'assessment_timing_audit.csv')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(course.to_string())
print(cohort.to_string(index=False))
