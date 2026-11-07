import sys, json
import pandas as pd, numpy as np
root=sys.argv[1]
def rd(name, **kw): return pd.read_csv(f"{root}/{name}", **kw)
si=rd("studentInfo.csv")
sr=rd("studentRegistration.csv")
co=rd("courses.csv")
asx=rd("assessments.csv")
vle=rd("vle.csv")
sa=rd("studentAssessment.csv")
print("SHAPES", {k:v.shape for k,v in dict(studentInfo=si,studentRegistration=sr,courses=co,assessments=asx,vle=vle,studentAssessment=sa).items()})
print("SI COLS", list(si.columns))
print("SR COLS", list(sr.columns))
print("ASSESS COLS", list(asx.columns))
print("SA COLS", list(sa.columns))
print("VLE COLS", list(vle.columns))
print("FINAL", si.final_result.value_counts(dropna=False).to_dict())
print("UNIQUE STUDENTS", si.id_student.nunique(), "ROWS", len(si))
print("PRESENTATIONS", si.groupby(['code_module','code_presentation']).size().to_dict())
print("REG MISSING UNREG", sr.date_unregistration.isna().sum(), "of", len(sr))
print("UNREG NONNULL BY RESULT")
m=si.merge(sr,on=['code_module','code_presentation','id_student'],how='left')
print(m.groupby('final_result').date_unregistration.apply(lambda s: s.notna().sum()).to_dict())
print("COURSE LENGTH", co.to_dict('records'))
# first TMA/CMA per presentation (weight>0)
a=asx[(asx.assessment_type.isin(['TMA','CMA'])) & (asx.weight>0)].copy()
print("ASSESS DATE NA", asx.date.isna().sum(), "TMA weight>0 rows", len(a))
print("FIRST TMA PER PRES", a.groupby(['code_module','code_presentation']).date.min().to_dict())
print("ACTIVITY TYPES", vle.activity_type.value_counts(dropna=False).to_dict())
print("VLE DUP SITE KEYS", vle.duplicated(['code_module','code_presentation','id_site']).sum())
# studentVle
sv=pd.read_csv(f"{root}/studentVle.csv", dtype={'code_module':'category','code_presentation':'category','id_student':'int64','id_site':'int64','date':'int16','sum_click':'int16'})
print("SV SHAPE", sv.shape, "DATE MIN/MAX", int(sv.date.min()), int(sv.date.max()))
neg=(sv.date<0)
print("NEG ROWS", int(neg.sum()), "SHARE", float(neg.mean()), "NEG STUDENTS", int(sv.loc[neg,'id_student'].nunique()))
print("SV PRESENTATIONS", sv.code_presentation.nunique())
# withdrawal audit: clicks after unregistration
mm=m[m.date_unregistration.notna()][['code_module','code_presentation','id_student','date_unregistration','final_result']]
w=sv.merge(mm,on=['code_module','code_presentation','id_student'],how='inner')
w['after']=w.date>w.date_unregistration
print("WITHDRAWN CLICK ROWS", len(w), "AFTER UNREG ROWS", int(w['after'].sum()), "STUDENTS WITH AFTER", int(w.loc[w['after'],'id_student'].nunique()), "OF", int(w.id_student.nunique()))
# distribution of unreg date
print("UNREG DATE QUANTILES", m.date_unregistration.dropna().quantile([0,.1,.25,.5,.75,.9,1]).to_dict())
