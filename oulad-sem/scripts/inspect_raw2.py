import sys
import pandas as pd, numpy as np
root=sys.argv[1]
na=['?']
def rd(name, **kw): return pd.read_csv(f"{root}/{name}", na_values=na, **kw)
si=rd("studentInfo.csv"); sr=rd("studentRegistration.csv"); co=rd("courses.csv"); asx=rd("assessments.csv"); vle=rd("vle.csv"); sa=rd("studentAssessment.csv")
print("SI dtypes\n", si.dtypes.to_string())
print("FINAL", si.final_result.value_counts(dropna=False).to_dict())
print("MISSING BY COL", si.isna().sum().to_dict())
print("MISSING SR", sr.isna().sum().to_dict())
print("MISSING SA", sa.isna().sum().to_dict())
print("SCORE describe", sa.score.describe().to_dict(), "score NA", sa.score.isna().sum())
print("BANKED", sa.is_banked.value_counts(dropna=False).to_dict())
print("UNREG NONNULL BY RESULT")
m=si.merge(sr,on=['code_module','code_presentation','id_student'],how='left')
print(m.groupby('final_result').date_unregistration.apply(lambda s: s.notna().sum()).to_dict())
print("UNREG DATE QUANTILES", m.date_unregistration.dropna().quantile([0,.1,.25,.5,.75,.9,1]).to_dict())
print("UNREG NEGATIVE", int((m.date_unregistration<0).sum()))
# after-unregistration clicks
sv=pd.read_csv(f"{root}/studentVle.csv", na_values=na)
print("SV dtype", sv.dtypes.to_dict())
mm=m[m.date_unregistration.notna()][['code_module','code_presentation','id_student','date_unregistration','final_result']]
w=sv.merge(mm,on=['code_module','code_presentation','id_student'],how='inner')
w['after']=w.date>w.date_unregistration
print("WITHDRAWN+other-with-unreg CLICK ROWS", len(w), "AFTER ROWS", int(w['after'].sum()), "STUDENTS WITH AFTER", int(w.loc[w['after'],'id_student'].nunique()), "OF", int(w.id_student.nunique()))
print("BY RESULT", w.groupby('final_result')['after'].agg(['size','sum']).to_dict())
# per student after counts among withdrawn
ws=w[w.final_result=='Withdrawn'].groupby(['code_module','code_presentation','id_student']).agg(n=('after','size'),after=('after','sum'),unreg=('date_unregistration','first'),last=('date','max'))
print("WITHDRAWN students with any post-unreg click", int((ws.after>0).sum()), "of", len(ws))
print("median post-unreg clicks among those", ws.loc[ws.after>0,'after'].median(), "max", ws.after.max())
# clicks in 4 weeks before/after unreg (weekly relative)
ws2=w[w.final_result=='Withdrawn'].copy()
ws2['rel_week']=((ws2.date-ws2.date_unregistration)//7).astype(int)
print("WITHDRAWN weekly click rows by rel_week", ws2[(ws2.rel_week>=-4)&(ws2.rel_week<=4)].groupby('rel_week').size().to_dict())
# label counts per presentation/stratum for sampling
print("STRATA", si.groupby(['code_module','code_presentation','final_result']).size().describe().to_dict(), "n_strata", si.groupby(['code_module','code_presentation','final_result']).ngroups)
# repeats
print("STUDENTS >1 RECORD", int((si.groupby('id_student').size()>1).sum()))
print("presentation types", sorted(si.code_presentation.unique()))
