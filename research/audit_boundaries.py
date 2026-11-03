"""Small diagnostic probes; does not alter the reviewed repository."""
import json
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oulad_analysis.features import build_features

result={}
with zipfile.ZipFile(ROOT/'data/oulad.zip') as z:
    result['raw_files_match_archive_crc']={}
    for name in ('studentInfo.csv','studentVle.csv','vle.csv'):
        crc=0
        with (ROOT/'data/raw'/name).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                crc=zlib.crc32(chunk,crc)
        result['raw_files_match_archive_crc'][name]=crc==z.getinfo(name).CRC
    description=z.read('OULAD.names').decode('cp1252')
    (Path(__file__).parent/'audit_20261102'/'source_dictionary.txt').write_text(description,encoding='utf-8')
for scenario in ('missing_date','unknown_label'):
    with tempfile.TemporaryDirectory() as t:
        data=Path(t)/'data'; data.mkdir(); out=Path(t)/'out'
        pd.DataFrame([['AAA','2013J',1,'Mystery' if scenario=='unknown_label' else 'Pass']],columns=['code_module','code_presentation','id_student','final_result']).to_csv(data/'studentInfo.csv',index=False)
        pd.DataFrame([['AAA','2013J',1,10,0,5],['AAA','2013J',1,10,None if scenario=='missing_date' else 1,100]],columns=['code_module','code_presentation','id_student','id_site','date','sum_click']).to_csv(data/'studentVle.csv',index=False)
        pd.DataFrame([['AAA','2013J',10,'page']],columns=['code_module','code_presentation','id_site','activity_type']).to_csv(data/'vle.csv',index=False)
        f=build_features(data,out)
        q=json.loads((out/'data_quality.json').read_text())
        result[scenario]={'accepted':True,'total_clicks':float(f.total_clicks.iloc[0]),'reported_window_missing_core_rows':q['window_missing_core_rows'],'labels':q['label_counts']}
result['test_run']={'command':'py -m unittest discover -s tests -v','tests':1,'status':'OK','python':'3.14.4','note':'Existing synthetic test, not full retraining or Python 3.11 environment validation'}
path=Path(__file__).parent/'audit_20261102'/'boundary_checks.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
