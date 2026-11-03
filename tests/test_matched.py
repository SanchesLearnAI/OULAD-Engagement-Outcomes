import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from oulad_analysis.matched import make_pairs, compare_matched, _component_intervals
from oulad_analysis.features import CONCENTRATION_FEATURES


class MatchedTest(unittest.TestCase):
    def fixture(self):
        f=pd.DataFrame([
            ('A','2013J',1,'Fail',100),('A','2013J',2,'Pass',100),
            ('A','2013J',3,'Fail',200),('A','2013J',4,'Pass',208),
            ('A','2013J',5,'Fail',0),('A','2013J',6,'Pass',0),
            ('A','2013J',7,'Fail',300),('B','2013J',8,'Pass',300),
            ('A','2014J',9,'Pass',300),('A','2013J',10,'Withdrawn',300)],
            columns=['code_module','code_presentation','id_student','final_result','total_clicks'])
        f['log_clicks']=np.log1p(f.total_clicks)
        for name in ['active_days','longest_inactive_run','daily_concentration','daily_entropy']+CONCENTRATION_FEATURES+[f'week_{i}_share' for i in range(4)]:
            f[name]=f.final_result.eq('Fail').astype(float)
        return f

    def test_exact_caliper_context_and_no_reuse(self):
        f=self.fixture()
        exact=make_pairs(f,0)
        self.assertEqual(len(exact),2)
        self.assertTrue(exact.relative_gap.eq(0).all())
        pairs=make_pairs(f,.05)
        self.assertEqual(len(pairs),3)
        self.assertTrue(pairs.relative_gap.le(.05).all())
        self.assertTrue(pairs.fail_index.is_unique and pairs.pass_index.is_unique)
        self.assertNotIn(7,pairs.fail_student.tolist())
        pd.testing.assert_frame_equal(pairs,make_pairs(f,.05))

    def test_pair_effects_and_shared_student_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            compare_matched(self.fixture(),Path(d),repeats=30)
            result=pd.read_csv(Path(d)/'matched_effects.csv')
            self.assertTrue(np.allclose(result.fail_minus_pass,1))
            self.assertTrue(np.allclose(result.ci_low,1))
            self.assertTrue(np.allclose(result.ci_high,1))
        pairs=pd.DataFrame({'fail_student':[1,2],'pass_student':[2,3]})
        interval=_component_intervals(pairs,np.array([[1.],[3.]]),42,30)
        np.testing.assert_allclose(interval,[[2.],[2.]])


if __name__=='__main__':
    unittest.main()
