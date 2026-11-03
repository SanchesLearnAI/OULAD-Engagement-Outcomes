import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oulad_analysis.assessment_node import build_node_features, evaluate_node
from oulad_analysis.assessment_node_compare import compare_common_cohort


class AssessmentNodeTest(unittest.TestCase):
    def test_deadline_window_eligibility_and_submission_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            info = pd.DataFrame([("A", "2014J", i, result) for i, result in
                                 [(1,"Fail"),(2,"Pass"),(3,"Withdrawn"),(4,"Fail"),(5,"Pass")]],
                                columns=["code_module","code_presentation","id_student","final_result"])
            registration = info[["code_module","code_presentation","id_student"]].copy()
            registration["date_registration"] = [-2,0,0,20,0]
            registration["date_unregistration"] = [None,None,29,None,None]
            assessments = pd.DataFrame([("A","2014J",10,"TMA",10,0),
                                        ("A","2014J",11,"TMA",15,10),
                                        ("A","2014J",12,"Exam","?",90)],
                                       columns=["code_module","code_presentation","id_assessment","assessment_type","date","weight"])
            submitted = pd.DataFrame([(11,1,40,0),(11,2,20,0),(11,5,10,1)],
                                     columns=["id_assessment","id_student","date_submitted","is_banked"])
            events = pd.DataFrame([("A","2014J",1,1,99),("A","2014J",1,2,5),
                                   ("A","2014J",1,15,2),("A","2014J",1,16,3),
                                   ("A","2014J",1,29,4),("A","2014J",1,30,99),
                                   ("A","2014J",2,8,3)],
                                  columns=["code_module","code_presentation","id_student","date","sum_click"])
            for name, frame in [("studentInfo",info),("studentRegistration",registration),
                                ("assessments",assessments),("studentAssessment",submitted),("studentVle",events)]:
                frame.to_csv(path/f"{name}.csv",index=False)
            frame,stats = build_node_features(path,chunksize=2)
            self.assertNotIn("date_submitted", frame.columns)
            self.assertEqual(frame.id_student.tolist(),[1,2])
            self.assertEqual(stats["weighted_tma_presentations"],1)
            one=frame.set_index("id_student").loc[1]
            self.assertEqual(one.due_day,15)
            self.assertEqual(one.cutoff_day,29)
            self.assertEqual(one.pre_clicks,7)
            self.assertEqual(one.post_clicks,7)
            self.assertEqual(one.submitted_by_cutoff,0)
            self.assertEqual(frame.set_index("id_student").loc[2,"submitted_after_due"],1)
            three, three_stats = build_node_features(path, post_weeks=3, chunksize=2)
            four, four_stats = build_node_features(path, post_weeks=4, chunksize=2)
            self.assertEqual(three_stats["post_days"],21)
            self.assertEqual(four_stats["post_days"],28)
            self.assertEqual(three.set_index("id_student").loc[1,"post_clicks"],106)
            self.assertEqual(three.set_index("id_student").loc[1,"submitted_by_cutoff"],0)
            self.assertEqual(four.set_index("id_student").loc[1,"submitted_by_cutoff"],1)
            self.assertEqual(four.set_index("id_student").loc[1,"cutoff_day"],43)
            self.assertAlmostEqual(four.set_index("id_student").loc[1,"post_active_week_fraction"],.75)
            with self.assertRaises(ValueError):
                build_node_features(path, post_weeks=5)

    def test_grouped_model_output(self):
        rows=[]
        for i in range(60):
            rows.append(dict(code_module="A",code_presentation="2014J",id_student=i,
                             final_result="Fail" if i%2 else "Pass",total_clicks=i+1,
                             pre_clicks=i%7,post_clicks=i+1,pre_active_days=i%4,
                             post_active_days=i%6,last3_share=.2,post_both_weeks=i%2,
                             post_active_week_fraction=(i%3)/2,
                             post_minus_pre_log=1.0,submitted_by_cutoff=i%3==0,
                             submitted_after_due=i%5==0,module_presentation="A_2014J",
                             log_clicks=__import__('numpy').log1p(i+1),
                             log_clicks_squared=__import__('numpy').log1p(i+1)**2))
        models, intervals, course = evaluate_node(pd.DataFrame(rows),seed=42,bootstrap=20)
        self.assertEqual(len(models),20)
        self.assertEqual(len(intervals),3)
        self.assertTrue((models.test_students < models.test_rows+1).all())
        self.assertFalse(course.empty)

    def test_common_cohort_uses_same_students_at_every_horizon(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for weeks in (2, 3, 4):
                path = Path(directory) / str(weeks)
                path.mkdir()
                paths[weeks] = path
                rows = []
                for i in range(70 + (4 - weeks) * 5):
                    clicks = i + weeks
                    rows.append(dict(code_module="A", code_presentation="2014J", id_student=i,
                                     final_result="Fail" if i % 2 else "Pass",
                                     module_presentation="A_2014J", pre_clicks=i % 5,
                                     post_clicks=clicks, log_clicks=np.log1p(clicks),
                                     log_clicks_squared=np.log1p(clicks)**2,
                                     submitted_by_cutoff=i % 3 == 0, submitted_after_due=i % 5 == 0,
                                     pre_active_days=i % 4, post_active_days=i % 6,
                                     last3_share=.2, post_both_weeks=i % 2,
                                     post_active_week_fraction=(i % weeks) / weeks,
                                     post_minus_pre_log=1.0))
                pd.DataFrame(rows).to_parquet(path / "node_features.parquet", index=False)
            result = compare_common_cohort(paths)
            self.assertEqual(set(result.common_cohort_rows), {70})
            self.assertEqual(result.groupby("seed").test_rows.nunique().max(), 1)
            self.assertEqual(len(result), 30)


if __name__ == "__main__":
    unittest.main()
