import unittest

import numpy as np
import pandas as pd

from oulad_analysis.assessment_node import SUBMIT, TRACK
from oulad_analysis.wu_method_probe import run_probe


class WuMethodProbeTest(unittest.TestCase):
    def test_fixed_task_and_split_contract(self):
        rows = []
        for batch in range(6):
            for student in range(40):
                clicks = 1 + student + batch
                row = dict(code_module="A", code_presentation=f"201{batch}J",
                           module_presentation=f"A_201{batch}J", id_student=batch * 100 + student,
                           final_result="Fail" if student % 3 == 0 else "Pass",
                           log_clicks=np.log1p(clicks), log_clicks_squared=np.log1p(clicks) ** 2,
                           pre_clicks=student % 11, post_clicks=clicks)
                row.update({name: (student + batch) % 5 for name in SUBMIT + TRACK})
                rows.append(row)
        result, intervals = run_probe(pd.DataFrame(rows), bootstrap=20)
        self.assertEqual(len(result), 12)
        self.assertEqual(len(intervals), 9)
        self.assertTrue(result.groupby("split").test_rows.nunique().eq(1).all())
        self.assertTrue(result[result.split.eq("student_holdout")].overlap_student_ids.eq(0).all())
        self.assertTrue(result[result.split.eq("presentation_no_repeated_students")].overlap_student_ids.eq(0).all())


if __name__ == "__main__":
    unittest.main()
