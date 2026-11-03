import unittest

import numpy as np
import pandas as pd

from oulad_analysis.assessment_node import SUBMIT, TRACK
from oulad_analysis.tma_score_model import add_train_only_rank, evaluate


class TmaScoreModelTest(unittest.TestCase):
    def test_rank_uses_training_marks_only(self):
        train = pd.DataFrame({"module_presentation": ["A_2014J"] * 3,
                              "first_tma_score": [20.0, 40.0, np.nan]})
        test = pd.DataFrame({"module_presentation": ["A_2014J"] * 3,
                             "first_tma_score": [30.0, 100.0, np.nan]})
        ranked_train, ranked_test = add_train_only_rank(train, test)
        np.testing.assert_allclose(ranked_train.first_tma_rank_train[:2], [.25, .75])
        np.testing.assert_allclose(ranked_test.first_tma_rank_train[:2], [.5, 1.0])
        self.assertTrue(pd.isna(ranked_test.first_tma_rank_train.iloc[2]))
        test.loc[1, "first_tma_score"] = 999.0
        _, changed = add_train_only_rank(train, test)
        self.assertEqual(changed.first_tma_rank_train.iloc[0], .5)

    def test_all_models_use_same_student_holdout(self):
        rows = []
        for batch in range(3):
            for student in range(55):
                clicks = 20 + student + batch
                row = {"code_module": "A", "code_presentation": f"201{batch}J",
                       "module_presentation": f"A_201{batch}J",
                       "id_student": batch * 100 + student,
                       "final_result": "Fail" if student % 3 == 0 else "Pass",
                       "log_clicks": np.log1p(clicks),
                       "log_clicks_squared": np.log1p(clicks) ** 2,
                       "pre_clicks": student % 13, "post_clicks": clicks,
                       "first_tma_score": np.nan if student % 8 == 0 else float(50 + student % 40),
                       "score_recorded_for_submission": student % 8 != 0,
                       "resource_type_count": 2, "resource_entropy": .5,
                       "post_resource_type_count": 1, "post_resource_entropy": .2,
                       "resource_share__forumng": student / 100,
                       "post_resource_share__forumng": student / 100}
                row.update({name: student % 5 for name in SUBMIT + TRACK})
                rows.append(row)
        metrics, intervals = evaluate(pd.DataFrame(rows), 21, bootstrap=20)
        self.assertEqual(len(metrics), 30)
        self.assertEqual(len(intervals), 5)
        self.assertEqual(metrics.groupby("seed").test_rows.nunique().max(), 1)
        self.assertTrue(metrics.ap.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
