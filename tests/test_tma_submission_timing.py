import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oulad_analysis.tma_submission_timing import VERSIONS, add_submission_distance


class SubmissionTimingTest(unittest.TestCase):
    def test_early_late_and_future_submission_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame([("A", "2014J", 10, "TMA", 0, 10)],
                         columns=["code_module", "code_presentation", "id_assessment",
                                  "assessment_type", "date", "weight"]).to_csv(path / "assessments.csv", index=False)
            pd.DataFrame([(10, 1, -2, 0, 80), (10, 2, 3, 0, 50), (10, 3, 15, 0, 20)],
                         columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]).to_csv(
                path / "studentAssessment.csv", index=False)
            node = pd.DataFrame([dict(code_module="A", code_presentation="2014J", id_student=i,
                                      final_result=result, cutoff_day=14)
                                 for i, result in [(1, "Fail"), (2, "Pass"), (3, "Pass")]])
            features = node.assign(due_day=0, pre_clicks=[5, 10, 0], post_clicks=[15, 5, 0],
                                   submitted_by_cutoff=[1, 1, 0], submitted_after_due=[0, 1, 0])
            frame = add_submission_distance(features, node, path)
            self.assertEqual(frame.days_early.iloc[:2].tolist(), [2, 0])
            self.assertEqual(frame.days_late.iloc[:2].tolist(), [0, 3])
            self.assertEqual(frame.relative_submission_day.iloc[:2].tolist(), [-2, 3])
            self.assertTrue(pd.isna(frame.loc[2, "days_early"]))
            self.assertTrue(pd.isna(frame.loc[2, "first_tma_score"]))
            self.assertNotIn("date_submitted", frame.columns)

    def test_model_comparisons_add_only_submission_distance(self):
        self.assertEqual(VERSIONS["M1_add_submission_distance"][:-2], VERSIONS["M1_timing"])
        self.assertEqual(VERSIONS["S0_score_add_submission_distance"][:-2], VERSIONS["S0_score"])


if __name__ == "__main__":
    unittest.main()
