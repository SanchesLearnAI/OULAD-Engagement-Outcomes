import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oulad_analysis.tma_score_gap import VERSIONS, build_score_frame


class TmaScoreGapTest(unittest.TestCase):
    def test_score_interaction_respects_submission_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame([("A", "2014J", 10, "TMA", 0, 10)],
                         columns=["code_module", "code_presentation", "id_assessment",
                                  "assessment_type", "date", "weight"]).to_csv(path / "assessments.csv", index=False)
            pd.DataFrame([(10, 1, 2, 0, 80), (10, 2, 15, 0, 50)],
                         columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]).to_csv(
                path / "studentAssessment.csv", index=False)
            node = pd.DataFrame([dict(code_module="A", code_presentation="2014J", id_student=i,
                                      final_result=result, cutoff_day=14)
                                 for i, result in [(1, "Fail"), (2, "Pass")]])
            features = node.assign(due_day=0, pre_clicks=[5, 10], post_clicks=[15, 0])
            frame = build_score_frame(features, node, path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.relative_click_gap.tolist(), [.5, -1])
            self.assertAlmostEqual(frame.loc[0, "score_x_gap"], .4)
            self.assertTrue(pd.isna(frame.loc[1, "first_tma_score"]))
            self.assertTrue(pd.isna(frame.loc[1, "score_x_gap"]))

    def test_interaction_is_only_added_after_score_and_gap(self):
        self.assertEqual(VERSIONS["S2_score_gap_interaction"][:-1], VERSIONS["S1_score_gap"])


if __name__ == "__main__":
    unittest.main()
