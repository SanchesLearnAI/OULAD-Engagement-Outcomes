import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oulad_analysis.tma_score_probe import attach_scores, make_joint_pairs


class TmaScoreProbeTest(unittest.TestCase):
    def test_score_is_restricted_by_submission_and_rank_is_within_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame([
                ("A", "2014J", 10, "TMA", 7, 0),
                ("A", "2014J", 11, "TMA", 10, 10),
                ("B", "2014J", 12, "TMA", 12, 10),
            ], columns=["code_module", "code_presentation", "id_assessment",
                        "assessment_type", "date", "weight"]).to_csv(path / "assessments.csv", index=False)
            pd.DataFrame([
                (11, 1, 11, 0, 80), (11, 2, 31, 0, 90), (11, 3, 12, 0, 60),
                (12, 4, 15, 0, 70),
            ], columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]).to_csv(
                path / "studentAssessment.csv", index=False)
            node = pd.DataFrame([
                ("A", "2014J", 1, "Pass", 31),
                ("A", "2014J", 2, "Fail", 30),
                ("A", "2014J", 3, "Withdrawn", 31),
                ("B", "2014J", 4, "Pass", 33),
            ], columns=["code_module", "code_presentation", "id_student", "final_result", "cutoff_day"])
            scored = attach_scores(node, path).set_index("id_student")
            self.assertEqual(scored.loc[1, "first_tma_score"], 80)
            self.assertTrue(pd.isna(scored.loc[2, "first_tma_score"]))
            self.assertEqual(scored.loc[1, "first_tma_rank_pct"], 1)
            self.assertEqual(scored.loc[3, "first_tma_rank_pct"], .5)
            self.assertEqual(scored.loc[4, "first_tma_rank_pct"], 1)

    def test_joint_matching_respects_course_click_and_score_limits(self):
        frame = pd.DataFrame([
            ("A", "2014J", 1, "Fail", 100, 70),
            ("A", "2014J", 2, "Pass", 100, 74),
            ("A", "2014J", 3, "Fail", 200, 80),
            ("A", "2014J", 4, "Pass", 208, 82),
            ("A", "2014J", 5, "Pass", 200, 95),
            ("B", "2014J", 6, "Pass", 200, 80),
        ], columns=["code_module", "code_presentation", "id_student",
                    "final_result", "total_clicks", "first_tma_score"])
        pairs = make_joint_pairs(frame, seed=42)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(pairs.fail_index.is_unique and pairs.pass_index.is_unique)
        self.assertTrue(pairs.relative_click_gap.le(.05).all())
        self.assertTrue(pairs.score_gap.abs().le(5).all())
        self.assertEqual(set(pairs.pass_student), {2, 4})
        pd.testing.assert_frame_equal(pairs, make_joint_pairs(frame, seed=42))


if __name__ == "__main__":
    unittest.main()
