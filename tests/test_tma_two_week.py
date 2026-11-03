import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_analysis.matched import make_pairs
from oulad_analysis.tma_two_week import _matching_frame, _time_features, build_behavior_features


class TmaTwoWeekTest(unittest.TestCase):
    def test_equal_post_totals_can_have_distinct_timing(self):
        early = [10] + [0] * 13
        sustained = [1] * 10 + [0] * 4
        features = _time_features(np.array([early, sustained], dtype=float))
        self.assertEqual(features["post_longest_inactive_run"].tolist(), [13, 4])
        self.assertEqual(features["post_week2_share"].tolist(), [0, .3])
        self.assertGreater(features["post_entropy_norm"][1], features["post_entropy_norm"][0])

    def test_raw_events_reconcile_and_window_boundaries_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame([
                ("A", "2014J", 1, "resource"), ("A", "2014J", 2, "forum"),
            ], columns=["code_module", "code_presentation", "id_site", "activity_type"]).to_csv(
                path / "vle.csv", index=False)
            pd.DataFrame([
                ("A", "2014J", 1, 1, -14, 100),  # excluded
                ("A", "2014J", 1, 1, -13, 2),
                ("A", "2014J", 1, 1, 0, 3),
                ("A", "2014J", 1, 1, 1, 4),
                ("A", "2014J", 1, 2, 8, 6),
                ("A", "2014J", 1, 1, 15, 100),  # excluded
            ], columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]).to_csv(
                path / "studentVle.csv", index=False)
            node = pd.DataFrame([
                dict(code_module="A", code_presentation="2014J", id_student=1, final_result="Fail",
                     due_day=0, cutoff_day=14, pre_clicks=5, post_clicks=10, post_active_days=2),
                dict(code_module="A", code_presentation="2014J", id_student=2, final_result="Pass",
                     due_day=0, cutoff_day=14, pre_clicks=0, post_clicks=0, post_active_days=0),
            ])
            frame, post, quality = build_behavior_features(node, path)
            self.assertEqual(post.sum(axis=1).tolist(), [10, 0])
            self.assertEqual(frame.post_longest_inactive_run.tolist(), [6, 14])
            self.assertAlmostEqual(frame.loc[0, "post_resource_share__forum"], .6)
            self.assertEqual(quality["zero_post_click_records"], 1)
            altered = frame.assign(total_clicks=[100, 100], post_clicks=[10, 0])
            self.assertTrue(make_pairs(_matching_frame(altered), 0).empty)


if __name__ == "__main__":
    unittest.main()
