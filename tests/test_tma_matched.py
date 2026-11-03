import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_analysis.tma_matched import (
    add_resource_features, compare_node_matched,
    compare_post_volume_sensitivity, load_resource_counts,
)


class TmaMatchedTest(unittest.TestCase):
    def test_resource_join_cutoffs_and_total_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame([
                ("A", "2014J", 10, "forumng"),
                ("A", "2014J", 11, "resource"),
                ("B", "2014J", 10, "quiz"),
            ], columns=["code_module", "code_presentation", "id_site", "activity_type"]).to_csv(
                path / "vle.csv", index=False)
            pd.DataFrame([
                ("A", "2014J", 1, 10, -13, 2),
                ("A", "2014J", 1, 10, 0, 3),
                ("A", "2014J", 1, 11, 21, 5),
                ("A", "2014J", 1, 11, 22, 7),
                ("A", "2014J", 1, 99, 28, 11),
                ("A", "2014J", 1, 10, 29, 100),
                ("A", "2014J", 2, 10, 0, 5),
                ("A", "2014J", 2, 11, 21, 5),
            ], columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]).to_csv(
                path / "studentVle.csv", index=False)
            base = pd.DataFrame([
                ("A", "2014J", 1, "Fail", 0), ("A", "2014J", 2, "Pass", 0),
            ], columns=["code_module", "code_presentation", "id_student", "final_result", "due_day"])
            counts, quality = load_resource_counts(path, base)
            self.assertEqual(quality["unmatched_event_rows"], 1)
            self.assertEqual(set(quality["resource_types_observed"]), {"forumng", "resource", "__unmatched__"})
            three = base.copy()
            three["total_clicks"] = [10, 10]
            three["post_clicks"] = [5, 5]
            four = base.copy()
            four["total_clicks"] = [28, 10]
            four["post_clicks"] = [23, 5]
            a = add_resource_features(three, counts, 21, quality["resource_types_observed"])
            b = add_resource_features(four, counts, 28, quality["resource_types_observed"])
            self.assertEqual(a.loc[0, "resource_clicks__resource"], 5)
            self.assertEqual(b.loc[0, "resource_clicks__resource"], 12)
            self.assertEqual(b.loc[0, "resource_clicks____unmatched__"], 11)
            self.assertAlmostEqual(a.loc[0, "resource_share__forumng"], .5)
            self.assertAlmostEqual(b.loc[0, "post_resource_share____unmatched__"], 11 / 23)
            self.assertAlmostEqual(a.loc[0, "resource_share__resource"], .5)
            with self.assertRaisesRegex(ValueError, "disagree"):
                wrong = three.copy()
                wrong.loc[0, "total_clicks"] = 99
                add_resource_features(wrong, counts, 21, quality["resource_types_observed"])

    def test_node_pairs_are_within_presentation_and_report_behavior(self):
        frame = pd.DataFrame([
            ("A", "2014J", 1, "Fail", 100, 0, 2),
            ("A", "2014J", 2, "Pass", 100, 1, 4),
            ("A", "2014J", 3, "Fail", 200, 0, 3),
            ("A", "2014J", 4, "Distinction", 208, 1, 5),
            ("B", "2014J", 5, "Pass", 200, 1, 5),
        ], columns=["code_module", "code_presentation", "id_student", "final_result",
                    "total_clicks", "submitted_by_cutoff", "post_active_days"])
        frame["log_clicks"] = np.log1p(frame.total_clicks)
        frame["post_clicks"] = frame.total_clicks // 2
        frame["pre_clicks"] = frame.total_clicks - frame.post_clicks
        frame["submitted_after_due"] = 0
        for name in ["pre_active_days", "last3_share", "post_both_weeks",
                     "post_active_week_fraction", "post_minus_pre_log"]:
            frame[name] = 0
        for name in ["resource_type_count", "resource_entropy",
                     "post_resource_type_count", "post_resource_entropy",
                     "resource_share__forumng", "post_resource_share__forumng"]:
            frame[name] = 0
        with tempfile.TemporaryDirectory() as directory:
            balance, effects, slices = compare_node_matched(
                frame, Path(directory), 21, ["forumng"], repeats=30)
            exact = balance[balance.tolerance.eq(0)].iloc[0]
            loose = balance[balance.tolerance.eq(.05)].iloc[0]
            self.assertEqual(exact.pairs, 1)
            self.assertEqual(loose.pairs, 2)
            submission = effects[(effects.tolerance.eq(.05)) &
                                 effects.feature.eq("submitted_by_cutoff")].iloc[0]
            self.assertEqual(submission.fail_minus_pass, -1)
            self.assertEqual(set(slices.code_module), {"A"})
            sensitivity = compare_post_volume_sensitivity(frame, 21, ["forumng"], repeats=30)
            self.assertEqual(set(sensitivity.pairs), {2})


if __name__ == "__main__":
    unittest.main()
