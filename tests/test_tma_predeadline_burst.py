import unittest

import numpy as np
import pandas as pd

from oulad_analysis.tma_predeadline_burst import BURST, VERSIONS, add_burst_features


class TmaPredeadlineBurstTest(unittest.TestCase):
    def test_short_windows_exclude_due_day_and_reconcile(self):
        pre = np.zeros((2, 14), dtype=float)
        pre[0, 0] = 10  # d-13
        pre[0, 10] = 3  # d-3
        pre[0, 12] = 12  # d-1
        pre[0, 13] = 100  # d: outside the burst windows
        pre[1, 0] = 10
        pre[1, 13] = 100
        frame = pd.DataFrame({"pre_clicks": pre.sum(axis=1),
                              "last3_clicks": pre[:, -3:].sum(axis=1),
                              "submitted_after_due": [1, 0]})
        result = add_burst_features(frame, pre)
        self.assertAlmostEqual(result.loc[0, "pre_last1_share"], 12 / 25)
        self.assertAlmostEqual(result.loc[0, "pre_last3_share"], 15 / 25)
        self.assertAlmostEqual(result.loc[1, "pre_last3_share"], 0)
        self.assertAlmostEqual(result.loc[0, "pre_due_share"], 100 / 125)
        self.assertEqual(result.pre_before_due_clicks.tolist(), [25, 10])
        self.assertEqual(result.pre_before_due_active_days.tolist(), [3, 1])
        self.assertGreater(result.loc[0, "pre_last3_rate_lift"], result.loc[1, "pre_last3_rate_lift"])
        self.assertEqual(result.loc[1, "pre_last3_lift_x_late"], 0)
        self.assertEqual(result.loc[0, "pre_last3_lift_x_late"], result.loc[0, "pre_last3_rate_lift"])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            add_burst_features(frame, pre + 1)

    def test_model_versions_preserve_existing_overdue_flag(self):
        baseline = VERSIONS["D0_timing_distance"]
        self.assertIn("submitted_after_due", baseline)
        self.assertEqual(VERSIONS["D13_plus_both_lifts"], baseline + BURST)
        self.assertIn("pre_last3_lift_x_late", VERSIONS["D13_plus_late_interactions"])
        self.assertNotIn("submitted_after_due", VERSIONS["P0_pre_deadline_volume"])
        self.assertNotIn("log_pre_clicks", VERSIONS["P0_pre_deadline_volume"])


if __name__ == "__main__":
    unittest.main()
