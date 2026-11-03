import unittest

import numpy as np
import pandas as pd

from oulad_analysis.tma_click_clusters import summarize_volume_clusters


class TmaClickClusterTest(unittest.TestCase):
    def test_compares_outcomes_only_within_the_same_volume_cluster(self):
        clicks = [0, 0, 4, 4, 100, 100, 42, 43]
        labels = np.array([-1, -1, 0, 0, 1, 1, 0, 1])
        rows = []
        for i, count in enumerate(clicks):
            fail = i % 2 == 0
            rows.append(dict(code_module="A", code_presentation="2014J", id_student=i + 1,
                             final_result="Fail" if fail else "Pass", post_clicks=count,
                             pre_clicks=5 if fail else 6, pre_active_days=1 if fail else 2,
                             post_active_days=0 if count == 0 else 1 if fail else 3,
                             post_both_weeks=0 if fail else int(count > 0),
                             post_longest_inactive_run=14 if count == 0 else 9 if fail else 5,
                             post_week2_share=0 if fail else .5,
                             post_peak7_share=0 if count == 0 else .9 if fail else .6,
                             post_entropy_norm=0 if count == 0 else .2 if fail else .5,
                             post_resource_share__forumng=0 if fail else .2,
                             post_resource_share__oucontent=.2 if fail else 0,
                             submitted_by_cutoff=int(not fail), submitted_after_due=int(fail)))
        frame = pd.DataFrame(rows)
        pairs = pd.DataFrame([dict(code_module="A", code_presentation="2014J",
                                   fail_index=i, pass_index=i + 1, fail_student=i + 1,
                                   pass_student=i + 2, fail_clicks=clicks[i],
                                   pass_clicks=clicks[i + 1],
                                   relative_gap=0 if clicks[i] == clicks[i + 1] else 43 / 42 - 1)
                              for i in (0, 2, 4, 6)])
        result = summarize_volume_clusters(frame, pairs, labels, 2, seed=42, repeats=30)
        self.assertEqual(result["cross_cluster_pairs"], 1)
        self.assertEqual(result["balance"].set_index("cluster").pairs.to_dict(),
                         {"zero": 1, "low": 1, "high": 1})
        low = result["effects"].set_index(["cluster", "feature"]).loc[("low", "post_active_days")]
        self.assertEqual(low.fail_minus_pass, -2)
        self.assertEqual(low.ci_low, -2)
        self.assertTrue(result["pre_sensitivity"].pairs.eq(1).all())


if __name__ == "__main__":
    unittest.main()
