import unittest
import numpy as np
import pandas as pd
from oulad_analysis.tma_response_attributes import attributes


class ResponseAttributesTest(unittest.TestCase):
    def test_direction_persistence_missing_and_no_outcome_dependency(self):
        frame = pd.DataFrame(dict(first_tma_score=[90, 40, 40, np.nan],
            relative_click_gap=[-.8, .8, .8, .8], pre_active_days=[12, 2, 2, 2],
            post_active_days=[2, 10, 10, 10], post_week2_share=[.1, .8, .1, .8]))
        result = attributes(frame)
        self.assertAlmostEqual(result.high_mark_click_drop.iloc[0], .81*.8)
        self.assertEqual(result.low_mark_click_rise.iloc[0], 0)
        self.assertEqual(result.high_mark_click_drop.iloc[1], 0)
        self.assertGreater(result.low_mark_rise_sustained_week2.iloc[1],
                           result.low_mark_rise_sustained_week2.iloc[2])
        self.assertTrue(pd.isna(result.low_mark_click_rise.iloc[3]))
        # No label is required; injecting arbitrary labels must not affect features.
        other = attributes(frame.assign(final_result=['Pass', 'Fail', 'Pass', 'Fail']))
        pd.testing.assert_frame_equal(result, other.drop(columns='final_result'))


if __name__ == '__main__':
    unittest.main()
