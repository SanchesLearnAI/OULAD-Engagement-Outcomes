import unittest

import numpy as np

from oulad_analysis.tma_relative_gap_model import VERSIONS, relative_click_gap


class RelativeGapModelTest(unittest.TestCase):
    def test_signed_symmetric_gap_and_zero_convention(self):
        pre = np.array([0, 0, 10, 10, 30, 5])
        post = np.array([0, 10, 0, 10, 10, 15])
        np.testing.assert_allclose(relative_click_gap(pre, post),
                                   [0, 1, -1, 0, -.5, .5])
        with self.assertRaises(ValueError):
            relative_click_gap(np.array([-1]), np.array([2]))
        with self.assertRaises(ValueError):
            relative_click_gap(np.array([1]), np.array([np.nan]))

    def test_comparators_change_only_the_engineered_feature(self):
        self.assertEqual(VERSIONS["M0_plus_relative_gap"][:-1],
                         VERSIONS["M0_pre_post_volume_submission"])
        self.assertEqual(VERSIONS["M1_timing_plus_relative_gap"][:-1],
                         VERSIONS["M1_add_post_timing"])


if __name__ == "__main__":
    unittest.main()
