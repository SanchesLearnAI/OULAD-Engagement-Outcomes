import unittest
import numpy as np
import pandas as pd
from oulad_analysis.tma_submission_persistence import attributes, COMBOS


class SubmissionPersistenceTest(unittest.TestCase):
    def test_timing_status_and_persistence_combinations(self):
        frame=pd.DataFrame(dict(submitted_by_cutoff=[1,1,0],submitted_after_due=[0,1,0],
            days_late=[0,7,np.nan],days_early=[7,0,np.nan],
            post_active_days=[7,2,0],post_both_weeks=[1,0,0],
            post_longest_inactive_run=[3,7,14],post_week2_share=[.5,0,0]))
        x=attributes(frame)
        self.assertEqual(x.submission_status.tolist(),['on_time','late','not_submitted'])
        self.assertEqual(x.ontime_both_weeks.tolist(),[1,0,0])
        self.assertAlmostEqual(x.late_days_x_inactive_run.iloc[1],.25)
        self.assertAlmostEqual(x.early_days_x_active_days.iloc[0],.25)
        self.assertEqual(x.unsubmitted_x_inactive_run.tolist(),[0,0,1])
        self.assertEqual(x.late_days_x_inactive_run.iloc[2],0)
        self.assertFalse(x[COMBOS].isna().any().any())
        other=attributes(frame.assign(final_result=['Fail','Pass','Pass']))
        pd.testing.assert_frame_equal(x[COMBOS],other[COMBOS])


if __name__=='__main__':
    unittest.main()
