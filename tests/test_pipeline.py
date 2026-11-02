from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_analysis.cluster import cluster_behaviors
from oulad_analysis.compare import compare_similar_clicks
from oulad_analysis.features import build_features
from oulad_analysis.model import evaluate_models


class PipelineTest(unittest.TestCase):
    def test_end_to_end_and_window_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            rng = np.random.default_rng(17)
            students = []
            events = []
            for student in range(1, 81):
                presentation = "2013J" if student <= 40 else "2014J"
                outcome = "Fail" if student % 3 == 0 else "Pass"
                students.append(("AAA", presentation, student, outcome))
                if student == 1:
                    continue
                for day in (0, 4 + student % 3, 12, 20 + student % 5):
                    events.append(("AAA", presentation, student, 100 + student % 2, day,
                                   int(rng.integers(1, 12)) + int(outcome == "Fail")))
            events.append(events[0])  # Repeated source rows are counted and reported.
            events.append(("AAA", "2013J", 2, 100, -1, 1000))
            events.append(("AAA", "2013J", 2, 100, 28, 1000))
            pd.DataFrame(students, columns=["code_module", "code_presentation", "id_student", "final_result"]).to_csv(data / "studentInfo.csv", index=False)
            pd.DataFrame(events, columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]).to_csv(data / "studentVle.csv", index=False)
            pd.DataFrame([("AAA", p, site, kind) for p in ("2013J", "2014J") for site, kind in ((100, "forum"), (101, "page"))],
                         columns=["code_module", "code_presentation", "id_site", "activity_type"]).to_csv(data / "vle.csv", index=False)

            frame = build_features(data, output)
            self.assertEqual(len(frame), 80)
            zero = frame.set_index("id_student").loc[1]
            self.assertEqual(zero.total_clicks, 0)
            self.assertEqual(zero.longest_inactive_run, 28)
            self.assertEqual(zero.zero_click, 1)
            self.assertEqual(frame.set_index("id_student").loc[2, "total_clicks"], sum(event[-1] for event in events if event[2] == 2 and 0 <= event[4] <= 27))
            quality = json.loads((output / "data_quality.json").read_text())
            self.assertEqual(quality["window_rows"] - quality["window_distinct_rows"], 1)
            shares = frame.loc[frame.total_clicks > 0, [f"week_{i}_share" for i in range(4)]].sum(axis=1)
            self.assertTrue(np.allclose(shares, 1))

            compare_similar_clicks(frame, output)
            cluster_behaviors(frame, output, max_rows=80)
            evaluate_models(frame, output)
            results = json.loads((output / "model_results.json").read_text())
            self.assertEqual(len(results), 7)
            self.assertEqual({r["test_rows"] for r in results}, {results[0]["test_rows"]})
            self.assertTrue((output / "precision_recall.png").is_file())


if __name__ == "__main__":
    unittest.main()
