import numpy as np
import pandas as pd
import pytest

from src.cohort import build_master
from src.config import load_config

RESULTS = ["Pass", "Distinction", "Fail", "Withdrawn"]


def _inputs():
    n = len(RESULTS)
    key = {"code_module": ["AAA"] * n, "code_presentation": ["2013J"] * n,
           "id_student": list(range(1, n + 1))}
    si = pd.DataFrame({**key, "final_result": RESULTS, "num_of_prev_attempts": [0] * n})
    sr = pd.DataFrame({**key, "date_registration": [-10.0] * n,
                       "date_unregistration": [np.nan, np.nan, np.nan, 40.0]})
    courses = pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2013J"],
                            "module_presentation_length": [268]})
    assessments = pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2013J"],
                                "assessment_type": ["TMA"], "weight": [10.0],
                                "date": [19.0]})
    return si, sr, courses, assessments


def _by_result(m, col):
    return [m.loc[m["final_result"] == r, col].iloc[0] for r in RESULTS]


def test_fail_is_positive_class():
    m, _ = build_master(*_inputs(), positive_class="fail")
    y1 = _by_result(m, "y_t1")
    assert y1[:3] == [0.0, 0.0, 1.0] and np.isnan(y1[3])
    assert _by_result(m, "y_t2") == [0.0, 0.0, 1.0, 1.0]


def test_is_pass_independent_of_direction():
    for pc in ("fail", "pass"):
        m, _ = build_master(*_inputs(), positive_class=pc)
        ip = _by_result(m, "is_pass")
        assert ip[:3] == [1.0, 1.0, 0.0] and np.isnan(ip[3])


def test_frozen_config_declares_fail():
    assert load_config().data["cohort"]["positive_class"] == "fail"


def test_unknown_positive_class_rejected():
    with pytest.raises(ValueError):
        build_master(*_inputs(), positive_class="withdrawn")
