"""Cohort, anchor and label tests for the TMA design."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.tma.anchor import (binary_cohort, build_tma_cohort, eligibility_steps,
                            first_weighted_tma)

RESULTS = ["Pass", "Distinction", "Fail", "Withdrawn"]


def _inputs(post_days=14):
    n = len(RESULTS)
    key = {"code_module": ["AAA"] * n, "code_presentation": ["2013J"] * n,
           "id_student": list(range(1, n + 1))}
    si = pd.DataFrame({**key, "final_result": RESULTS, "num_of_prev_attempts": [0] * n})
    sr = pd.DataFrame({**key, "date_registration": [-10.0] * n,
                       "date_unregistration": [np.nan, np.nan, np.nan, 40.0]})
    courses = pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2013J"],
                            "module_presentation_length": [268]})
    assessments = pd.DataFrame({
        "code_module": ["AAA"] * 3, "code_presentation": ["2013J"] * 3,
        "id_assessment": [1, 2, 3], "assessment_type": ["TMA", "TMA", "CMA"],
        "weight": [0.0, 10.0, 5.0], "date": [5.0, 19.0, 12.0]})
    submissions = pd.DataFrame({
        "id_assessment": [2, 2, 2, 2], "id_student": [1, 2, 3, 4],
        "date_submitted": [10.0, 25.0, 40.0, 12.0],
        "is_banked": [0, 0, 0, 0], "score": [60.0, 70.0, np.nan, 55.0]})
    return si, sr, courses, assessments, submissions


def test_anchor_is_first_positive_weight_tma():
    assessments = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"], "code_presentation": ["2013J"] * 3,
        "id_assessment": [1, 2, 3], "assessment_type": ["TMA", "TMA", "CMA"],
        "weight": [0.0, 10.0, 20.0], "date": [5.0, 19.0, 9.0]})
    anchor = first_weighted_tma(assessments)
    assert len(anchor) == 1
    assert int(anchor["due_day"].iloc[0]) == 19
    assert int(anchor["id_assessment"].iloc[0]) == 2


def test_fail_is_positive_class():
    frame, _ = build_tma_cohort(*_inputs(), post_days=14)
    binary = binary_cohort(frame)
    by = {r: int(binary.loc[binary["final_result"] == r, "y_fail"].iloc[0]) for r in RESULTS[:3]}
    assert by == {"Pass": 0, "Distinction": 0, "Fail": 1}
    assert not binary["final_result"].eq("Withdrawn").any()


def test_withdrawn_at_cutoff_is_excluded_from_binary():
    frame, stats = build_tma_cohort(*_inputs(), post_days=14)
    steps = eligibility_steps(frame).set_index("step")["records"].to_dict()
    assert steps["all_enrolments"] == 4
    assert steps["withdrawn_at_cutoff"] == 1
    assert steps["binary_records"] == 3


def test_registration_after_deadline_is_ineligible():
    si, sr, courses, assessments, submissions = _inputs()
    sr.loc[1, "date_registration"] = 30.0
    frame, _ = build_tma_cohort(si, sr, courses, assessments, submissions, post_days=14)
    assert not bool(frame.loc[frame["id_student"].eq(2), "eligible"].iloc[0])


def test_banked_attempt_is_ineligible():
    si, sr, courses, assessments, submissions = _inputs()
    submissions.loc[0, "is_banked"] = 1
    frame, stats = build_tma_cohort(si, sr, courses, assessments, submissions, post_days=14)
    assert not bool(frame.loc[frame["id_student"].eq(1), "eligible"].iloc[0])
    assert stats["banked_excluded"] == 1


def test_submission_masked_to_cutoff():
    frame, _ = build_tma_cohort(*_inputs(), post_days=14)
    late = frame[frame["id_student"].eq(3)].iloc[0]
    # Submitted on day 40, which is after due 19 + 14 = 33.
    assert late["submitted_by_cutoff"] == 0
    assert np.isnan(late["relative_submission_day"])


def test_frozen_config_declares_fail_and_two_week_window():
    cfg = load_config().data
    assert cfg["cohort"]["positive_class"] == "fail"
    assert cfg["study"]["post_days"] == 14
    assert cfg["study"]["pre_days"] == 14
    assert cfg["study"]["match_on"] == "post_clicks"
    assert cfg["study"]["multiple_testing"] == "holm"


def test_unknown_positive_class_rejected():
    si, sr, courses, assessments, submissions = _inputs()
    with pytest.raises(Exception):
        build_tma_cohort(si, sr, courses, assessments, submissions,
                         post_days=14)["final_result"].map({"Bogus": 1})
