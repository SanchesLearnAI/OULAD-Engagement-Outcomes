"""Regression tests for the verification scripts' decision rules.

These run on synthetic frames so they stay fast and do not need the OULAD data.
"""
import numpy as np
import pandas as pd

from scripts.verify_frozen_increment import (DEDUP_FEATURES, DEEP_FEATURES, THRESHOLD_AP,
                                              topk_precision)
from scripts.verify_subgroup_mde import bh, collapse, mde_auc


def test_deep_feature_set_matches_the_frozen_protocol():
    # research/TMA_DEEP_PROTOCOL.md was amended when E2 was withdrawn: the three
    # degenerate window-deadline counters were replaced by the anchor facts.
    assert len(DEEP_FEATURES) == 8
    assert "anchor_days_late" in DEEP_FEATURES
    assert "n_internal_silences" in DEEP_FEATURES
    assert not {"missed_deadlines", "days_late_sum",
                "deadlines_in_window"} & set(DEEP_FEATURES)
    assert THRESHOLD_AP == 0.005


def test_deduplicated_set_drops_the_duplicated_columns():
    assert "longest_silence" not in DEDUP_FEATURES
    assert "anchor_submitted" not in DEDUP_FEATURES
    assert set(DEDUP_FEATURES) <= set(DEEP_FEATURES)


def test_topk_precision_counts_ties_at_their_hit_rate():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert topk_precision(y, p, 0.2) == 1.0            # top two are both positive
    assert abs(topk_precision(y, p, 0.5) - 0.6) < 1e-12  # 3 of the top 5
    tied = np.array([0.5] * 10)
    assert abs(topk_precision(y, tied, 0.3) - y.mean()) < 1e-12


def test_bh_is_monotone_and_capped():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    adj = bh(p)
    assert np.all(adj >= p - 1e-12)
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj) >= -1e-12)


def test_mde_grows_as_groups_shrink():
    big = mde_auc(3000, 3000, 0.31)
    small = mde_auc(150, 150, 0.31)
    assert big < small
    assert 0.001 < big < 0.2
    assert small > 0.05


def test_mde_is_undefined_for_single_record_groups():
    # One record per side carries no AUC information, so the MDE must be NaN
    # rather than a misleadingly small number.
    assert np.isnan(mde_auc(1, 1, 0.3))


def test_imd_thirds_collapse():
    raw = pd.Series(["0-10%", "20-30%", "50-60%", "90-100%", "unknown-band"])
    out = collapse(raw, "thirds")
    assert out.tolist()[:2] == ["most_deprived", "most_deprived"]
    assert out.tolist()[2] == "middle"
    assert out.tolist()[3] == "least_deprived"
    assert out.tolist()[4] == "unknown"


def test_education_grouping_keeps_three_levels():
    raw = pd.Series(["Lower Than A Level", "No Formal quals", "A Level or Equivalent",
                     "HE Qualification"])
    out = collapse(raw, "grouped")
    assert out.tolist() == ["below_alevel", "below_alevel", "alevel", "he_or_above"]


def test_collapse_without_rule_is_identity():
    raw = pd.Series(["M", "F"])
    assert collapse(raw, None).tolist() == ["M", "F"]
