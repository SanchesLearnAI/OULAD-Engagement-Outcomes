"""Tests for the restored pre-window, gap and volume-cluster features."""
import numpy as np
import pandas as pd
import pytest

from src.tma import daily, exploration


def test_pre_features_reconcile_to_the_window():
    pre = np.asarray([[1, 2, 3] + [0] * 10 + [4]], dtype=float)
    out = daily.pre_features(pre)
    assert out["pre_clicks"].tolist() == [10.0]
    assert out["pre_active_days"].tolist() == [4.0]
    assert out["last3_clicks"].tolist() == [4.0]


def test_burst_features_exclude_the_deadline_day():
    pre = np.zeros((1, 14))
    pre[0, -1] = 100.0        # deadline day: excluded from the burst aggregates
    pre[0, -2] = 5.0          # d-1
    pre[0, -4] = 5.0          # d-3
    out = daily.burst_features(pre, np.array([0.0]))
    assert out["pre_before_due_clicks"].tolist() == [10.0]
    assert out["pre_last1_share"].tolist() == [0.5]
    assert out["pre_last3_share"].tolist() == [1.0]
    assert abs(float(out["pre_due_share"][0]) - 100 / 110) < 1e-12
    assert out["pre_last1_lift_x_late"].tolist() == [0.0]


def test_burst_interaction_follows_the_late_flag():
    pre = np.zeros((2, 14))
    pre[:, -2] = 10.0
    early = daily.burst_features(pre, np.array([0.0, 0.0]))["pre_last1_lift_x_late"]
    late = daily.burst_features(pre, np.array([1.0, 1.0]))["pre_last1_lift_x_late"]
    assert early.tolist() == [0.0, 0.0]
    assert late.tolist() != [0.0, 0.0]


def test_burst_requires_the_fourteen_day_window():
    with pytest.raises(ValueError):
        daily.burst_features(np.zeros((1, 10)), np.zeros(1))


def test_relative_gap_is_bounded():
    pre = np.array([10.0, 0.0, 0.0, 30.0])
    post = np.array([30.0, 0.0, 5.0, 10.0])
    gap = daily.relative_click_gap(pre, post)
    assert abs(gap[0] - 0.5) < 1e-12
    assert abs(gap[1] - 0.0) < 1e-12      # both zero -> neutral, not 0/0
    assert abs(gap[2] - 1.0) < 1e-12      # only post clicks -> +1
    assert abs(gap[3] + 0.5) < 1e-12
    assert np.all(np.abs(gap) <= 1.0)


def test_relative_gap_rejects_bad_input():
    with pytest.raises(ValueError):
        daily.relative_click_gap(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        daily.relative_click_gap(np.array([-1.0]), np.array([1.0]))


def test_both_zero_flag_matches_the_definition():
    frame = pd.DataFrame({
        "code_module": ["AAA"] * 3, "code_presentation": ["2013J"] * 3,
        "id_student": [1, 2, 3], "due_day": [19.0] * 3,
        "final_result": ["Fail", "Pass", "Fail"],
        "submitted_after_due": [0, 1, 0],
        "submitted_by_cutoff": [1, 1, 0],
    })
    rows = []
    for student, clicks in ((1, 0.0), (2, 5.0), (3, 5.0)):
        for day in range(-13, 15):
            rows.append({"code_module": "AAA", "code_presentation": "2013J",
                         "id_student": student, "relative_day": day,
                         "clicks": 0.0 if (student == 1 and day > 0) else clicks})
    matrix = pd.DataFrame(rows)
    out, _, _ = daily.build_daily_features(frame, matrix, None,
                                           pre_days=14, post_days=14)
    assert out["both_zero_clicks"].tolist() == [1, 0, 0]


def test_volume_clusters_keep_zero_separate():
    clicks = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 30.0, 40.0, 45.0, 60.0, 70.0, 80.0])
    table, labels, k = exploration.volume_clusters(clicks, seed=42, k_range=(2, 3))
    assert len(table) >= 1
    assert labels[0] == -1 and labels[1] == -1
    assert labels[2] != -1
    assert k in (2, 3)


def test_within_cluster_pairs_drops_crossing_pairs():
    labels = np.array([0, 0, 1, 1, 0, 1])
    pairs = pd.DataFrame({"fail_index": [0, 4], "pass_index": [1, 5]})
    kept, crossed = exploration.within_cluster_pairs(pairs, labels)
    assert crossed == 1
    assert len(kept) == 1 and int(kept["cluster"].iloc[0]) == 0


def test_cluster_names_are_stable():
    assert exploration.volume_cluster_name(-1, 3) == "zero"
    assert exploration.volume_cluster_name(0, 2) == "low"
    assert exploration.volume_cluster_name(2, 3) == "high"
    assert exploration.volume_cluster_name(3, 4) == "volume_4"
