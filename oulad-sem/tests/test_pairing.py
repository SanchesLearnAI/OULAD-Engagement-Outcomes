"""Pairing, bootstrap intervals and Holm adjustment."""
import numpy as np
import pandas as pd

from src.tma import effects, pairing


def _frame(rows):
    """rows: (id_student, final_result, post_clicks, pre_clicks)."""
    data = {"code_module": [], "code_presentation": [], "id_student": [],
            "final_result": [], "post_clicks": [], "pre_clicks": [],
            "post_active_days": [], "post_longest_inactive_run": [],
            "post_week2_share": [], "submitted_by_cutoff": [],
            "submitted_after_due": [], "relative_submission_day": [],
            "log_post_clicks": [], "log_pre_clicks": []}
    for i, (student, result, post, pre) in enumerate(rows):
        data["code_module"].append("AAA")
        data["code_presentation"].append("2013J")
        data["id_student"].append(student)
        data["final_result"].append(result)
        data["post_clicks"].append(post)
        data["pre_clicks"].append(pre)
        data["post_active_days"].append(1.0 + i % 3)
        data["post_longest_inactive_run"].append(5.0 + i % 2)
        data["post_week2_share"].append(0.3 + 0.01 * i)
        data["submitted_by_cutoff"].append(1.0 if i % 4 else 0.0)
        data["submitted_after_due"].append(0.0 if i % 3 else 1.0)
        data["relative_submission_day"].append(float(i % 5) if i % 4 else np.nan)
        data["log_post_clicks"].append(float(np.log1p(post)))
        data["log_pre_clicks"].append(float(np.log1p(pre)))
    return pd.DataFrame(data)


def test_pairs_are_unique_and_inside_presentation():
    rows = []
    for i in range(5):
        base = 10.0 * (i + 1)
        rows.append((2 * i, "Fail", base, 20 + i))
        rows.append((2 * i + 1, "Pass", base * 1.02, 21 + i))
    pairs = pairing.make_pairs(_frame(rows), 0.10, 42)
    assert len(pairs) == 5
    assert pairs["fail_index"].is_unique and pairs["pass_index"].is_unique
    assert not set(pairs["fail_index"]) & set(pairs["pass_index"])
    assert set(pairs["code_module"]) == {"AAA"}


def test_zero_clicks_only_match_zero_clicks():
    rows = [(1, "Fail", 0, 5), (2, "Pass", 0, 6), (3, "Fail", 50, 7), (4, "Pass", 50, 8)]
    pairs = pairing.make_pairs(_frame(rows), 0.50, 42)
    for _, r in pairs.iterrows():
        assert (r["fail_clicks"] == 0) == (r["pass_clicks"] == 0)


def test_tolerance_boundary_is_respected():
    rows = [(1, "Fail", 100.0, 10), (2, "Pass", 106.0, 11), (3, "Pass", 104.0, 12)]
    pairs = pairing.make_pairs(_frame(rows), 0.05, 42)
    assert len(pairs) == 1
    assert pairs["pass_clicks"].iloc[0] == 104.0     # 106 is outside 5%
    tight = pairing.make_pairs(_frame(rows), 0.03, 42)
    assert len(tight) == 0                            # 104 exceeds a 3% tolerance


def test_relative_submission_day_uses_only_defined_pairs():
    frame = _frame([(i, "Fail" if i % 2 else "Pass", 10 + i, 20 + i) for i in range(8)])
    pairs = pairing.make_pairs(frame, 0.5, 42)
    with_day = effects.valid_mask(frame, pairs, "relative_submission_day")
    without_cutoff = frame["relative_submission_day"].isna().to_numpy()
    for i, ok in enumerate(with_day):
        if ok:
            assert not (without_cutoff[pairs["fail_index"].iloc[i]]
                        or without_cutoff[pairs["pass_index"].iloc[i]])


def test_holm_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = effects.holm_adjust(p)
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.20*1=0.20
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.20])
    assert effects.holm_adjust([0.5, 0.6]) == [1.0, 1.0]


def test_component_intervals_pair_students():
    rows = [(1, "Fail", 10, 20), (2, "Pass", 10, 21),
            (3, "Fail", 30, 40), (4, "Pass", 30, 41)]
    frame = _frame(rows)
    pairs = pairing.make_pairs(frame, 0.5, 42)
    differences = effects.feature_differences(frame, pairs, ["post_active_days"])
    interval = pairing.component_intervals(pairs, differences, 42, 200)
    assert interval.shape == (2, 1)
    assert interval[0, 0] <= interval[1, 0]


def test_primary_metrics_family_is_six():
    assert len(effects.PRIMARY_METRICS) == 6
    # Submission distance is exploratory here: only the relative day is primary.
    assert "relative_submission_day" in effects.PRIMARY_METRICS
    assert not {"days_early", "days_late"} & set(effects.PRIMARY_METRICS)


def test_effect_table_reports_pair_counts_per_metric():
    frame = _frame([(i, "Fail" if i % 2 else "Pass", 10 + i, 20 + i) for i in range(12)])
    pairs = pairing.make_pairs(frame, 0.5, 42)
    table = effects.effect_table(frame, pairs, ["post_active_days",
                                                "relative_submission_day"], 42, 200)
    counts = table.set_index("feature")["pairs"].to_dict()
    assert counts["post_active_days"] == len(pairs)
    assert counts["relative_submission_day"] <= len(pairs)
