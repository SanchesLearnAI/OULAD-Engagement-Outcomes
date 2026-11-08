"""Exploration blocks C1–C4: balance, re-matching, heterogeneity, warning set."""
import numpy as np
import pandas as pd

from src.tma import exploration, pairing


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "code_module": "AAA",
            "code_presentation": "2013J",
            "id_student": 1000 + i,
            "module_presentation": "AAA_2013J",
            "final_result": "Fail" if i % 2 else "Pass",
            "y_t3": "Fail" if i % 2 else "Pass",
            "post_clicks": float(10 * (i // 2 + 1)),
            "post_active_days": float(rng.integers(0, 8)),
            "post_longest_inactive_run": float(rng.integers(1, 10)),
            "post_week2_share": float(rng.uniform(0, 1)),
            "log_post_clicks": float(np.log1p(10 * (i // 2 + 1))),
            "log_pre_clicks": float(np.log1p(10)),
            "gender": "M" if i % 3 else "F",
            "age_band": "0-35" if i % 4 else "35-55",
            "highest_education": "A Level or Equivalent" if i % 2 else "Lower Than A Level",
            "imd_band": "0-10%" if i % 5 else "20-30%",
            "disability": "N",
            "num_of_prev_attempts": i % 3,
            "studied_credits": float(60 + i),
        })
    return pd.DataFrame(rows)


def test_background_balance_reports_matched_and_unmatched():
    frame = _frame()
    pairs = pairing.make_pairs(frame, 0.5, 42)
    table = exploration.background_balance(frame, pairs)
    assert len(table) == len(exploration.PRE_ENTRY)
    assert set(table["kind"]) <= {"categorical", "numeric"}
    assert table["smd_matched"].notna().all()


def test_balanced_flag_uses_ten_percent_threshold():
    frame = _frame()
    frame.loc[frame["final_result"].eq("Fail"), "studied_credits"] += 500
    pairs = pairing.make_pairs(frame, 0.5, 42)
    table = exploration.background_balance(frame, pairs)
    row = table[table["covariate"].eq("studied_credits")].iloc[0]
    assert abs(row["smd_matched"]) > 0.10
    assert not bool(row["balanced_matched"])


def test_constrained_repair_keeps_only_agreeing_pairs():
    frame = _frame()
    pairs = pairing.make_pairs(frame, 0.5, 42)
    subset, info = exploration.constrained_repair(
        frame, pairs, exact=("highest_education",))
    assert len(subset) <= len(pairs)
    assert int(info["pairs"].iloc[0]) == len(subset)
    for _, row in subset.iterrows():
        fail = frame.iloc[int(row["fail_index"])]
        passing = frame.iloc[int(row["pass_index"])]
        assert fail["highest_education"] == passing["highest_education"]


def test_rematching_interval_brackets_the_observed_value():
    frame = _frame(n=60)
    summary, draws = exploration.rematching_interval(
        frame, metric="post_active_days", tolerance=0.5, seed=42, repeats=8)
    row = summary.iloc[0]
    assert row["repeats_used"] > 0
    assert row["ci_low"] <= row["ci_high"]
    assert int(row["repeats_used"]) == len(draws)


def test_heterogeneity_flags_dispersion():
    pres = pd.DataFrame({
        "code_module": ["AAA"] * 4,
        "code_presentation": ["2013J", "2013B", "2014J", "2014B"],
        "feature": ["post_active_days"] * 4,
        "pairs": [100, 100, 100, 100],
        "fail_minus_pass": [-0.1, -0.5, -0.3, -0.9],
    })
    table = exploration.heterogeneity(pres, "post_active_days")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["presentations"] == 4
    assert row["ci_low"] < row["pooled_effect"] < row["ci_high"]
    assert 0 <= row["i_squared"] <= 100


def test_non_submitter_outcomes_keep_three_classes():
    master = pd.DataFrame({
        "code_module": ["AAA"] * 4,
        "code_presentation": ["2013J"] * 4,
        "id_student": [1, 2, 3, 4],
        "eligible": [True, True, True, True],
        "due_day": [19.0] * 4,
        "submitted_by_cutoff": [0, 0, 1, 0],
        "submitted_after_due": [0, 0, 0, 1],
        "y_t3": ["Fail", "Withdrawn", "Pass", "Pass"],
        "final_result": ["Fail", "Withdrawn", "Pass", "Pass"],
    })
    table = exploration.non_submitter_outcomes(master)
    assert set(table["outcome"]) == {"Fail", "Withdrawn", "Pass"}
    assert int(table["non_submitters"].iloc[0]) == 3
    assert abs(table["share_of_non_submitters"].sum() - 1.0) < 1e-9


def test_at_risk_warning_targets_fail_or_withdrawn():
    frame = _frame(n=60)
    frame["y_fail"] = frame["y_t3"].isin(["Fail", "Withdrawn"]).astype(int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(frame))
    splits = {"main": {"train": idx[:36].tolist(), "val": idx[36:48].tolist(),
                       "test": idx[48:].tolist()}}
    data, table = exploration.at_risk_warning_set(frame, splits, seed=42, models=["M0"])
    assert "y_not_pass" in data
    assert int(data["y_not_pass"].sum()) == int(frame["y_fail"].sum())
    assert len(table) >= 1
