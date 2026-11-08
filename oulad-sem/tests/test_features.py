"""Grouped splits, feature construction and leakage assertions."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.tma import daily
from src.tma.modeling import columns_for


def _cohort(n=80):
    rng = np.random.default_rng(0)
    mods = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    pres = ["2013J", "2013B", "2014J", "2014B"]
    rows = []
    for i in range(n):
        rows.append({
            "code_module": mods[i % len(mods)],
            "code_presentation": pres[i % len(pres)],
            "id_student": 1000 + (i % 50),
            "y_fail": int(i % 3 != 0),
        })
    return pd.DataFrame(rows)


def _features():
    key = ["code_module", "code_presentation", "id_student"]
    frame = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "due_day": [19.0, 19.0],
        "final_result": ["Fail", "Pass"],
        "y_fail": [1, 0],
    })
    rows = []
    for student in (1, 2):
        for day in range(-13, 15):
            rows.append({"code_module": "AAA", "code_presentation": "2013J",
                         "id_student": student, "relative_day": day,
                         "clicks": float(max(day, 1))})
    daily_df = pd.DataFrame(rows)
    post_total = float(sum(max(day, 1) for day in range(1, 15)))   # 105 per record
    resources = pd.DataFrame([
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1,
         "activity_type": "resource", "clicks": post_total},
        {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2,
         "activity_type": "resource", "clicks": post_total},
    ])
    return frame, daily_df, resources


def test_features_cover_window_and_reconcile():
    frame, daily_df, resources = _features()
    out, post, quality = daily.build_daily_features(frame, daily_df, resources,
                                                    pre_days=14, post_days=14)
    assert len(out) == 2
    assert out["post_clicks"].tolist() == [105.0, 105.0]
    assert out["pre_clicks"].tolist() == [14.0, 14.0]   # 14 pre days of one click each
    assert post.shape == (2, 14)
    assert quality["binary_records"] == 2


def test_no_score_or_submission_date_in_strict_features():
    cfg = load_config().data
    frame, daily_df, resources = _features()
    out, _, _ = daily.build_daily_features(frame, daily_df, resources)
    for model in ("M0", "M1", "M2"):
        cols = columns_for(out, model)
        assert not any("score" in c.lower() for c in cols), model
        assert "date_submitted" not in cols
    assert "score" in columns_for(out, "M4") or not cfg["leakage"]["include_scores"]


def test_grouped_splits_have_no_student_overlap():
    from src.splits import build_tma_splits
    cfg = load_config().data
    cohort, splits, _ = build_tma_splits(_cohort(), cfg, 42)

    def overlap(a, b):
        return len(set(cohort["id_student"].iloc[a]) & set(cohort["id_student"].iloc[b]))

    assert overlap(splits["main"]["train"], splits["main"]["test"]) == 0
    assert overlap(splits["main"]["train"], splits["main"]["val"]) == 0
    assert overlap(splits["main"]["val"], splits["main"]["test"]) == 0
    assert overlap(splits["temporal"]["train"], splits["temporal"]["test"]) == 0
    for fold in splits["lomo"]:
        assert overlap(fold["train"], fold["test"]) == 0


def test_split_partition_covers_cohort():
    from src.splits import build_tma_splits
    cfg = load_config().data
    cohort, splits, _ = build_tma_splits(_cohort(), cfg, 42)
    total = (len(splits["main"]["train"]) + len(splits["main"]["val"])
             + len(splits["main"]["test"]))
    assert total == len(cohort)
    seen = np.zeros(len(cohort), dtype=int)
    for fold in splits["group_kfold"]:
        seen[fold["test"]] += 1
    assert (seen == 1).all()
