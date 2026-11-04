import numpy as np
import pandas as pd
from src.config import load_config
from src.splits import build_splits


def _master(n=80):
    rng = np.random.default_rng(0)
    mods = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    pres = ["2013J", "2013B", "2014J", "2014B"]
    rows = []
    for i in range(n):
        rows.append({
            "code_module": mods[i % len(mods)],
            "code_presentation": pres[i % len(pres)],
            "id_student": 1000 + (i % 50),
            "y_t1": float(i % 3 != 0),
        })
    return pd.DataFrame(rows)


def _overlap(df, a, b):
    return len(set(df["id_student"].iloc[a]) & set(df["id_student"].iloc[b]))


def test_grouped_splits_have_no_student_overlap():
    cfg = load_config().data
    cohort, splits, _ = build_splits(_master(), cfg, 42)
    assert _overlap(cohort, splits["main"]["train"], splits["main"]["test"]) == 0
    assert _overlap(cohort, splits["main"]["train"], splits["main"]["val"]) == 0
    assert _overlap(cohort, splits["main"]["val"], splits["main"]["test"]) == 0


def test_temporal_and_lomo_are_student_disjoint():
    cfg = load_config().data
    cohort, splits, _ = build_splits(_master(), cfg, 42)
    assert _overlap(cohort, splits["temporal"]["train"], splits["temporal"]["test"]) == 0
    for f in splits["lomo"]:
        assert _overlap(cohort, f["train"], f["test"]) == 0
