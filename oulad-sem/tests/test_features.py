import numpy as np
import pandas as pd
from src.config import load_config
from src.features import build_features, feature_columns


def _inputs():
    key = ["code_module", "code_presentation", "id_student"]
    master = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "first_tma_date": [19.0, 19.0],
    })
    vle = pd.DataFrame({
        "code_module": ["AAA"] * 6,
        "code_presentation": ["2013J"] * 6,
        "id_student": [1, 1, 1, 2, 2, 2],
        "id_site": [10, 10, 10, 11, 11, 11],
        "date": [-1, 0, 27, 28, 5, 6],
        "sum_click": [5, 2, 3, 9, 4, 7],
        "activity_type": ["resource"] * 6,
    })
    sa = pd.DataFrame({
        "code_module": ["AAA"] * 2,
        "code_presentation": ["2013J"] * 2,
        "id_student": [1, 2],
        "date_submitted": [3, 10],
        "date": [20, 20],
        "is_banked": [0, 0],
        "score": [70.0, 80.0],
    })
    return master, vle, sa


def test_features_cutoff_and_leakage():
    master, vle, sa = _inputs()
    cfg = load_config().data
    f, info = build_features(master, vle, sa, 28, cfg)
    assert len(f) == 2
    assert info["max_date_seen"] < 28
    assert not info["forbidden_columns"]
    assert not any(c.lower().startswith(("w2_", "d_")) for c in f.columns)
    assert "score" not in " ".join(f.columns).lower()


def test_pre_start_feature_is_selected():
    master, vle, sa = _inputs()
    cfg = load_config().data
    f, _ = build_features(master, vle, sa, 28, cfg, pre_start="feature")
    assert "pre_start_clicks" in f.columns
    cols = feature_columns(f, 3, 28, include_pre_start=True)
    assert "pre_start_clicks" in cols
    assert "pre_start_active_days" in cols
    cols_no = feature_columns(f, 3, 28, include_pre_start=False)
    assert "pre_start_clicks" not in cols_no
