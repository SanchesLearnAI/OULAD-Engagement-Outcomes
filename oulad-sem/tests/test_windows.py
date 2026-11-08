"""Window boundaries for the TMA design."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.tma import daily


def _daily(rows):
    return pd.DataFrame(rows, columns=["code_module", "code_presentation",
                                       "id_student", "relative_day", "clicks"])


def _index():
    return pd.MultiIndex.from_tuples([("AAA", "2013J", 1)])


def test_window_boundaries_inclusive():
    cfg = load_config().data
    pre_days = int(cfg["study"]["pre_days"])
    post_days = int(cfg["study"]["post_days"])
    assert pre_days == 14 and post_days == 14
    rows = [("AAA", "2013J", 1, day, 1.0) for day in (-14, -13, 0, 1, 14, 15)]
    matrix = daily.click_matrix(_daily(rows), _index(), -pre_days + 1, post_days)
    assert matrix.shape == (1, pre_days + post_days)
    assert matrix.sum() == 4          # -13, 0, 1 and 14 count; -14 and 15 do not


def test_pre_window_includes_deadline_day():
    pre = daily.click_matrix(_daily([("AAA", "2013J", 1, 0, 7.0),
                                     ("AAA", "2013J", 1, 1, 3.0)]),
                             _index(), -13, 0)
    assert pre.sum() == 7.0


def test_longest_zero_run_counts_trailing_and_leading_days():
    clicks = np.array([[0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=float)
    assert daily.longest_zero_run(clicks).tolist() == [3, 0]


def test_time_features_zero_volume():
    post = np.zeros((2, 14), dtype=float)
    out = daily.time_features(post, 14)
    assert out["post_active_days"].tolist() == [0.0, 0.0]
    assert out["post_longest_inactive_run"].tolist() == [14, 14]
    assert out["post_week2_share"].tolist() == [0.0, 0.0]
    assert out["post_first_active_day"].tolist() == [15, 15]


def test_week_features_both_weeks():
    post = np.zeros((1, 14), dtype=float)
    post[0, 0] = 1.0
    post[0, 13] = 1.0
    out = daily.week_features(post, 14)
    assert out["post_both_weeks"].tolist() == [1]
    assert out["post_active_weeks"].tolist() == [2]
    post2 = np.zeros((1, 14), dtype=float)
    post2[0, 0] = 1.0
    assert daily.week_features(post2, 14)["post_both_weeks"].tolist() == [0]


def test_pre_features_last_three_days():
    pre = np.zeros((1, 14), dtype=float)
    pre[0, -1] = 5.0
    pre[0, 0] = 5.0
    out = daily.pre_features(pre)
    assert out["pre_clicks"].tolist() == [10.0]
    assert out["last3_clicks"].tolist() == [5.0]
    assert out["last3_share"].tolist() == [0.5]


def test_week_one_only_has_no_second_week_activity():
    post = np.zeros((1, 1), dtype=float)
    post[0, 0] = 3.0
    out = daily.time_features(post, 1)
    weeks = daily.week_features(post, 1)
    assert out["post_active_days"].tolist() == [1.0]
    assert weeks["post_both_weeks"].tolist() == [0]
