import pandas as pd
from src.config import load_config
from src.windows import bounds, w2_end_value


def test_bounds_cutoff_28():
    cfg = load_config().data
    w1, w2 = bounds(28, None, cfg)
    assert w1 == (0, 27)
    assert w2 is None


def test_bounds_cutoff_27():
    cfg = load_config().data
    w1, w2 = bounds(27, None, cfg)
    assert w1 == (0, 26)
    assert w2 is None


def test_w2_end_fallback_and_tma():
    cfg = load_config().data
    assert w2_end_value(19, cfg) == 55       # first TMA before week 4
    assert w2_end_value(None, cfg) == 55
    assert w2_end_value(40, cfg) == 39       # 29..55 -> first_tma-1
    assert w2_end_value(100, cfg) == 55      # cap
