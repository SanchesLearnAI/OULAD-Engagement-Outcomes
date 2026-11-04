"""Window boundaries.  All dates are relative to presentation start (day 0)."""
from __future__ import annotations

import pandas as pd


def bounds(cutoff: int, first_tma, config) -> tuple[tuple[int, int] | None,
                                                    tuple[int, int] | None]:
    """Return (W1, W2) inclusive day ranges, or None when the window is empty."""
    w1 = (0, min(int(config["windows"]["w1_end"]), cutoff - 1))
    if w1[1] < w1[0]:
        w1 = None
    if cutoff <= 28:
        return w1, None
    end = cutoff - 1
    if (config["windows"].get("use_first_assessment", True)
            and first_tma is not None and not pd.isna(first_tma)):
        end = min(end, int(first_tma) - 1)
    end = min(end, int(config["windows"]["w2_end"]))
    w2 = (28, end) if end >= 28 else None
    return w1, w2


def w2_end_value(first_tma, config) -> int:
    """Per-presentation W2 end.

    Plan 2.3: if the first weighted TMA/CMA is before week 4 (day <= 28),
    fall back to the fixed 28-55 window and note it; otherwise use
    min(55, first_tma-1).
    """
    cap = int(config["windows"]["w2_end"])
    if first_tma is None or pd.isna(first_tma):
        return cap
    ft = int(first_tma)
    if ft <= 28:
        return cap
    return min(cap, ft - 1)


def w2_end_label(first_tma) -> str:
    if first_tma is None or pd.isna(first_tma):
        return "55_fallback_no_tma"
    return f"min(55, first_tma-1={int(first_tma)-1})"
