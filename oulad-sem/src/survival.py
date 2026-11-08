"""Competing-risk long table, cumulative incidence and cause-specific Cox."""

# Legacy module kept for the archived day-0-27 design (see
# results/oulad_sem_day0_27_baseline_20261108/). The TMA pipeline in
# stages/ does not import it.

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY

CAUSE = {"Pass": 1, "Fail": 2, "Withdrawn": 3}


def student_event_table(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy().reset_index(drop=True)
    length_weeks = np.floor(m["module_presentation_length"].fillna(240) / 7).astype(int)
    unreg_week = np.floor(m["date_unregistration"].fillna(10**6) / 7).astype(int)
    is_w = (m["y_t3"] == "Withdrawn").to_numpy()
    event_week = np.where(is_w & m["date_unregistration"].notna().to_numpy(),
                          unreg_week, length_weeks)
    event_week = np.clip(event_week, 0, length_weeks)
    out = m[KEY + ["y_t3", "module_presentation_length", "date_unregistration"]].copy()
    out["row"] = np.arange(len(out))
    out["event_type"] = m["y_t3"].map(CAUSE).astype(int)
    out["event_week"] = event_week.astype(int)
    out["course_week"] = length_weeks.astype(int)
    return out


def long_table(student_df: pd.DataFrame, vle: pd.DataFrame,
               master_index: pd.DataFrame) -> pd.DataFrame:
    ew = student_df["event_week"].to_numpy()
    lens = ew + 1
    row = np.repeat(np.arange(len(student_df)), lens)
    week = np.concatenate([np.arange(n) for n in lens])
    lt = pd.DataFrame({"row": row, "week": week})
    lt = lt.merge(master_index[["row"] + KEY], on="row", how="left")
    v = vle[KEY + ["date", "sum_click"]].copy()
    v = v[v["date"] >= 0]
    v["week"] = (v["date"] // 7).astype(int)
    wk = (v.groupby(KEY + ["week"], observed=True)["sum_click"].sum()
          .rename("weekly_clicks").reset_index())
    lt = lt.merge(wk, on=KEY + ["week"], how="left")
    lt["weekly_clicks"] = lt["weekly_clicks"].fillna(0)
    lt = lt.merge(student_df[["row", "event_week", "event_type"]], on="row",
                  how="left")
    lt["event"] = ((lt["week"] == lt["event_week"]) &
                   (lt["event_type"] > 0)).astype(int)
    lt["cause"] = np.where(lt["event"] == 1, lt["event_type"], 0)
    lt = lt.sort_values(["row", "week"])
    lt["cum_clicks"] = lt.groupby("row", observed=True)["weekly_clicks"].cumsum()
    return lt.drop(columns=["event_week"])


def cumulative_incidence(student_df: pd.DataFrame) -> pd.DataFrame:
    max_w = int(student_df["event_week"].max())
    ew = student_df["event_week"].to_numpy()
    et = student_df["event_type"].to_numpy()
    rows = []
    S = 1.0
    cif = {1: 0.0, 2: 0.0, 3: 0.0}
    for t in range(0, max_w + 1):
        at_risk = int((ew >= t).sum())
        n = {1: int(((ew == t) & (et == 1)).sum()),
             2: int(((ew == t) & (et == 2)).sum()),
             3: int(((ew == t) & (et == 3)).sum())}
        h = {c: (n[c] / at_risk) if at_risk else 0.0 for c in n}
        h_all = sum(h.values())
        for c in (1, 2, 3):
            cif[c] += S * h[c]
        rows.append({"week": t, "at_risk": at_risk,
                     "events_pass": n[1], "events_fail": n[2],
                     "events_withdrawn": n[3],
                     "h_pass": h[1], "h_fail": h[2], "h_withdrawn": h[3],
                     "survival": S,
                     "cif_pass": cif[1], "cif_fail": cif[2],
                     "cif_withdrawn": cif[3]})
        S *= (1 - h_all)
    return pd.DataFrame(rows)
