"""Cohort construction, target definitions, withdrawal audit and sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY


def build_master(si: pd.DataFrame, sr: pd.DataFrame, courses: pd.DataFrame,
                 assessments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    si = si.copy()
    sr = sr.copy()
    courses = courses.copy()
    key = KEY
    if si.duplicated(key).any() or sr.duplicated(key).any():
        raise ValueError("studentInfo/studentRegistration key is not unique")
    m = si.merge(sr, on=key, how="left", validate="one_to_one", indicator=True)
    info["join_loss"] = int((m["_merge"] == "left_only").sum())
    info["rows"] = int(len(m))
    m = m.drop(columns=["_merge"])
    m = m.merge(courses, on=["code_module", "code_presentation"], how="left",
                validate="many_to_one")
    info["courses_missing"] = int(m["module_presentation_length"].isna().sum())

    a = assessments.copy()
    a = a[(a["assessment_type"].isin(["TMA", "CMA"])) & (a["weight"] > 0) &
          (a["date"].notna())]
    first = (a.groupby(["code_module", "code_presentation"], observed=True)["date"]
             .min().rename("first_tma_date").reset_index())
    m = m.merge(first, on=["code_module", "code_presentation"], how="left",
                validate="many_to_one")
    info["first_tma_missing"] = int(m["first_tma_date"].isna().sum())
    info["first_tma_presentations"] = int(len(first))

    m["is_withdrawn"] = (m["final_result"] == "Withdrawn").astype(int)
    m["has_unregistration"] = m["date_unregistration"].notna().astype(int)

    fr = m["final_result"]
    m["y_t1"] = np.where(fr.isin(["Pass", "Distinction"]), 1.0,
                         np.where(fr == "Fail", 0.0, np.nan))
    m["y_t2"] = np.where(fr.isin(["Pass", "Distinction"]), 1.0, 0.0)
    m["y_t3"] = np.where(fr.isin(["Pass", "Distinction"]), "Pass",
                         np.where(fr == "Fail", "Fail", "Withdrawn"))
    m["is_repeater"] = (m["num_of_prev_attempts"] > 0).astype(int)

    # deterministic order is required because splits.json stores row positions
    m = m.sort_values(KEY).reset_index(drop=True)
    info["n_withdrawn"] = int(m["is_withdrawn"].sum())
    info["n_repeaters"] = int(m["is_repeater"].sum())
    info["n_t1"] = int(m["y_t1"].notna().sum())
    info["n_t2"] = int(len(m))
    info["final_result_counts"] = {str(k): int(v) for k, v in
                                   m["final_result"].value_counts().items()}
    return m, info


def add_at_risk_flags(master: pd.DataFrame, cutoffs: list[int]) -> pd.DataFrame:
    m = master.copy()
    unreg = m["date_unregistration"]
    for c in cutoffs:
        m[f"withdrawn_before_c{c}"] = ((unreg.notna()) & (unreg <= c)).astype(int)
        m[f"in_at_risk_c{c}"] = (1 - m[f"withdrawn_before_c{c}"]).astype(int)
    return m


def cohort_flow(master: pd.DataFrame, cutoffs: list[int]) -> pd.DataFrame:
    rows = []
    for c in cutoffs:
        fl = master
        ar = master[master[f"in_at_risk_c{c}"] == 1]
        t1 = ar[ar["y_t1"].notna()]
        t3 = ar["y_t3"].value_counts()
        rows.append({
            "cutoff": c,
            "n_total": int(len(fl)),
            "n_withdrawn": int(fl["is_withdrawn"].sum()),
            "n_unreg_before_c": int(fl[f"withdrawn_before_c{c}"].sum()),
            "n_at_risk": int(len(ar)),
            "n_t1": int(len(t1)),
            "base_rate_t1": float(t1["y_t1"].mean()) if len(t1) else float("nan"),
            "n_t2": int(len(ar)),
            "base_rate_t2": float(ar["y_t2"].mean()) if len(ar) else float("nan"),
            "n_t3_pass": int(t3.get("Pass", 0)),
            "n_t3_fail": int(t3.get("Fail", 0)),
            "n_t3_withdrawn": int(t3.get("Withdrawn", 0)),
        })
    return pd.DataFrame(rows)


def withdrawal_audit(master: pd.DataFrame, vle: pd.DataFrame):
    """Test the rejected bundle: does the click stream stop at unregistration?"""
    un = master[master["date_unregistration"].notna()][
        KEY + ["date_unregistration", "final_result", "is_withdrawn"]].copy()
    v = vle[KEY + ["date", "sum_click"]].merge(un, on=KEY, how="inner")
    v["after"] = v["date"] > v["date_unregistration"]
    v["clicks_after"] = np.where(v["after"], v["sum_click"], 0)
    per_pres = (v.groupby(["code_module", "code_presentation"], observed=True)
                .agg(click_rows=("after", "size"),
                     rows_after=("after", "sum"),
                     students=("id_student", "nunique"),
                     students_after=("id_student", lambda s: 0)).reset_index())
    # students with at least one post-unregistration click, per presentation
    post = (v[v["after"]].groupby(["code_module", "code_presentation"],
                                  observed=True)["id_student"].nunique()
            .rename("students_with_post_click").reset_index())
    per_pres = per_pres.drop(columns=["students_after"]).merge(
        post, on=["code_module", "code_presentation"], how="left")
    per_pres["students_with_post_click"] = \
        per_pres["students_with_post_click"].fillna(0).astype(int)
    per_pres["post_click_share"] = per_pres["rows_after"] / per_pres["click_rows"]

    w = v[v["is_withdrawn"] == 1].copy()
    w["rel_week"] = np.floor((w["date"] - w["date_unregistration"]) / 7).astype(int)
    weekly = (w[(w["rel_week"] >= -4) & (w["rel_week"] <= 4)]
              .groupby("rel_week", observed=True)
              .agg(click_rows=("sum_click", "size"),
                   clicks=("sum_click", "sum"),
                   students=("id_student", "nunique")).reset_index())
    weekly["window"] = np.where(weekly["rel_week"] < 0, "before", "after")

    per_student = (w.groupby(KEY, observed=True)
                   .agg(rows=("after", "size"), rows_after=("after", "sum"),
                        clicks_after=("clicks_after", "sum"),
                        unreg=("date_unregistration", "first"),
                        last_date=("date", "max")).reset_index())
    stats = {
        "unregistered_enrolments": int(len(un)),
        "unregistered_with_clicks": int(per_student.shape[0]),
        "students_with_post_click": int((per_student["rows_after"] > 0).sum()),
        "post_unregistration_click_rows": int(v["after"].sum()),
        "post_unregistration_clicks": int(v.loc[v["after"], "sum_click"].sum()),
        "post_click_share_rows": float(v["after"].mean()),
        "median_clicks_after_among_those": float(
            per_student.loc[per_student["rows_after"] > 0, "clicks_after"].median()
        ) if (per_student["rows_after"] > 0).any() else 0.0,
        "max_clicks_after": int(per_student["clicks_after"].max()) if len(per_student) else 0,
        "negative_unregistration": int((master["date_unregistration"] < 0).sum()),
        "post_clicks_exist": bool(v["after"].sum() > 0),
    }
    return per_pres, weekly, stats


def stratified_sample(master: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Allocate >=1 per (module, presentation, final_result) stratum, then largest remainder."""
    # Sampling strata are module x presentation x final_result, not enrolments.
    strata = master.groupby(["code_module", "code_presentation", "final_result"],
                            observed=True).size()
    strata = strata[strata > 0]
    sizes = strata.reset_index(name="n")
    k = len(sizes)
    if n < k:
        raise ValueError(f"sample size {n} < number of strata {k}")
    alloc = np.ones(k, dtype=int)
    remaining = n - k
    quota = remaining * sizes["n"].to_numpy() / sizes["n"].sum()
    base = np.floor(quota).astype(int)
    alloc += base
    left = n - alloc.sum()
    if left > 0:
        order = np.argsort(-(quota - base))
        for i in order[:left]:
            alloc[i] += 1
    sizes["alloc"] = np.minimum(alloc, sizes["n"])
    # if caps reduced the total, redistribute deterministically
    short = n - int(sizes["alloc"].sum())
    if short > 0:
        for i in np.argsort(-sizes["n"].to_numpy()):
            if short == 0:
                break
            room = int(sizes.loc[i, "n"] - sizes.loc[i, "alloc"])
            take = min(room, short)
            sizes.loc[i, "alloc"] += take
            short -= take
    rng = np.random.default_rng(seed)
    picks = []
    for row in sizes.itertuples(index=False):
        sub = master[(master["code_module"] == row.code_module) &
                     (master["code_presentation"] == row.code_presentation) &
                     (master["final_result"] == row.final_result)]
        idx = rng.choice(sub.index.to_numpy(), size=int(row.alloc), replace=False)
        picks.append(pd.DataFrame({
            "code_module": row.code_module,
            "code_presentation": row.code_presentation,
            "id_student": master.loc[idx, "id_student"].to_numpy(),
            "final_result": row.final_result,
            "stratum_n": int(row.n),
            "sample_seed": seed,
        }))
    return pd.concat(picks, ignore_index=True)
