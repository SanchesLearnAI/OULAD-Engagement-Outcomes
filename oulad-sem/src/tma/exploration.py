"""Exploration blocks C1–C4 (the third work segment of the merge plan).

Each function is self-contained, writes one table, and is called from S11 behind
an ``exploration`` config flag.  Nothing here changes the frozen primary
indicators or the primary model; every table is labelled exploratory.

C1  pre-entry background balance of the matched pairs
C2  re-matching uncertainty and per-presentation heterogeneity
C3  students who had not submitted by the deadline: three-outcome description
C4  at-risk early-warning set that includes later Withdrawn records
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pairing
from .anchor import binary_cohort

CATEGORICAL = ["gender", "age_band", "highest_education", "imd_band", "disability"]
NUMERIC = ["num_of_prev_attempts", "studied_credits"]
PRE_ENTRY = CATEGORICAL + NUMERIC


def _smd(frame: pd.DataFrame, pairs: pd.DataFrame, column: str,
         matched: bool) -> float:
    """Standardised mean difference, Fail minus Pass/Distinction."""
    if matched:
        fail = frame.iloc[pairs["fail_index"].to_numpy()]
        passing = frame.iloc[pairs["pass_index"].to_numpy()]
    else:
        fail = frame[frame["final_result"].eq("Fail")]
        passing = frame[~frame["final_result"].eq("Fail")]
    if column in CATEGORICAL:
        # Categorical columns are compared on the share of the modal category,
        # which keeps the number interpretable as a proportion difference.
        levels = sorted(frame[column].dropna().astype(str).unique())
        diffs = []
        for level in levels:
            a = fail[column].astype(str).eq(level).mean()
            b = passing[column].astype(str).eq(level).mean()
            diffs.append((a - b) / np.sqrt(max((a * (1 - a) + b * (1 - b)) / 2, 1e-12)))
        return float(np.nanmax(np.abs(diffs)) * np.sign(np.nanmean(diffs)))
    a = pd.to_numeric(fail[column], errors="coerce")
    b = pd.to_numeric(passing[column], errors="coerce")
    sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd and np.isfinite(sd) else float("nan")


def background_balance(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """C1: pre-entry covariates, before and after pairing."""
    rows = []
    for column in PRE_ENTRY:
        if column not in frame:
            continue
        rows.append({"covariate": column,
                     "kind": "categorical" if column in CATEGORICAL else "numeric",
                     "smd_unmatched": _smd(frame, pairs, column, matched=False),
                     "smd_matched": _smd(frame, pairs, column, matched=True)})
    table = pd.DataFrame(rows)
    if len(table):
        table["abs_smd_matched"] = table["smd_matched"].abs()
        table["balanced_matched"] = table["abs_smd_matched"] <= 0.10
    return table


def constrained_repair(frame: pd.DataFrame, pairs: pd.DataFrame, *,
                       exact: tuple[str, ...] = ("highest_education", "imd_band"),
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1 sensitivity: drop pairs that disagree on the exact covariates.

    A full re-match under extra exact constraints is a different design; this
    keeps the realised pairs and reports how the estimand changes on the subset
    that already agrees, which is the cheapest honest check.
    """
    mask = np.ones(len(pairs), dtype=bool)
    for column in exact:
        if column not in frame:
            continue
        fail = frame.iloc[pairs["fail_index"].to_numpy()][column].astype(str).to_numpy()
        passing = frame.iloc[pairs["pass_index"].to_numpy()][column].astype(str).to_numpy()
        mask &= fail == passing
    subset = pairs.loc[mask].reset_index(drop=True)
    rows = [{"exact_on": " + ".join(exact), "pairs": int(len(subset)),
             "share_of_main_pairs": float(len(subset) / max(len(pairs), 1))}]
    return subset, pd.DataFrame(rows)


def rematching_interval(frame: pd.DataFrame, *, metric: str, tolerance: float,
                        seed: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C2: resample students, then re-pair inside each resample.

    The main bootstrap fixes the pairs; this one redraws them, so the interval
    includes matching uncertainty.  Each replicate resamples student components,
    keeps the drawn records, re-runs the greedy pairing and recomputes the mean
    difference.  It is slow and therefore capped.
    """
    rng = np.random.default_rng(seed + 4242)
    base = pairing.make_pairs(frame, tolerance, seed)
    observed = float((frame.iloc[base["fail_index"].to_numpy()][metric].to_numpy(dtype=float)
                      - frame.iloc[base["pass_index"].to_numpy()][metric].to_numpy(dtype=float)).mean())
    students = np.concatenate([frame["id_student"].to_numpy()])
    unique = np.unique(students)
    draws, sizes = [], []
    for i in range(repeats):
        sample = rng.choice(unique, size=len(unique), replace=True)
        keep = np.zeros(len(frame), dtype=bool)
        position = {s: i for i, s in enumerate(unique)}
        index = np.zeros(len(unique), dtype=int)
        for s in sample:
            index[position[s]] += 1
        for student, count in zip(unique, index):
            if count:
                keep |= frame["id_student"].to_numpy() == student
        # Duplicated draws are collapsed: the pairing is 1:1, so a repeated
        # student cannot contribute twice.
        sub = frame.loc[keep].reset_index(drop=True)
        pairs = pairing.make_pairs(sub, tolerance, seed + i)
        if len(pairs) < 10:
            continue
        values = (sub.iloc[pairs["fail_index"].to_numpy()][metric].to_numpy(dtype=float)
                  - sub.iloc[pairs["pass_index"].to_numpy()][metric].to_numpy(dtype=float))
        draws.append(float(values.mean()))
        sizes.append(int(len(pairs)))
    draws = np.asarray(draws, dtype=float)
    summary = pd.DataFrame([{
        "metric": metric, "tolerance": tolerance, "seed": seed,
        "repeats_requested": repeats, "repeats_used": int(len(draws)),
        "observed": observed, "mean_rematched": float(draws.mean()) if len(draws) else np.nan,
        "ci_low": float(np.quantile(draws, .025)) if len(draws) else np.nan,
        "ci_high": float(np.quantile(draws, .975)) if len(draws) else np.nan,
        "median_pairs": float(np.median(sizes)) if sizes else np.nan}])
    return summary, pd.DataFrame({"draw": np.arange(len(draws)), "value": draws})


def heterogeneity(pres_effects: pd.DataFrame, metric: str) -> pd.DataFrame:
    """C2: DerSimonian-Laird random-effects summary of per-presentation effects."""
    data = pres_effects[pres_effects["feature"].eq(metric)].copy()
    if len(data) < 3:
        return pd.DataFrame()
    y = data["fail_minus_pass"].to_numpy(dtype=float)
    n = data["pairs"].to_numpy(dtype=float)
    # Within-study variance proxy: treat the pair count as the information and
    # use the pooled spread as the sampling variance scale.
    var = y.var(ddof=1) / np.maximum(n, 1)
    w = 1.0 / var
    fixed = float((w * y).sum() / w.sum())
    q = float((w * (y - fixed) ** 2).sum())
    k = len(y)
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1.0 / (var + tau2)
    pooled = float((w_star * y).sum() / w_star.sum())
    se = float(np.sqrt(1.0 / w_star.sum()))
    tau = float(np.sqrt(tau2))
    # Share of variance from between-presentation differences.
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100) if q > 0 else 0.0
    return pd.DataFrame([{
        "metric": metric, "presentations": int(k), "pooled_effect": pooled,
        "se": se, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se,
        "tau2": tau2, "tau": tau, "i_squared": i2,
        "q": q, "fixed_effect": fixed,
        "min_presentation": float(y.min()), "max_presentation": float(y.max())}])


def non_submitter_outcomes(master: pd.DataFrame) -> pd.DataFrame:
    """C3: among records with no submission by d+14, how do the outcomes split?

    Describes the subgroup the binary comparison cannot speak to; the unit is the
    enrolment and the outcome keeps all three classes.
    """
    eligible = master[master["eligible"] & master["due_day"].notna()]
    late = eligible[eligible["submitted_by_cutoff"].eq(0)]
    if late.empty:
        return pd.DataFrame()
    rows = []
    for outcome, group in late.groupby("y_t3"):
        rows.append({"outcome": outcome, "records": int(len(group)),
                     "share_of_non_submitters": float(len(group) / len(late)),
                     "share_of_eligible": float(len(group) / len(eligible)),
                     "share_never_submitted": float(
                         group["submitted_by_cutoff"].eq(0).mean()),
                     "share_late_submission": float(
                         group["submitted_after_due"].eq(1).mean()),
                     "withdrawn_share": float(group["final_result"].eq("Withdrawn").mean())})
    table = pd.DataFrame(rows)
    table["non_submitters"] = int(len(late))
    table["eligible_records"] = int(len(eligible))
    return table.sort_values("records", ascending=False).reset_index(drop=True)


def non_submitter_behaviour(frame: pd.DataFrame, *, post_days: int = 14) -> pd.DataFrame:
    """C3: post-deadline behaviour of the non-submitters, by final outcome."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for outcome, group in frame.groupby("y_t3"):
        rows.append({"outcome": outcome, "records": int(len(group)),
                     "post_clicks_mean": float(group["post_clicks"].mean()),
                     "post_clicks_median": float(group["post_clicks"].median()),
                     "post_active_days_mean": float(group["post_active_days"].mean()),
                     "post_longest_inactive_run_mean": float(
                         group["post_longest_inactive_run"].mean()),
                     "zero_post_click_share": float(group["post_clicks"].eq(0).mean()),
                     "post_days": int(post_days)})
    return pd.DataFrame(rows)


def at_risk_warning_set(frame: pd.DataFrame, splits: dict, *, seed: int,
                        models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C4: predict ''not passing'' (Fail or Withdrawn) in the at-risk set.

    The cohort is every record that was enrolled at the deadline, so Withdrawn is
    a positive case here.  This is a different target from the binary comparison
    and is reported separately.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from .modeling import _fit, columns_for

    data = frame.copy()
    data["y_not_pass"] = data["y_t3"].isin(["Fail", "Withdrawn"]).astype(int)
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    rows = []
    for model in models:
        cols = [c for c in columns_for(data, model) if c in data]
        prob, thr, rule = _fit(data, cols, tr, te, "logistic", seed, va, 0.80)
        if prob is None:
            continue
        y = data["y_not_pass"].to_numpy()[te]
        rows.append({"model": model, "n_test": int(len(te)),
                     "positive_share": float(y.mean()),
                     "average_precision": float(average_precision_score(y, prob)),
                     "roc_auc": float(roc_auc_score(y, prob)),
                     "brier": float(brier_score_loss(y, prob)),
                     "threshold_rule": rule})
    return data, pd.DataFrame(rows)
