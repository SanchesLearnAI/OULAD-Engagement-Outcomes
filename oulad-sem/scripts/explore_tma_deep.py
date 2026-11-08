"""Deep exploration on the frozen TMA two-week design (E1-E5).

This is exploratory research code: it reads the delivered interim/processed
tables, adds analyses that the frozen protocol deliberately left out, and writes
only aggregate outputs.  Nothing here is part of the primary result.

E1  Return latency after the deadline (did they come back at all, and when)
E2  Multi-deadline submission trajectory inside the observation window
E3  Composition shift: pre- versus post-deadline activity types
E4  Discriminative day-pattern mining (volume symbols, run structure)
E5  Data-minimization ablation and subgroup audit (Wu et al. 2026 protocol)

Run from ``oulad-sem/``::

    python scripts/explore_tma_deep.py --out ../results/tma_deep_exploration_20261108
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import daily, pairing  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402
from src.tma.stats import bh  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]
ASSESS_CATS = {"quiz", "externalquiz", "questionnaire"}
FORUM_CATS = {"forumng"}
CONTENT_CATS = {"oucontent", "resource", "subpage", "url", "page", "ouwiki",
                "glossary", "dataplus", "dualpane", "homepage", "htmlactivity",
                "oucollaborate", "ouelluminate", "sharedsubpage"}


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


# ---------------------------------------------------------------- E1 / E2 / E3
def return_latency(frame: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First post-deadline active day, and whether the student ever returns."""
    out = frame.copy()
    out["return_day"] = out["post_first_active_day"]
    out["never_returns"] = (out["post_clicks"] == 0).astype(int)
    rows = [{"metric": "share_never_returns", "pairs": int(len(pairs)),
             "fail_minus_pass": float(
                 out.iloc[pairs["fail_index"]]["never_returns"].mean()
                 - out.iloc[pairs["pass_index"]]["never_returns"].mean())},
            {"metric": "return_day", "pairs": int(len(pairs)),
             "fail_minus_pass": float(
                 out.iloc[pairs["fail_index"]]["return_day"].mean()
                 - out.iloc[pairs["pass_index"]]["return_day"].mean())}]
    by_outcome = (out.groupby("final_result")
                  .agg(records=("id_student", "size"),
                       never_returns=("never_returns", "mean"),
                       median_return_day=("return_day", "median"),
                       p25_return_day=("return_day", lambda s: s.quantile(.25)),
                       p75_return_day=("return_day", lambda s: s.quantile(.75)))
                  .reset_index())
    return by_outcome, pd.DataFrame(rows)


def return_curve(frame: pd.DataFrame, post_days: int = 14) -> pd.DataFrame:
    """Cumulative share back by day t, by outcome (survival-style curve)."""
    rows = []
    for t in range(1, post_days + 1):
        back = frame["post_first_active_day"].le(t)
        for outcome, group in frame.groupby("final_result"):
            rows.append({"post_day": t, "outcome": outcome,
                         "records": int(len(group)),
                         "share_returned": float(back[group.index].mean())})
    return pd.DataFrame(rows)


def multi_deadline(master: pd.DataFrame, assessments: pd.DataFrame,
                   submissions: pd.DataFrame, post_days: int
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Anchor submission facts plus the spacing between consecutive TMAs.

    A first version treated "how many TMA deadlines fall inside the observation
    window" as a behavioural variable. That premise does not hold on OULAD: the
    smallest gap between two consecutive positive-weight TMA deadlines in any of
    the 19 presentations is 21 days, while the post window is 14 days, so the
    count is always exactly one. The spacing table below is what makes that
    visible; the degenerate derivatives were removed.
    """
    tma = assessments[assessments["assessment_type"].eq("TMA")
                      & assessments["weight"].gt(0)].copy()
    tma["date"] = pd.to_numeric(tma["date"], errors="coerce")
    tma = tma.dropna(subset=["date"]).sort_values(BATCH + ["date", "id_assessment"])
    tma["deadline_rank"] = tma.groupby(BATCH).cumcount() + 1

    gaps = []
    for (module, presentation), group in tma.groupby(BATCH):
        dates = group["date"].to_numpy(dtype=float)
        if len(dates) < 2:
            continue
        gaps.append({"code_module": module, "code_presentation": presentation,
                     "tma_deadlines": int(len(dates)),
                     "min_gap_days": float(np.min(np.diff(dates))),
                     "first_deadline": float(dates[0])})
    spacing = pd.DataFrame(gaps)
    if len(spacing):
        spacing["second_deadline_in_window"] = spacing["min_gap_days"] <= post_days

    sub = submissions[["id_assessment", "id_student", "date_submitted", "is_banked"]]
    sub = sub[sub["is_banked"].fillna(0).eq(0)]
    anchor = tma[tma["deadline_rank"].eq(1)][BATCH + ["id_assessment", "date"]]
    anchor = anchor.rename(columns={"date": "due_day"})
    merged = anchor.merge(sub, on="id_assessment", how="left", validate="one_to_many")

    first = master[KEY + ["due_day", "eligible", "final_result"]].dropna(subset=["due_day"])
    rows = []
    for (module, presentation), group in merged.groupby(BATCH):
        cohort = first[(first["code_module"] == module)
                       & (first["code_presentation"] == presentation)]
        if cohort.empty:
            continue
        base = float(cohort["due_day"].iloc[0])
        observed = group["date_submitted"].where(group["date_submitted"].le(base + post_days))
        piece = group[["code_module", "code_presentation", "id_student"]].copy()
        piece["anchor_due_day"] = base
        piece["anchor_submitted"] = observed.notna().astype(int)
        piece["anchor_days_late"] = (observed - base).clip(lower=0)
        rows.append(piece.drop_duplicates(KEY))
    if not rows:
        return spacing, pd.DataFrame(), pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    agg = (long.groupby(KEY)
           .agg(anchor_submitted=("anchor_submitted", "max"),
                anchor_days_late=("anchor_days_late", "max"))
           .reset_index())
    return spacing, long, agg


def composition_shift(frame: pd.DataFrame, daily: pd.DataFrame,
                      catalog: pd.DataFrame, post_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre/post share of assessment, forum and content clicks per record."""
    meta = catalog[BATCH + ["id_site", "activity_type"]]
    events = frame[KEY].merge(daily, on=KEY, how="left")
    # Rebuild the pre window with types; the post window comes from S04 shares.
    rows = []
    pre = frame["pre_clicks"].to_numpy(dtype=float)
    post = frame["post_clicks"].to_numpy(dtype=float)
    shares = {c: frame[c].to_numpy(dtype=float)
              for c in frame.columns if c.startswith("post_resource_share__")}

    def post_share(names):
        return sum(shares.get(f"post_resource_share__{n}", np.zeros(len(frame))) for n in names)

    out = frame[KEY + ["final_result", "pre_clicks", "post_clicks"]].copy()
    out["post_assess_share"] = post_share(ASSESS_CATS)
    out["post_forum_share"] = post_share(FORUM_CATS)
    out["post_content_share"] = post_share(CONTENT_CATS)
    out["post_other_share"] = 1 - (out["post_assess_share"] + out["post_forum_share"]
                                  + out["post_content_share"])
    rows.append({"note": "pre-window composition needs the raw event stream",
                 "records": int(len(out)),
                 "mean_post_assess_share": float(out["post_assess_share"].mean()),
                 "mean_post_forum_share": float(out["post_forum_share"].mean()),
                 "mean_post_content_share": float(out["post_content_share"].mean())})
    summary = (out.groupby("final_result")
               .agg(records=("id_student", "size"),
                    assess=("post_assess_share", "mean"),
                    forum=("post_forum_share", "mean"),
                    content=("post_content_share", "mean"),
                    other=("post_other_share", "mean")).reset_index())
    return summary, out


# -------------------------------------------------------------------------- E4
def volume_patterns(post: np.ndarray, y: np.ndarray, length: int = 3
                    ) -> pd.DataFrame:
    """Discriminative contiguous day patterns of coarse volume symbols.

    Symbols per day: 0 = no clicks, 1 = below the record's own positive median,
    2 = at or above it.  This is a small, interpretable substitute for full
    sequential pattern mining on a 14-symbol alphabet.
    """
    positive = np.where(post > 0, post, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(positive, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    symbols = np.zeros_like(post, dtype=int)
    symbols[(post > 0) & (post < medians[:, None])] = 1
    symbols[(post > 0) & (post >= medians[:, None])] = 2
    rows = []
    n_days = post.shape[1]
    for start in range(0, n_days - length + 1):
        chunk = symbols[:, start:start + length]
        codes = chunk[:, 0] * 9 + chunk[:, 1] * 3 + chunk[:, 2]
        for code in np.unique(codes):
            key = f"{code // 9}{code % 9 // 3}{code % 3}"
            mask = codes == code
            n1, n0 = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
            if n1 + n0 < 50:
                continue
            p1 = n1 / max(int((y == 1).sum()), 1)
            p0 = n0 / max(int((y == 0).sum()), 1)
            rows.append({"pattern": key, "start_day": start + 1,
                         "support_fail": p1, "support_pass": p0,
                         "lift": (p1 + 1e-9) / (p0 + 1e-9),
                         "n_fail_with": n1, "n_pass_with": n0})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Aggregate the same pattern across start positions.
    agg = (table.groupby("pattern")
           .agg(mean_support_fail=("support_fail", "mean"),
                mean_support_pass=("support_pass", "mean"),
                max_abs_lift=("lift", lambda s: float(np.max(np.abs(np.log(s + 1e-9))))),
                best_start_day=("support_fail", "idxmax"),
                occurrences=("pattern", "size"))
           .reset_index())
    agg["log_lift"] = np.log((agg["mean_support_fail"] + 1e-9)
                             / (agg["mean_support_pass"] + 1e-9))
    return agg.sort_values("log_lift")


def run_structure(pairs: pd.DataFrame, post: np.ndarray) -> pd.DataFrame:
    """Run-length structure of the post-deadline activity stream per pair."""
    active = post > 0
    runs, longest_on, longest_off = [], [], []
    for row in active:
        change = np.flatnonzero(np.diff(row.astype(int))) + 1
        lengths = np.diff(np.concatenate([[0], change, [len(row)]]))
        # The flag for each run is the value at the run's FIRST position, so the
        # flag vector must start with row[0] rather than end with row[-1].
        run_active = np.concatenate([[bool(row[0])], row[change]])
        runs.append(len(lengths))
        longest_on.append(max([l for l, on in zip(lengths, run_active) if on] or [0]))
        longest_off.append(max([l for l, on in zip(lengths, run_active) if not on] or [0]))
    runs = np.asarray(runs, dtype=float)
    longest_on = np.asarray(longest_on, dtype=float)
    longest_off = np.asarray(longest_off, dtype=float)
    fail = pairs["fail_index"].to_numpy()
    passing = pairs["pass_index"].to_numpy()
    rows = []
    for name, values in (("n_runs", runs), ("longest_active_run", longest_on),
                         ("longest_inactive_run_direct", longest_off)):
        rows.append({"feature": name, "pairs": int(len(pairs)),
                     "fail_mean": float(values[fail].mean()),
                     "pass_mean": float(values[passing].mean()),
                     "fail_minus_pass": float((values[fail] - values[passing]).mean())})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- E5
BACKGROUND_CAT = ["gender", "region", "highest_education", "imd_band", "age_band",
                  "disability"]
BACKGROUND_NUM = ["num_of_prev_attempts", "studied_credits", "date_registration"]


def design(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    cats = [c for c in columns if c in BACKGROUND_CAT]
    nums = [c for c in columns if c not in cats]
    parts = [frame[nums].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)]
    names = list(nums)
    if cats:
        dummies = pd.get_dummies(frame[cats].astype(str), dummy_na=True, dtype=float)
        parts.append(dummies.to_numpy(dtype=float))
        names += list(dummies.columns)
    return np.column_stack(parts), names


def fit_predict(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=42))
    model.fit(X[tr], y[tr])
    return model.predict_proba(X[te])[:, 1]


def topk_precision(y: np.ndarray, p: np.ndarray, k: float = 0.05) -> float:
    n = max(int(round(len(y) * k)), 1)
    order = np.argsort(-p)
    boundary = p[order[n - 1]]
    above = p > boundary
    ties = p == boundary
    slots = n - int(above.sum())
    hit = y[above].sum() + (y[ties].mean() if ties.any() and slots > 0 else 0) * slots
    return float(hit / n)


def data_minimisation(frame: pd.DataFrame, behaviour: list[str], seed: int = 42,
                      repeats: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Behaviour-only vs behaviour+background under one fixed split."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    y = frame["y_fail"].to_numpy(dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(frame, y, frame["id_student"].to_numpy()))
    sets = {
        "behaviour": [c for c in behaviour if c in frame],
        "background": [c for c in BACKGROUND_CAT + BACKGROUND_NUM if c in frame],
        "all": [c for c in behaviour if c in frame]
               + [c for c in BACKGROUND_CAT + BACKGROUND_NUM if c in frame],
    }
    preds, rows = {}, []
    for name, cols in sets.items():
        X, names = design(frame, cols)
        p = fit_predict(X, y, tr, te)
        preds[name] = p
        rows.append({"feature_set": name, "n_features": len(names),
                     "n_test": int(len(te)),
                     "average_precision": float(average_precision_score(y[te], p)),
                     "roc_auc": float(roc_auc_score(y[te], p)),
                     "brier": float(brier_score_loss(y[te], p)),
                     "top5_precision": topk_precision(y[te], p, 0.05),
                     "top10_precision": topk_precision(y[te], p, 0.10),
                     "no_skill": float(y[te].mean())})
    table = pd.DataFrame(rows)

    rng = np.random.default_rng(seed + 11)
    groups = frame["id_student"].to_numpy()[te]
    members = list(pd.Series(np.arange(len(te))).groupby(groups).indices.values())
    deltas = {"ap_all_minus_behaviour": [], "auc_all_minus_behaviour": [],
              "ap_behaviour_minus_background": [], "auc_behaviour_minus_background": []}
    for _ in range(repeats):
        pick = np.concatenate([members[i] for i in rng.integers(len(members),
                                                              size=len(members))])
        if len(np.unique(y[te][pick])) < 2:
            continue
        yy = y[te][pick]
        deltas["ap_all_minus_behaviour"].append(
            average_precision_score(yy, preds["all"][pick])
            - average_precision_score(yy, preds["behaviour"][pick]))
        deltas["auc_all_minus_behaviour"].append(
            roc_auc_score(yy, preds["all"][pick]) - roc_auc_score(yy, preds["behaviour"][pick]))
        deltas["ap_behaviour_minus_background"].append(
            average_precision_score(yy, preds["behaviour"][pick])
            - average_precision_score(yy, preds["background"][pick]))
        deltas["auc_behaviour_minus_background"].append(
            roc_auc_score(yy, preds["behaviour"][pick]) - roc_auc_score(yy, preds["background"][pick]))
    ci = pd.DataFrame([{"contrast": k, "mean": float(np.mean(v)),
                        "ci_low": float(np.quantile(v, .025)),
                        "ci_high": float(np.quantile(v, .975)),
                        "replicates": len(v)}
                       for k, v in deltas.items() if v])

    # Subgroup audit on the fixed test split.
    sub_rows = []
    for attr in ("gender", "imd_band", "highest_education", "disability"):
        if attr not in frame:
            continue
        values = frame[attr].astype(str).to_numpy()[te]
        for level in pd.unique(values):
            mask = values == level
            if mask.sum() < 100 or len(np.unique(y[te][mask])) < 2:
                continue
            for name in ("behaviour", "all"):
                sub_rows.append({"attribute": attr, "subgroup": level,
                                 "feature_set": name, "n": int(mask.sum()),
                                 "at_risk_share": float(y[te][mask].mean()),
                                 "roc_auc": float(roc_auc_score(y[te][mask],
                                                                preds[name][mask])),
                                 "average_precision": float(
                                     average_precision_score(y[te][mask], preds[name][mask]))})
    sub = pd.DataFrame(sub_rows)
    if len(sub):
        # Gap per attribute between the two largest subgroups, behaviour vs all.
        # No p-value is attached here: the subgroup sizes are small and the gaps
        # are descriptive. ``verify_subgroup_mde.py`` adds a student-bootstrap
        # interval for the same contrasts, which is the number to quote.
        gaps = []
        for attr, group in sub.groupby("attribute"):
            levels = group.groupby("subgroup")["n"].sum().sort_values(ascending=False).index[:2]
            for name in ("behaviour", "all"):
                sel = group[group["subgroup"].isin(levels) & group["feature_set"].eq(name)]
                if len(sel) == 2:
                    gaps.append({"attribute": attr, "feature_set": name,
                                 "subgroups": " vs ".join(sel["subgroup"].tolist()),
                                 "auc_gap": float(sel["roc_auc"].iloc[0] - sel["roc_auc"].iloc[1])})
        if gaps:
            gap_table = pd.DataFrame(gaps)
            gap_table["abs_gap"] = gap_table["auc_gap"].abs()
            gap_table["note"] = "descriptive; see verify_subgroup_mde.py for intervals"
            sub = sub.merge(gap_table[["attribute", "feature_set", "auc_gap", "note"]],
                            on=["attribute", "feature_set"], how="left")
    return table, pd.concat([ci.assign(kind="delta"), sub.assign(kind="subgroup")],
                            ignore_index=True)


# ------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    post_days = int(cfg["study"]["post_days"])

    master = pd.read_parquet(root / "data" / "interim" / "master.parquet")
    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    catalog = pd.read_csv(root / "data" / "raw" / "oulad" / "vle.csv", na_values=["?"])
    assessments = pd.read_csv(root / "data" / "raw" / "oulad" / "assessments.csv",
                              na_values=["?"])
    submissions = pd.read_csv(root / "data" / "raw" / "oulad" / "studentAssessment.csv",
                              na_values=["?"])
    cohort = binary_cohort(master)
    lookup = cohort[KEY + ["due_day"]]
    daily_events = daily.daily_events(root / "data" / "raw" / "oulad" / "studentVle.csv",
                                      lookup, pre_days=14, post_days=post_days)
    index = pd.MultiIndex.from_frame(frame[KEY])
    post = daily.click_matrix(daily_events, index, 1, post_days)

    pairs = pairing.make_pairs(frame, 0.05, cfg["seed"])
    y = frame["y_fail"].to_numpy(dtype=int)
    summary = {"records": int(len(frame)), "fail": int(y.sum()),
               "pairs": int(len(pairs)),
               "config_hash": cfg.hash_ref,
               "post_days": post_days}

    # E1
    by_outcome, latency = return_latency(frame, pairs)
    by_outcome.to_csv(out / "e1_return_by_outcome.csv", index=False)
    latency.to_csv(out / "e1_return_latency_pairs.csv", index=False)
    curve = return_curve(frame, post_days)
    curve.to_csv(out / "e1_return_curve.csv", index=False)

    # E2
    spacing, long, agg = multi_deadline(master, assessments, submissions, post_days)
    spacing.to_csv(out / "e2_deadline_spacing.csv", index=False)
    long.to_csv(out / "e2_anchor_events.csv", index=False)
    merged = frame.merge(agg, on=KEY, how="left", validate="one_to_one")
    e2_cols = ["anchor_submitted", "anchor_days_late"]
    merged[e2_cols] = merged[e2_cols].fillna(0)
    e2_rows = []
    for name in e2_cols:
        fail = merged.iloc[pairs["fail_index"]][name].to_numpy(dtype=float)
        passing = merged.iloc[pairs["pass_index"]][name].to_numpy(dtype=float)
        e2_rows.append({"feature": name, "pairs": int(len(pairs)),
                        "fail_mean": float(fail.mean()),
                        "pass_mean": float(passing.mean()),
                        "fail_minus_pass": float((fail - passing).mean())})
    pd.DataFrame(e2_rows).to_csv(out / "e2_anchor_pairs.csv", index=False)
    (merged.groupby("final_result")[e2_cols].mean().reset_index()
     .to_csv(out / "e2_anchor_by_outcome.csv", index=False))
    merged[KEY + ["final_result"] + e2_cols].to_parquet(
        out / "e2_anchor_records.parquet", index=False)

    # E3
    comp, comp_records = composition_shift(frame, daily_events, catalog, post_days)
    comp.to_csv(out / "e3_composition_by_outcome.csv", index=False)
    (frame[KEY + ["final_result"]]
     .merge(comp_records, on=KEY + ["final_result"], how="left")
     .drop(columns=["final_result"]).head(0))  # schema only
    comp_records[KEY + ["post_assess_share", "post_forum_share",
                        "post_content_share"]].to_parquet(
        out / "e3_composition_records.parquet", index=False)

    # E4
    patterns = volume_patterns(post, y)
    patterns.to_csv(out / "e4_volume_patterns.csv", index=False)
    structure = run_structure(pairs, post)
    structure.to_csv(out / "e4_run_structure_pairs.csv", index=False)

    # E5
    behaviour = [c for c in frame.columns
                 if c.startswith(("post_", "pre_", "last3", "log_", "submitted",
                                 "days_", "relative_submission"))]
    ablation, extra = data_minimisation(frame, behaviour, cfg["seed"], args.repeats)
    ablation.to_csv(out / "e5_data_minimisation.csv", index=False)
    extra.to_csv(out / "e5_bootstrap_and_subgroups.csv", index=False)

    summary.update({
        "e1_never_returns_fail_minus_pass": float(latency.iloc[0]["fail_minus_pass"]),
        "e2_min_gap_between_tma_days": float(spacing["min_gap_days"].min())
        if len(spacing) else None,
        "e2_presentations_with_second_deadline_in_window": int(
            spacing["second_deadline_in_window"].sum()) if len(spacing) else 0,
        "e2_anchor_days_late_fail_minus_pass": float(e2_rows[-1]["fail_minus_pass"]),
        "e4_patterns_mined": int(len(patterns)),
        "e5_ap_all": float(ablation.set_index("feature_set").loc["all", "average_precision"]),
        "e5_ap_behaviour": float(ablation.set_index("feature_set").loc["behaviour", "average_precision"]),
        "e5_ap_background": float(ablation.set_index("feature_set").loc["background", "average_precision"]),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps({
        "config_sha256": cfg.sha256,
        "inputs_sha256": {name: sha256(root / "data" / "raw" / "oulad" / f"{name}.csv")
                          for name in ("assessments", "studentAssessment", "studentVle",
                                       "studentInfo", "studentRegistration", "vle")},
        "script_sha256": sha256(Path(__file__)),
        "limitations": ("exploratory; reuses the frozen cohort and pairs; repeated test "
                        "use; no causal claim; background features are administratively "
                        "collected and not available to a purely behavioural model"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
