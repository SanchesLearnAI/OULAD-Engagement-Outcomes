"""S12 report, poster, README, zip and citation checks."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_root

MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
SPLITS = ["main_test", "group_kfold", "temporal", "lomo"]
REJECTED_PHRASES = ["0.58--0.75", "5,040", "35/53", "每 15 分钟", "6 分钟",
                    "assessment 点击是已证实泄漏通道", "time-on-task 系数反号"]
REJECTED_NUMBERS = ["0.626", "0.663", "0.636", "69.69", "95.23"]
CITATION_TOKENS = ("et al", "doi", "arxiv", "reported", "study", "paper",
                   "literature", "LEAP", "Wu", "Aljohani")


def f(x, n=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "--"
        return f"{float(x):.{n}f}"
    except Exception:
        return "--"


def pct(x, n=1):
    try:
        if x is None or not np.isfinite(float(x)):
            return "--"
        return f"{100*float(x):.{n}f}\\%"
    except Exception:
        return "--"


def _tex(s):
    if s is None:
        return "--"
    s = str(s)
    # Protect original backslashes so the braces added by \textbackslash{}
    # are not themselves escaped by the brace rules below.
    s = s.replace("\\", "\x00")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("\x00", r"\textbackslash{}")
    return s


def load_tables(ctx):
    t = {}
    for p in sorted(ctx.paths.tables_dir.glob("*.csv")):
        try:
            t[p.stem] = pd.read_csv(p)
        except Exception:
            pass
    stages = {}
    for p in sorted((ctx.run_dir / "stages").glob("S*.stage.json")):
        try:
            stages[p.stem.split(".")[0]] = json.loads(p.read_text())
        except Exception:
            pass
    return t, stages


def _table(headers, rows, colspec, caption, label, size="\\small"):
    out = [r"\begin{table}[t]", r"\centering", size,
           r"\begin{tabular}{" + colspec + "}", r"\toprule",
           " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}",
            r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(out)


def _fig(rel, caption, label, width="0.86\\linewidth"):
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=" + width + "]{" + rel + "}",
        r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure}",
    ])


def _mv(models, model, split, metric):
    r = models[(models["model"] == model) & (models["split"] == split) &
               (models["metric"] == metric)]
    return float(r["value"].iloc[0]) if len(r) else float("nan")


def build_latex(ctx) -> str:
    t, stages = load_tables(ctx)
    draft = ctx.mode == "simulation"
    flow = t.get("t02_cohort_flow", pd.DataFrame())
    c28 = flow[flow["cutoff"] == 28].iloc[0] if len(flow) else None
    models = t.get("t08_model_metrics", pd.DataFrame())
    split = t.get("t05_split_summary", pd.DataFrame())

    def srow(name):
        r = split[split["split"] == name]
        return r.iloc[0] if len(r) else None

    main_row, temp_row = srow("main_train_test"), srow("temporal")
    lomo_rows = split[split["split"].astype(str).str.startswith("lomo_")]

    s01m = stages.get("S01", {}).get("metrics", {})
    s01_counts = {k: int(s01m[k]) for k in ("Pass", "Fail", "Withdrawn", "Distinction")
                  if k in s01m}
    if draft and c28 is not None:
        n_total = int(c28["n_total"])
        n_t1 = int(c28["n_t1"])
        n_passdist = int(c28["n_t3_pass"])
        n_fail = int(c28["n_t3_fail"])
        n_withdrawn = int(c28["n_withdrawn"])
        base_t1 = float(c28["base_rate_t1"])
    elif s01_counts:
        n_total = int(sum(s01_counts.values()))
        n_passdist = int(s01_counts.get("Pass", 0)) + int(s01_counts.get("Distinction", 0))
        n_t1 = n_passdist + int(s01_counts.get("Fail", 0))
        n_fail = int(s01_counts.get("Fail", 0))
        n_withdrawn = int(s01_counts.get("Withdrawn", 0))
        base_t1 = n_fail / max(n_t1, 1)  # positive (Fail) share
    else:
        n_total = int(c28["n_total"]) if c28 is not None else 0
        n_t1 = int(c28["n_t1"]) if c28 is not None else 0
        n_fail = int(c28["n_t3_fail"]) if c28 is not None else 0
        n_passdist = max(n_t1 - n_fail, 0)
        base_t1 = float(c28["base_rate_t1"]) if c28 is not None else float("nan")
        n_withdrawn = int(c28["n_withdrawn"]) if c28 is not None else 0

    s02 = stages.get("S02", {}).get("metrics", {})
    audit = s02.get("withdrawal", s02)
    s04 = stages.get("S04", {}).get("metrics", {})
    s06 = stages.get("S06", {}).get("metrics", {})
    s07 = stages.get("S07", {}).get("metrics", {})
    s11 = stages.get("S11", {}).get("metrics", {})
    s05 = stages.get("S05", {}).get("metrics", {})
    leak_students = s05.get("students_in_train_and_test")
    leak_share = s05.get("test_row_overlap_share")
    leak_known = leak_students is not None and leak_share is not None
    if draft:
        leak_txt = ("in this rehearsal all grouped splits have zero student overlap")
    elif leak_known:
        leak_txt = (f"here a naive row-level 60/20/20 split would place {int(leak_students)} "
                    f"students in both train and test ({pct(leak_share)} of test rows), "
                    "while every grouped split has zero student overlap")
    else:
        leak_txt = "every grouped split has zero student overlap"

    # Model metrics table.
    rows = []
    for metric, lab in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                        ("recall", "Recall at val threshold"),
                        ("precision", "Precision at val threshold"),
                        ("f1", "F1"), ("brier", "Brier"),
                        ("calibration_slope", "Calibration slope")]:
        rows.append([lab] + [f(_mv(models, m, "main_test", metric)) for m in MODELS])
    main_rows = (models[models["split"] == "main_test"] if len(models) else models)
    ref_ns = (float(main_rows["no_skill_pr_auc"].iloc[0])
              if len(main_rows) and "no_skill_pr_auc" in main_rows else float("nan"))
    ref_f1 = (float(main_rows["all_positive_f1"].iloc[0])
              if len(main_rows) and "all_positive_f1" in main_rows else float("nan"))
    thr_rules = (sorted(main_rows["threshold_rule"].dropna().astype(str).unique())
                 if len(main_rows) and "threshold_rule" in main_rows else [])
    model_table = _table(
        ["Metric"] + MODELS, rows, "l" + "c" * len(MODELS),
        "Main held-out test (grouped split), Fail is the positive class. Reference points: "
        f"no-skill PR-AUC {f(ref_ns)}; F1 of predicting every enrolment as Fail {f(ref_f1)}. "
        "Threshold rule used: " + (_tex(", ".join(thr_rules)) if thr_rules else "--")
        + ". M4 contains scores and is an upper bound only.",
        "tab:models")

    # External validation table.
    ext_rows = []
    for m in ("M0", "M1", "M3", "M5"):
        ext_rows.append([m,
                         f(_mv(models, m, "group_kfold", "pr_auc")),
                         f(_mv(models, m, "group_kfold", "roc_auc")),
                         f(_mv(models, m, "temporal", "pr_auc")),
                         f(_mv(models, m, "temporal", "roc_auc")),
                         f(_mv(models, m, "lomo", "pr_auc")),
                         f(_mv(models, m, "lomo", "roc_auc"))])
    ext_table = _table(
        ["Model", "CV PR", "CV ROC", "2013→2014 PR", "2013→2014 ROC",
         "LOMO PR", "LOMO ROC"], ext_rows, "lcccccc",
        "Grouped CV, forward-temporal and leave-one-module-out performance.",
        "tab:external")

    # Cohort + split tables.
    flow_rows = []
    for c in (7, 28, 56):
        r = flow[flow["cutoff"] == c]
        if len(r):
            r = r.iloc[0]
            row_t1 = int(r["n_t1"]) if draft else int(n_t1)
            row_base = float(r["base_rate_t1"]) if draft else float(base_t1)
            row_pass = int(r["n_t3_pass"]) if draft else int(n_passdist)
            row_fail = int(r["n_t3_fail"]) if draft else int(n_fail)
            flow_rows.append([str(c), str(int(r["n_total"])), str(row_t1),
                              pct(row_base), str(row_pass),
                              str(row_fail), str(int(r["n_t3_withdrawn"]))])
    flow_caption = (
        "Cohort flow in the 500-enrolment rehearsal sample. T1 excludes Withdrawn; "
        "T3 reports sample Pass/Dist and Fail counts at cutoff c, with Withdrawn observed "
        "by cutoff c. At-risk filtering is a sensitivity analysis, not the main cohort."
        if draft else
        "Cohort flow at three cutoffs. T1 excludes Withdrawn and is not filtered by "
        "date\\_unregistration in the main analysis, so its counts are constant across cutoffs. "
        "T3 reports final full-cohort Pass/Dist and Fail counts separately from Withdrawn "
        "observed by cutoff c; the columns are descriptive, not a fixed-size partition. At-risk filtering "
        "is a sensitivity analysis, not the main cohort.")
    flow_table = _table(
        ["Cutoff", "Enrolments", "T1 n", "T1 Fail rate",
         "T3 Pass/Dist", "T3 Fail", "T3 Withdrawn at c"],
        flow_rows, "rcccccc", flow_caption,
        "tab:cohort", size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    split_rows = []
    if main_row is not None:
        split_rows.append(["Grouped 60/20/20", str(int(main_row["n_train"])),
                           str(int(main_row["n_test"])),
                           pct(main_row["train_positive_rate"]),
                           pct(main_row["test_positive_rate"]),
                           str(int(main_row["student_overlap"]))])
    if temp_row is not None:
        split_rows.append(["2013→2014", str(int(temp_row["n_train"])),
                           str(int(temp_row["n_test"])),
                           pct(temp_row["train_positive_rate"]),
                           pct(temp_row["test_positive_rate"]),
                           str(int(temp_row["student_overlap"]))])
    if len(lomo_rows):
        split_rows.append(["LOMO (7 folds)", f(lomo_rows["n_train"].mean(), 1),
                           f(lomo_rows["n_test"].mean(), 1), "--", "--",
                           str(int(lomo_rows["student_overlap"].max()))])
    split_table = _table(
        ["Split", "Train n", "Test n", "Train pos.", "Test pos.", "Student overlap"],
        split_rows, "lccccc", "Held-out designs. Student overlap is zero by construction.",
        "tab:splits")

    # Cutoff table.
    rec = t.get("t09_cutoff_metrics", pd.DataFrame())
    cut_rows = []
    for c in (7, 14, 21, 28, 35, 42, 49, 56):
        r = rec[(rec["cutoff"] == c) & (rec["cohort"] == "fixed") &
                (rec["model"] == "M3")]
        if len(r):
            r = r.iloc[0]
            cut_rows.append([str(c), f(r["pr_auc"]), f(r["roc_auc"]),
                             f(r["recall"]), f(r["precision"]), f(r["brier"])])
    cutoff_table = _table(
        ["Cutoff", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Brier"],
        cut_rows, "rccccc", "M3 performance as the observation window grows (fixed cohort).",
        "tab:cutoff")

    # Updating table.
    inc = t.get("t09_incremental", pd.DataFrame())
    u_rows = []
    for _, r in inc[(inc["cutoff"] == 56) & (inc["cohort"] == "fixed")].iterrows():
        if r["method"] in ("W1_only_M3", "W2_only", "All_features_c56",
                           "B1_stacking"):
            u_rows.append([_tex(r["method"]), f(r["pr_auc"]),
                           f(r["delta_pr_auc_vs_28"]),
                           "[" + f(r["ci_low"]) + ", " + f(r["ci_high"]) + "]",
                           f(r["brier"]), f(r["calibration_slope"]),
                           _tex(r.get("status", "ok"))])
    _b2_tab = inc[(inc["cutoff"] == 56) & (inc["cohort"] == "fixed") &
                  (inc["method"] == "B2_offset")] if len(inc) else pd.DataFrame()
    _b2s = (str(_b2_tab["status"].iloc[0])
            if len(_b2_tab) and "status" in _b2_tab else "unknown")
    upd_caption = ("W1→W2 updating at day 56. The main table contains the four valid "
                   "methods; B2 offset is excluded as a failed specification")
    if len(_b2_tab):
        upd_caption += " and is reported in Appendix Table~\\ref{tab:b2_failed}"
    upd_caption += ". A CI spanning zero means no demonstrated increment."
    upd_table = _table(
        ["Method", "PR-AUC", "Δ vs W1", "95\\% CI", "Brier", "Cal. slope", "Status"],
        u_rows, "lcccccc", upd_caption, "tab:updating")
    b2_rows = []
    for _, r in _b2_tab.iterrows():
        b2_rows.append([_tex(r["method"]), f(r["pr_auc"]),
                        f(r["delta_pr_auc_vs_28"]),
                        "[" + f(r["ci_low"]) + ", " + f(r["ci_high"]) + "]",
                        f(r["brier"]), f(r["calibration_slope"]),
                        _tex(r.get("status", "unknown"))])
    b2_caption = ("B2 offset at day 56. B2 is a failed calibration specification")
    if len(_b2_tab):
        b2_caption += (f" (calibration slope {f(_b2_tab['calibration_slope'].iloc[0])}, "
                       f"Brier {f(_b2_tab['brier'].iloc[0])})")
    b2_caption += "; it is not evidence about sequential updating."
    b2_table = _table(
        ["Method", "PR-AUC", "Δ vs W1", "95\\% CI", "Brier", "Cal. slope", "Status"],
        b2_rows, "lcccccc", b2_caption, "tab:b2_failed")

    # Three-class table.
    cm = t.get("t10_class_metrics", pd.DataFrame())
    t3_rows = []
    r28 = cm[cm["cutoff"] == 28] if len(cm) else pd.DataFrame()
    t3_28 = r28
    for _, r in r28.iterrows():
        t3_rows.append([_tex(r["class"]), f(r["precision"]), f(r["recall"]),
                        f(r["f1"]), str(int(r["support"])), f(r["macro_f1"])])
    t3_table = _table(["Class", "Precision", "Recall", "F1", "Support", "Macro F1"],
                      t3_rows, "lccccc", "Three-class outcome at day 28 on the held-out test (Pass/Distinction, Fail, Withdrawn).",
                      "tab:threeclass")

    # Robustness table.
    specs = t.get("t11_specs", pd.DataFrame())
    nc = t.get("t11_negative_controls", pd.DataFrame())
    pred = specs[specs["family"] == "predictive"] if len(specs) else pd.DataFrame()
    spec_line = "no specification table available"
    if len(pred):
        spec_line = (f"{len(pred)} predictive specifications; PR-AUC median "
                     f"{f(pred['metric_value'].median())}, range "
                     f"{f(pred['metric_value'].min())}–{f(pred['metric_value'].max())}; "
                     f"share above the baseline {pct((pred['metric_value'] >
                       pred[pred['spec_id']=='P000_baseline']['metric_value'].iloc[0]).mean())}.")
    placebo_expo = t.get("t11_placebo_exposure", pd.DataFrame())
    nc_rows = []
    for _, r in nc.iterrows():
        role = str(r.get("role", ""))
        if not role:
            role = ("leak_check" if r["control"] == "random_labels" else
                    "positive_control" if r["control"] == "pre_start_only" else
                    "placebo_failure" if "placebo" in str(r["control"]) else
                    "negative_control_outcome")
        interp = str(r["note"])
        if r["control"] == "random_labels":
            interp = "leakage sanity check; expected ~0.5"
        elif r["control"] == "pre_start_only":
            interp = "positive control for early signal; not a null"
        elif "placebo" in str(r["control"]):
            interp = "post-outcome persistence; placebo failure, not a leak test"
        elif r["control"] == "num_of_prev_attempts_gt0":
            interp = "negative-control outcome; behaviour may encode background"
        nc_rows.append([_tex(r["control"]), _tex(r["metric"]), f(r["value"]),
                        _tex(role), _tex(interp)])
    expo_pr = {}
    expo_roc = {}
    for _, r in placebo_expo.iterrows():
        nc_rows.append([_tex(r["exposure"]), _tex(r["metric"]), f(r["value"]),
                        "negative\\_control\\_exposure",
                        "non-assessment W2 change; if predictive, generic engagement/confounding"])
        if str(r["metric"]) == "pr_auc":
            expo_pr[str(r["exposure"])] = float(r["value"])
        elif str(r["metric"]) == "roc_auc":
            expo_roc[str(r["exposure"])] = float(r["value"])
    nc_table = _table(
        ["Diagnostic", "Metric", "Value", "Role", "Interpretation"],
        nc_rows, "llclp{0.30\\linewidth}",
        "Diagnostics and placebo checks. Roles distinguish leak checks, positive controls, "
        "placebo failures, negative-control outcomes, and negative-control exposures; "
        "non-null values are not clean nulls.",
        "tab:negctl", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}")

    context = t.get("t08_context_metrics", pd.DataFrame())
    ctxd = t.get("t08_context_delta", pd.DataFrame())
    ctxd_row = ctxd.iloc[0] if len(ctxd) else None
    ctx_rows = []
    for _, r in context.iterrows():
        ctx_rows.append([_tex(r["model"]), f(r["pr_auc"]), f(r["roc_auc"]),
                         f(r["recall"]), f(r["brier"]),
                         f(r["calibration_slope"])])
    ctx_caption = ("Context-only and context-adjusted models on the same held-out split. "
                   "Context variables are never part of the strict clickstream set.")
    if ctxd_row is not None:
        cap_zero = (float(ctxd_row["ci_low"]) <= 0 <= float(ctxd_row["ci_high"]))
        ctx_caption += (f" M1c--M0c paired differences (95\\% CI; PR interval "
                        f"{'includes' if cap_zero else 'excludes'} zero): "
                        f"ΔPR-AUC {f(ctxd_row['delta_pr_auc'])} "
                        f"[{f(ctxd_row['ci_low'])}, {f(ctxd_row['ci_high'])}]; "
                        f"ΔROC-AUC {f(ctxd_row['delta_roc_auc'])} "
                        f"[{f(ctxd_row['roc_ci_low'])}, {f(ctxd_row['roc_ci_high'])}] "
                        f"({int(ctxd_row['n_test'])} test rows, 1,000 paired bootstraps).")
    ctx_table = _table(["Model", "PR-AUC", "ROC-AUC", "Recall", "Brier", "Cal. slope"],
                       ctx_rows, "lccccc", ctx_caption, "tab:context")
    traj = t.get("t12_trajectory_weekly", pd.DataFrame())
    div = t.get("t12_divergence", pd.DataFrame())
    lm = t.get("t10_landmark_cox", pd.DataFrame())
    quiz = t.get("t11_quiz_sensitivity", pd.DataFrame())
    expo = t.get("t11_exposure_sensitivity", pd.DataFrame())
    bins = t.get("t11_matching_bins", pd.DataFrame())
    placebo = t.get("t11_placebo_outcomes", pd.DataFrame())
    modm = t.get("t08_module_metrics", pd.DataFrame())
    subm = t.get("t08_subgroup_metrics", pd.DataFrame())
    t3_n = int(t3_28["n_test"].iloc[0]) if len(t3_28) else 0
    missing_unreg = int(stages.get("S10", {}).get("metrics", {}).get(
        "withdrawn_missing_unreg", 0))
    quiz_delta = ({str(r["model"]): float(r["delta_pr_auc"])
                   for _, r in quiz.iterrows()} if len(quiz) else {})
    expo_diff = (expo["d_clicks_diff"] if len(expo) else pd.Series(dtype=float))
    bins_diff = (bins["d_clicks_diff"] if len(bins) else pd.Series(dtype=float))
    placebo_roc = (float(placebo["roc_auc"].iloc[0]) if len(placebo) else float("nan"))

    mod_rows = []
    for _, r in (modm[modm["group_type"] == "module"].iterrows()
                 if len(modm) else []):
        mod_rows.append([_tex(r["group"]), str(int(r["n"])), f(r["pr_auc"]),
                         f(r["roc_auc"]), f(r["recall"]), f(r["brier"]),
                         f(r["calibration_slope"]), "yes" if r["low_n"] else "no"])
    mod_table = _table(["Module", "n", "PR-AUC", "ROC-AUC", "Recall", "Brier",
                        "Cal. slope", "low n"], mod_rows, "lccccccc",
                       "Per-module held-out metrics for M3. Groups with fewer than five test "
                       "rows or a single outcome class are omitted, so counts need not sum to "
                       "the test n; low n = fewer than 200 test rows.", "tab:module")
    sub_rows = []
    for _, r in (subm.iterrows() if len(subm) else []):
        sub_rows.append([_tex(r["group_type"]), _tex(r["group"]), str(int(r["n"])),
                         f(r["pr_auc"]), f(r["roc_auc"]), f(r["recall"]),
                         f(r["calibration_slope"]), "yes" if r["low_n"] else "no"])
    sub_table = _table(["Group type", "Group", "n", "PR-AUC", "ROC-AUC",
                        "Recall", "Cal. slope", "low n"], sub_rows, "llcccccc",
                       "Subgroup held-out metrics for M3. Groups with fewer than five test "
                       "rows, a single outcome class, or a missing value are omitted, so counts "
                       "need not sum to the test n; low n = fewer than 200 test rows.",
                       "tab:subgroup",
                       size="\\footnotesize\\setlength{\\tabcolsep}{3pt}")
    sens_rows = []
    for m, d in quiz_delta.items():
        sens_rows.append(["quiz", f"{m} strict vs included", f(d), "--", "--"])
    for _, r in (expo.iterrows() if len(expo) else []):
        sens_rows.append(["exposure", _tex(r["exposure"]),
                          f(r["d_clicks_diff"]),
                          "[" + f(r["d_clicks_ci_low"]) + ", " +
                          f(r["d_clicks_ci_high"]) + "]",
                          str(int(r["n_matched"]))])
    if len(bins_diff):
        sens_rows.append(["matching bins", "range over 7 bin grids",
                          f(bins_diff.min()) + " to " + f(bins_diff.max()), "--", "--"])
    if np.isfinite(placebo_roc):
        sens_rows.append(["repeater diagnostic", "repeater prediction ROC",
                          f(placebo_roc), "--", "--"])
    sens_table = _table(["Family", "Comparison", "Value", "95\\% CI", "n"],
                        sens_rows, "llccl",
                        "Measurement and matching sensitivity checks. Values are differences "
                        "or diagnostic metrics, not causal effects.", "tab:sensitivity")
    lm_rows = []
    if len(lm):
        g = lm.groupby(["landmark", "cause"], as_index=False).agg(
            n_train=("n_train", "first"), n_test=("n_test", "first"),
            events_train=("events_train", "first"),
            events_test=("events_test", "first"),
            test_concordance=("test_concordance", "first"))
        for _, r in g.iterrows():
            lm_rows.append([str(int(r["landmark"])), _tex(r["cause"]),
                            str(int(r["n_train"])), str(int(r["n_test"])),
                            str(int(r["events_train"])), str(int(r["events_test"])),
                            f(r["test_concordance"])])
    lm_table = _table(["Landmark", "Cause", "Train n", "Test n", "Events train",
                       "Events test", "Test C"], lm_rows, "rcccccc",
                      "Landmark cause-specific Cox. Features are available at the landmark; "
                      "concordance is computed on the held-out test split.", "tab:landmark")
    traj_rows = []
    for _, r in (traj.iterrows() if len(traj) else []):
        traj_rows.append([str(int(r["week"])), str(int(r["n_pass"])),
                          str(int(r["n_fail"])), f(r["mean_pass"]),
                          f(r["mean_fail"]), f(r["diff"]),
                          "[" + f(r["ci_low"]) + ", " + f(r["ci_high"]) + "]"])
    traj_table = _table(["Week", "n Pass", "n Fail", "Pass mean", "Fail mean",
                         "Diff", "95\\% CI"], traj_rows, "rcccccc",
                        "Matched weekly trajectory (log1p clicks), cluster-bootstrap CI.",
                        "tab:trajectory")
    lim_rows = [
        ["Leakage (future clicks/scores)", "Cutoff assertions, grouped splits, M4 upper bound",
         "Feature availability after marking remains unobserved for M4"],
        ["Confounding", "CEM on W1 variables, E-value, diagnostic checks",
         "W1 engagement is post-baseline; unmeasured confounding remains"],
        ["Measurement", "Multiple engagement operationalisations and quiz sensitivity",
         "Clicks are a weak proxy for study time"],
        ["External validity", "2013→2014 and LOMO splits", "Performance drops outside random CV"],
        ["Fairness", "Module/subgroup metrics and calibration", "Small subgroups; no causal claim"],
        ["Withdrawal", "Three-class model and cumulative incidence",
         f"{missing_unreg} Withdrawn enrolments lack date\\_unregistration"],
        ["T3 test size", "T3 split reported separately", f"T3 test n={t3_n}, not the T1 test n"],
    ]
    lim_table = _table(["Threat", "Mitigation", "Residual risk"], lim_rows,
                       "p{0.26\\linewidth}p{0.34\\linewidth}p{0.32\\linewidth}",
                       "Limitations and residual risks.", "tab:limitations")

    # Text numbers.
    overlap = t.get("t03_overlap", pd.DataFrame())
    ov_min = overlap["mean_overlap_coefficient"].min() if len(overlap) else float("nan")
    ov_max = overlap["mean_overlap_coefficient"].max() if len(overlap) else float("nan")
    zero_share = overlap.loc[overlap["cutoff"] == 28, "zero_click_share_t1"]
    zero_share = float(zero_share.iloc[0]) if len(zero_share) else float("nan")
    stab = t.get("t07_stability", pd.DataFrame())
    chosen_k = s07.get("k_chosen", "--")
    chosen_ari = s07.get("mean_ari", float("nan"))
    chosen_j = s07.get("min_cluster_jaccard", float("nan"))
    chosen_sil = s07.get("silhouette", float("nan"))
    def _stab_row(k):
        if not len(stab):
            return None
        r = stab[stab["k"] == k]
        return r.iloc[0] if len(r) else None
    k2row = _stab_row(2)
    k2_ari = float(k2row["mean_ari"]) if k2row is not None else float("nan")
    k2_j = float(k2row["min_cluster_jaccard"]) if k2row is not None else float("nan")
    k2_sil = float(k2row["silhouette"]) if k2row is not None else float("nan")
    max_sil = float(stab["silhouette"].max()) if len(stab) else float("nan")
    max_sil_k = int(stab.loc[stab["silhouette"].idxmax(), "k"]) if len(stab) else None
    use_k2_compare = bool(chosen_k != 2 and np.isfinite(k2_ari) and
                          np.isfinite(chosen_ari) and k2_ari > chosen_ari)
    pr_main = _mv(models, "M1", "main_test", "pr_auc")
    roc_main = _mv(models, "M1", "main_test", "roc_auc")
    roc_cv = _mv(models, "M1", "group_kfold", "roc_auc")
    roc_temp = _mv(models, "M1", "temporal", "roc_auc")
    roc_lomo = _mv(models, "M1", "lomo", "roc_auc")
    rand_roc = float(nc.loc[nc["control"].str.contains("random"), "value"].iloc[0]) if len(nc) else float("nan")
    cem_n = s06.get("n_matched", "--")
    cem_smd = s06.get("binned_max_abs_smd", float("nan"))
    raw_smd = s06.get("raw_max_abs_smd", float("nan"))
    pre_start_auc = float(nc.loc[nc["control"] == "pre_start_only", "value"].iloc[0]) if len(nc) else float("nan")
    placebo_auc = float(nc.loc[nc["control"].str.contains("placebo"), "value"].iloc[0]) if len(nc) else float("nan")
    main_pr = {m: _mv(models, m, "main_test", "pr_auc") for m in MODELS}
    valid_pr = {m: v for m, v in main_pr.items() if np.isfinite(v)}
    best_model = max(valid_pr, key=valid_pr.get) if valid_pr else "M1"
    best_pr = valid_pr.get(best_model, float("nan"))
    best_cal = _mv(models, best_model, "main_test", "calibration_slope")
    strict_pr = {m: _mv(models, m, "main_test", "pr_auc")
                 for m in ("M0", "M1", "M2", "M3", "M5")}
    best_strict = max(strict_pr, key=lambda m: strict_pr[m]
                      if np.isfinite(strict_pr[m]) else -1)
    best_strict_pr = strict_pr[best_strict]
    m5_temp = _mv(models, "M5", "temporal", "roc_auc")
    m5_lomo = _mv(models, "M5", "lomo", "roc_auc")
    m1_temp = _mv(models, "M1", "temporal", "roc_auc")
    m1_lomo = _mv(models, "M1", "lomo", "roc_auc")
    ladder = [main_pr.get(m, float("nan")) for m in MODELS[:4]]
    monotone = all(np.isfinite(ladder[i]) and np.isfinite(ladder[i+1]) and
                   ladder[i+1] >= ladder[i] for i in range(3))
    rec_fixed = rec[(rec["cohort"] == "fixed") & (rec["model"] == "M3")].sort_values("cutoff") \
        if len(rec) else pd.DataFrame()
    best_cut = int(rec_fixed.loc[rec_fixed["pr_auc"].idxmax(), "cutoff"]) if len(rec_fixed) else 28
    c28_pr = float(rec_fixed.loc[rec_fixed["cutoff"] == 28, "pr_auc"].iloc[0]) if len(rec_fixed[rec_fixed["cutoff"] == 28]) else float("nan")
    c56_pr = float(rec_fixed.loc[rec_fixed["cutoff"] == 56, "pr_auc"].iloc[0]) if len(rec_fixed[rec_fixed["cutoff"] == 56]) else float("nan")
    cut_delta = c56_pr - c28_pr
    cut_series = rec_fixed["pr_auc"].to_numpy() if len(rec_fixed) else np.array([])
    cut_monotone = bool(len(cut_series) > 1 and np.all(np.diff(cut_series) >= 0))
    inc56 = inc[(inc["cutoff"] == 56) & (inc["cohort"] == "fixed") &
                (inc["method"] != "W1_only_M3")] if len(inc) else pd.DataFrame()
    table_methods = ["W2_only", "All_features_c56", "B1_stacking"]
    inc56_tab = inc56[inc56["method"].isin(table_methods)] if len(inc56) else pd.DataFrame()
    valid_inc = inc56_tab
    pos_valid = (valid_inc[valid_inc["ci_low"] > 0]
                 if len(valid_inc) else pd.DataFrame())
    best_inc = (valid_inc.loc[valid_inc["pr_auc"].idxmax()]
                if len(valid_inc) else None)
    t3_28 = cm[cm["cutoff"] == 28] if len(cm) else pd.DataFrame()
    if len(t3_28):
        hardest = t3_28.loc[t3_28["f1"].idxmin()]
        hardest_class, hardest_f1 = str(hardest["class"]), float(hardest["f1"])
    else:
        hardest_class, hardest_f1 = "Withdrawn", float("nan")
    base_pr = float(pred[pred["spec_id"] == "P000_baseline"]["metric_value"].iloc[0]) if len(pred) and (pred["spec_id"] == "P000_baseline").any() else float("nan")
    _m0 = models[(models["model"] == "M0") & (models["split"] == "main_test") &
                 (models["metric"] == "pr_auc")] if len(models) else pd.DataFrame()
    no_skill = float(_m0["positive_rate"].iloc[0]) if len(_m0) else float("nan")
    pre_signal = np.isfinite(pre_start_auc) and np.isfinite(base_pr) and pre_start_auc > base_pr
    placebo_signal = np.isfinite(placebo_auc) and np.isfinite(base_pr) and placebo_auc > base_pr
    pre_above_noskill = np.isfinite(pre_start_auc) and np.isfinite(no_skill) and pre_start_auc > no_skill
    placebo_above_noskill = np.isfinite(placebo_auc) and np.isfinite(no_skill) and placebo_auc > no_skill
    cem_support = float(s06.get("common_support_treated", float("nan")))
    ev = float(s06.get("e_value", float("nan")))
    match_methods = specs[specs["family"] == "matching"] if len(specs) else pd.DataFrame()
    match_vals = ({str(r["method"]): float(r["metric_value"])
                   for _, r in match_methods.iterrows()} if len(match_methods) else {})
    _mvals = [match_vals.get(k) for k in ("CEM", "PSM", "stratified")]
    _mvals = [v for v in _mvals if v is not None and np.isfinite(v)]
    match_sign_stable = (len(_mvals) == 3 and
                         (all(v > 0 for v in _mvals) or all(v < 0 for v in _mvals)))
    div_week = (div["divergence_week"].iloc[0] if len(div) else None)
    div_week = None if (div_week is None or (isinstance(div_week, float) and
                                              not np.isfinite(div_week))) else int(div_week)
    t3_n = int(t3_28["n_test"].iloc[0]) if len(t3_28) else 0
    t4 = t.get("t04_feature_audit", pd.DataFrame())
    zero_all = (float(t4.loc[t4["cutoff"] == 28, "zero_click_share_w1"].iloc[0])
                if len(t4[t4["cutoff"] == 28]) else float("nan"))
    missing_unreg = int(stages.get("S10", {}).get("metrics", {}).get(
        "withdrawn_missing_unreg", 0))
    ctx_pr = ({str(r["model"]): float(r["pr_auc"]) for _, r in context.iterrows()}
              if len(context) else {})
    ctx_roc = ({str(r["model"]): float(r["roc_auc"]) for _, r in context.iterrows()}
               if len(context) else {})
    ctx_cal = ({str(r["model"]): float(r["calibration_slope"])
                for _, r in context.iterrows()} if len(context) else {})
    ctx_alone_pr = ctx_pr.get("M-1_context", float("nan"))
    ctx_alone_roc = ctx_roc.get("M-1_context", float("nan"))
    ctx_delta = (ctx_pr.get("M1c_context+M1", float("nan")) -
                 ctx_pr.get("M0c_context+M0", float("nan")))
    ctx_delta_lo = float(ctxd_row["ci_low"]) if ctxd_row is not None else float("nan")
    ctx_delta_hi = float(ctxd_row["ci_high"]) if ctxd_row is not None else float("nan")
    ctx_roc_delta = (float(ctxd_row["delta_roc_auc"])
                     if ctxd_row is not None else float("nan"))
    ctx_roc_lo = (float(ctxd_row["roc_ci_low"])
                  if ctxd_row is not None else float("nan"))
    ctx_roc_hi = (float(ctxd_row["roc_ci_high"])
                  if ctxd_row is not None else float("nan"))
    ctx_ci_includes_zero = (np.isfinite(ctx_delta_lo) and np.isfinite(ctx_delta_hi) and
                            ctx_delta_lo <= 0 <= ctx_delta_hi)
    ctx_roc_ci_includes_zero = (np.isfinite(ctx_roc_lo) and np.isfinite(ctx_roc_hi) and
                                ctx_roc_lo <= 0 <= ctx_roc_hi)
    ctx_at_chance = (not np.isfinite(ctx_alone_pr) or not np.isfinite(ctx_alone_roc) or
                     not np.isfinite(no_skill) or
                     (ctx_alone_pr <= no_skill + 0.02 and ctx_alone_roc < 0.60))
    mod_pr = (modm[(modm["group_type"] == "module")]["pr_auc"]
              if len(modm) else pd.Series(dtype=float))
    mod_roc = (modm[(modm["group_type"] == "module")]["roc_auc"]
               if len(modm) else pd.Series(dtype=float))
    modm_mod = modm[modm["group_type"] == "module"] if len(modm) else pd.DataFrame()
    worst_mod = (modm_mod.loc[modm_mod["pr_auc"].idxmin()]
                 if len(modm_mod) and modm_mod["pr_auc"].notna().any() else None)
    worst_sub = (subm.loc[subm["recall"].idxmin()]
                 if len(subm) and subm["recall"].notna().any() else None)
    sub_cal = (subm["calibration_slope"] if len(subm) else pd.Series(dtype=float))
    sub_recall = (subm["recall"] if len(subm) else pd.Series(dtype=float))
    quiz_delta = ({str(r["model"]): float(r["delta_pr_auc"])
                   for _, r in quiz.iterrows()} if len(quiz) else {})
    expo_diff = (expo["d_clicks_diff"] if len(expo) else pd.Series(dtype=float))
    bins_diff = (bins["d_clicks_diff"] if len(bins) else pd.Series(dtype=float))
    placebo_roc = (float(placebo["roc_auc"].iloc[0]) if len(placebo) else float("nan"))
    b2_row = inc[(inc["cutoff"] == 56) & (inc["cohort"] == "fixed") &
                 (inc["method"] == "B2_offset")] if len(inc) else pd.DataFrame()
    b2_status = str(b2_row["status"].iloc[0]) if len(b2_row) and "status" in b2_row else "unknown"
    lm_conc = (lm.groupby(["landmark", "cause"])["test_concordance"].first()
               if len(lm) else pd.Series(dtype=float))

    parts = []
    parts.append(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{array}
\captionsetup{font=small}
\setlength{\parskip}{3pt}
\setlength{\parindent}{0pt}
\title{Same Engagement, Different Outcomes\\\large Explaining differences in academic results among similarly active online learners}
\author{Group 9: Jiayang Liu (50027216), Yanbin Chen (50026846),\\ Yanshun Wang (50026688), Yuming Zhang (50025933)}
\date{Data Mining Course Project \\ \today}
\begin{document}
\maketitle""")
    if draft:
        parts.append(r"""\begin{center}
\fcolorbox{red}{yellow}{\parbox{0.9\linewidth}{\centering\textbf{SIMULATION DRAFT.}
This report was produced from the 500-enrolment rehearsal run. It is an execution
artefact, not the final full-data submission. Numbers will change in the full run.}}
\end{center}""")

    parts.append(r"\noindent\textbf{Page-count note.} The main text ends on page "
                 r"\pageref{endmain}; references, disclosures, and the appendix are excluded "
                 r"from the 6--10 page main-text requirement.")
    parts.append(r"\section{Introduction}")
    parts.append(
        "Online learning platforms record when and how students use course resources. "
        "Two students can generate the same number of clicks in the first four weeks and "
        "still finish with very different results. Total activity is therefore an entry "
        "point, not an explanation: the temporal distribution of study, the mix of resource "
        "types, and the way activity changes after the first month may carry information that "
        "a single click count hides. This project asks: \\emph{among students with similar "
        "early engagement, which temporal and resource-use patterns are associated with "
        "passing rather than failing?} Withdrawn students are treated as a separate outcome, "
        "not as a missing value or as a kind of failure."
    )
    parts.append(
        "The project uses the Open University Learning Analytics Dataset (OULAD). The unit of "
        "analysis is one enrolment (module $\\times$ presentation $\\times$ student), not a "
        "unique person; the same student can appear in several presentations. We build an "
        "audited pipeline that freezes the analysis configuration, constructs features only "
        "from dates before each cutoff, splits by student, and reports calibration and "
        "uncertainty rather than a single accuracy figure. The main analyses are (i) coarsened "
        "exact matching (CEM) within module and presentation, (ii) clustering of first-month "
        "behavioural profiles with bootstrap stability, (iii) a classification ladder M0--M5, "
        "and (iv) a cutoff curve with two ways of updating a first-month prediction at week 8. "
        "A three-class competing-risk analysis keeps Withdrawn separate, and a specification "
        "curve with diagnostics and placebo checks tests whether the conclusions survive reasonable "
        "analysis choices."
    )
    parts.append(
        f"The headline result is qualified. In the main held-out split, the "
        f"M1 behavioural model reaches PR-AUC {f(pr_main)} and ROC-AUC {f(roc_main)}; "
        f"grouped cross-validation ROC-AUC is {f(roc_cv)}, forward-temporal ROC-AUC is "
        f"{f(roc_temp)}, and leave-one-module-out ROC-AUC is {f(roc_lomo)}. The spread among "
        f"the strict models is comparable to the change under temporal/module extrapolation, "
        f"so neither should be read as a deployment guarantee. The random-label negative "
        f"control has ROC-AUC {f(rand_roc)}, close to chance. These are associations, not "
        f"causal effects."
    )

    parts.append(r"\section{Problem Formulation}")
    parts.append(
        "Let $i$ index an enrolment. For the primary target T1, $y_i=1$ if the enrolment ends "
        "in Fail and $y_i=0$ if it ends in Pass or Distinction; Withdrawn enrolments are "
        "excluded from T1 and analysed separately. Fail is the positive class because the task "
        "is early warning. T2 counts Fail or Withdrawn as positive and T3 is the "
        "three-class outcome (Pass/Distinction, Fail, Withdrawn). For a cutoff $c$ (days from "
        "course start), the feature vector $X_i(c)$ may use only VLE clicks with date $<c$ and "
        "assessment submissions with date $<c$. No score, banked-credit flag, final result, or "
        "post-cutoff click is allowed in the strict feature set. The predictive task is to "
        "estimate $p_i(c)=\\Pr(y_i=1\\mid X_i(c))$ and to rank enrolments by risk."
    )
    parts.append(
        "Two windows are defined. W1 is days 0--27. W2 is days 28 to the earlier of day 55 and "
        "the first weighted TMA/CMA minus one; when the first TMA is before week 4, the plan "
        "falls back to the fixed 28--55 window and records the fallback. OULAD dates are "
        "relative to presentation start and include negative values; the primary analysis "
        "excludes pre-start clicks, and clipping or a separate pre-start feature is used only "
        "in sensitivity analyses. The primary metric is PR-AUC for the Fail class, the minority "
        "of T1, read against its no-skill value (the Fail prevalence); ROC-AUC is reported for "
        "comparability, and recall is reported at a threshold chosen on validation data at "
        "precision 0.80, with the rule actually used recorded in the model table. "
        "Calibration is assessed with Brier score, "
        "calibration slope/intercept, and reliability curves."
    )
    parts.append(
        "The cohort is defined in two steps. First, the Withdrawn audit asks whether the click "
        "stream stops at \\texttt{date\\_unregistration}. It does not: in the full data, "
        f"{_tex(audit.get('students_with_post_click', '--'))} withdrawn students have "
        f"{_tex(audit.get('post_unregistration_click_rows', '--'))} click rows after that date. "
        "The main cohort therefore keeps all T1 enrolments and treats the at-risk filter as a "
        "sensitivity analysis; Withdrawn remains a separate outcome in T3 and the competing-risk "
        "model. This is a deliberate correction to the tempting, but unsupported, practice of "
        "silently dropping everyone with an early unregistration date."
    )

    parts.append(r"\section{Methodology}")
    parts.append(r"\subsection{Data and leakage audit}")
    parts.append(
        "OULAD contains seven tables: courses (22 rows), assessments (206), VLE resources "
        "(6,364), studentInfo (32,593 enrolments), studentRegistration (32,593), "
        "studentAssessment (173,912), and studentVle (10,655,280). The missing-value marker is "
        "\\texttt{?}, not an empty field. VLE dates range from $-25$ to 269; 688,988 rows "
        "(6.47\\%) are before course start. The first TMA/CMA is often before day 28, so the W2 "
        "end is computed per presentation with the documented fallback."
    )
    parts.append(
        "The strict pipeline drops quiz/external-quiz sites, excludes banked assessments from "
        "submission features, and never reads \\texttt{score} outside the M4 upper-bound model. "
        "Every feature table records the maximum date actually used and is asserted to be below "
        "its cutoff. Features are grouped into volume, composition, breadth, regularity, timing, "
        "assessment-submission, and change families. All aggregates are cutoff-aware; no "
        "full-period aggregate is computed and then truncated."
    )
    parts.append(r"\subsection{Splits and matching}")
    parts.append(
        "The primary split is 60/20/20 by \\texttt{id\\_student}; a five-fold GroupKFold, a "
        "forward-temporal split (2013B/2013J train, 2014B/2014J test), and a leave-one-module-out "
        "split are also reported. Temporal and LOMO training rows are further restricted so that "
        "no student appears on both sides of a split. In the full-data design, the row-level "
        "control quantifies the leakage that a naive random split would create; "
        + leak_txt + ". CEM uses module $\\times$ "
        "presentation as exact strata and coarsens W1 clicks, "
        "active days, and activity entropy within presentation. Strata without both outcomes are "
        "discarded; treated units receive weight 1 and controls receive $n_t/n_c$ within stratum. "
        "Balance is assessed on the coarsened variables (the design variables) with SMD, and raw "
        "continuous SMD is reported as a diagnostic. A cluster bootstrap over strata gives "
        "confidence intervals for W2 differences. Matching is descriptive: it balances measured "
        "W1 variables and cannot remove unmeasured confounding or post-baseline selection."
    )
    parts.append(r"\subsection{Clustering, models, updating and competing risks}")
    parts.append(
        "Clustering uses W1 volume, breadth, regularity, and composition features, standardised "
        "within presentation and reduced by PCA to eight components. k-means is run for "
        "$k=2,\\ldots,8$ with 50 initialisations. Stability is measured by 300 bootstrap "
        "refits: ARI against the full-sample partition and cluster-wise Jaccard after Hungarian "
        "label alignment. The project threshold is mean ARI $\\ge 0.60$; this is a project "
        "convention, not a literature constant."
    )
    parts.append(
        "The model ladder adds one feature family at a time: M0 uses log clicks and active days; "
        "M1 adds breadth, regularity, and composition; M2 adds timing; M3 adds assessment "
        "submission timing without scores; M4 adds scores as an upper bound; M5 replaces the "
        "logistic model with HistGradientBoosting on M3 features. A context-only baseline "
        "(M-1) uses demographic and registration variables; M0c and M1c add the M0 and M1 "
        "features to that baseline, separating the clickstream increment from context. "
        "All preprocessing is fitted "
        "inside the training fold. The classification threshold is selected on validation data "
        "at precision 0.80 (falling back to maximum F1 when that precision is unreachable), then "
        "applied once to the held-out test set. Bootstrap intervals resample held-out rows."
    )
    parts.append(
        "For updating, the cutoff curve retrains M0--M3 at each cutoff on the same split. B1 "
        "stacking adds a cross-fitted W1 probability to a W2 model; B2 uses $\\operatorname{logit}"
        "(p_{W1})$ as a fixed offset in a binomial GLM. Cross-fitting is done with an inner "
        "GroupKFold on the training rows, so a row's W1 probability never comes from the same "
        "fit that predicts it. The competing-risk analysis fits a three-class multinomial model "
        "and cause-specific Cox models on weekly event times; the cumulative incidence function "
        "is computed with the Aalen--Johansen recursion. The specification curve varies cutoff, "
        "pre-start handling, minimum W1 clicks, target, feature level, model family, matching "
        "method, and clustering $k$, and adds random-label, pre-start-only, and placebo-window "
        "diagnostics."
    )

    parts.append(r"\section{Experiments}")
    parts.append(r"\subsection{Data, cohort and overlap}")
    parts.append(
        f"The analysis dataset has {n_total} enrolments; T1 contains {n_t1} with Fail (positive-class) rate "
        f"{pct(base_t1)}, and {n_withdrawn} enrolments are Withdrawn. Table~\\ref{{tab:cohort}} "
        f"shows the cohort at three cutoffs. The W1 click distributions of Pass/Distinction and "
        f"Fail {'overlap substantially but not completely' if (np.isfinite(ov_min) and ov_min >= 0.5) else 'overlap only partially'}: the presentation-weighted histogram-intersection "
        f"coefficient ranges from {f(ov_min)} to {f(ov_max)} across cutoffs, and the zero-click "
        f"share at day 28 is {pct(zero_share)}. "
        f"{'This supports the premise that similar early engagement exists, but the overlap is not complete.' if (np.isfinite(ov_min) and ov_min >= 0.5) else 'This is itself a finding: similar early engagement is not a safe assumption.'} "
        f"CEM common support for T1 is {f(cem_support)}, "
        f"so the matched comparison covers "
        f"{'a limited region' if (np.isfinite(cem_support) and cem_support < 0.5) else 'most of the treated group'}."
    )
    parts.append(flow_table)
    parts.append(split_table)
    parts.append(_fig("figures/f03_click_distribution_by_outcome.png",
                      "W1 click distributions by final outcome. Overlap is "
                      + ("substantial but incomplete" if (np.isfinite(ov_min) and ov_min >= 0.5) else "partial")
                      + "; it determines the common support available to matching.",
                      "fig:overlap", "0.72\\linewidth"))

    parts.append(r"\subsection{Matching and behavioural profiles}")
    wd = t.get("t06_w2_differences", pd.DataFrame())

    def wd_line(metric):
        r = wd[wd["metric"] == metric] if len(wd) else pd.DataFrame()
        if not len(r):
            return "not available"
        r = r.iloc[0]
        return (f"{f(r['diff_pass_minus_fail'])} "
                f"[{f(r['ci_low'])}, {f(r['ci_high'])}]")

    parts.append(
        f"CEM retained {_tex(cem_n)} T1 enrolments in strata containing both outcomes. Balance on the "
        f"coarsened design variables is exact by construction (maximum |SMD| {f(cem_smd)}); the "
        f"raw continuous diagnostic has maximum |SMD| {f(raw_smd)}. Common support for treated "
        f"units is {f(cem_support)}, so the matched comparison describes the overlap region "
        f"rather than a population-average effect. Within matched strata, the Pass/Distinction "
        f"minus Fail differences (95\\% cluster-bootstrap CI) are {wd_line('d_clicks')} for "
        f"click change, {wd_line('d_active_days')} for active-day change, and "
        f"{wd_line('d_entropy_activity')} for activity-entropy change. "
        f"Figure~\\ref{{fig:love}} summarises balance; Table~\\ref{{tab:models}} reports the "
        f"model ladder. The E-value for the high-vs-low W2-change contrast is {f(ev)}; an "
        f"unmeasured confounder with risk ratio about {f(ev)} could explain that contrast. "
        f"This is a modest E-value, so the matched contrast is fragile to unmeasured "
        f"confounding. "
        f"Matching-method contrasts on the same W2 click change are "
        f"{f(match_vals.get('CEM'))} (CEM), {f(match_vals.get('PSM'))} (PSM), and "
        f"{f(match_vals.get('stratified'))} (stratified); the magnitude differs across methods"
        + (" and the sign is not stable, so this is a descriptive sensitivity rather than a "
           "directional effect." if not match_sign_stable else
           "; this is a descriptive sensitivity rather than a directional effect.")
        + f" The exposure-definition sensitivity is summarised in Appendix "
        f"Table~\\ref{{tab:sensitivity}}."
    )
    parts.append(_fig("figures/f06_love_plot.png",
                      "CEM balance. The bins used to define strata are balanced; raw continuous "
                      "active days retain a residual difference, which is reported rather than hidden.",
                      "fig:love", "0.72\\linewidth"))
    parts.append(
        f"Clustering selected k={_tex(chosen_k)} with mean bootstrap ARI {f(chosen_ari)} and minimum "
        f"cluster Jaccard {f(chosen_j)}. "
        + ((f"k=2 is the more stable partition (ARI {f(k2_ari)}, Jaccard {f(k2_j)}), while k="
            f"{_tex(chosen_k)} has the higher silhouette ({f(chosen_sil)} vs {f(k2_sil)}); the "
            f"choice therefore trades some stability for separation. ") if use_k2_compare else
           (f"k=2 is both the most stable and the highest-silhouette partition. " if str(chosen_k) == "2" and
            np.isfinite(k2_sil) and np.isfinite(max_sil) and k2_sil >= max_sil - 1e-12 else
            (f"k={_tex(chosen_k)} is the most stable partition; it was selected by the stability-first "
             f"rule, although k={max_sil_k} has a higher silhouette ({f(max_sil)} vs "
             f"{f(chosen_sil)} for k={_tex(chosen_k)}). ")))
        + f"The {_tex(chosen_k)} profiles differ in outcome mix, and no "
        f"profile is homogeneous. The "
        f"cluster labels are behavioural descriptions, not learning-style types, and the chosen "
        f"partition should not be read as a latent class model."
    )
    parts.append(_fig("figures/f07_cluster_outcome.png",
                      "Outcome mix by W1 behavioural profile. Clusters are descriptive; the "
                      "within-cluster outcome variation is the phenomenon of interest.",
                      "fig:cluster", "0.72\\linewidth"))

    parts.append(r"\subsection{Classification ladder and external validation}")
    parts.append(
        f"Table~\\ref{{tab:models}} reports the main held-out test. The highest main-test "
        f"PR-AUC is {best_model} ({f(best_pr)}); the ladder "
        f"{'is' if monotone else 'is not'} monotone across M0--M3. M4 is an upper bound "
        f"because scores are not known at prediction time; it is not an early-warning model. "
        + ((f"Among strict models, {best_strict} has the highest main-test PR-AUC "
            f"({f(best_strict_pr)}), but its temporal and LOMO ROC-AUC ({f(m5_temp)}, "
            f"{f(m5_lomo)}) are lower than M1's ({f(m1_temp)}, {f(m1_lomo)}); the ranking is "
            f"not stable across extrapolation. ") if best_strict == "M5" else "")
        + f"The calibration slope of {best_model} is {f(best_cal)}; slopes below 1 indicate "
        f"over-confident probabilities, which is why Brier and reliability curves are reported "
        f"alongside ranking metrics."
    )
    parts.append(model_table)
    parts.append(ext_table)
    parts.append(_fig("figures/f08_pr_curves.png",
                      "Precision--recall curves on the main held-out test for M0--M5.",
                      "fig:pr", "0.62\\linewidth"))
    parts.append(_fig("figures/f08_calibration.png",
                      "Reliability curves on the main held-out test. Several richer models are "
                      "over-confident; M4 is an upper bound with scores.",
                      "fig:cal", "0.86\\linewidth"))
    ctx_note = (
        f"Context alone is close to chance in this run (PR-AUC {f(ctx_alone_pr)} vs no-skill "
        f"{f(no_skill)}; ROC-AUC {f(ctx_alone_roc)}; calibration slope "
        f"{f(ctx_cal.get('M-1_context', float('nan')))}), so this "
        f"{'rehearsal' if draft else 'run'} provides little "
        f"context signal for the click features to proxy. "
        if ctx_at_chance else
        f"Context alone reaches PR-AUC {f(ctx_alone_pr)} and ROC-AUC {f(ctx_alone_roc)}. ")
    parts.append(
        f"Table~\\ref{{tab:context}} adds a context-only baseline. " + ctx_note +
        f"Adding click volume (M0c) gives {f(ctx_pr.get('M0c_context+M0', float('nan')))}, "
        f"and adding behaviour (M1c) gives {f(ctx_pr.get('M1c_context+M1', float('nan')))}. "
        f"The paired M1c--M0c difference is {f(ctx_delta)} PR-AUC "
        f"(95\\% CI [{f(ctx_delta_lo)}, {f(ctx_delta_hi)}]) and "
        f"{f(ctx_roc_delta)} ROC-AUC "
        f"(95\\% CI [{f(ctx_roc_lo)}, {f(ctx_roc_hi)}]) on the same test rows. "
        + (("Neither the PR-AUC nor the ROC-AUC CI detects a difference beyond context in this run. "
            if ctx_ci_includes_zero and ctx_roc_ci_includes_zero else
            "The PR-AUC CI includes zero, so the behavioural increment is not detectable on "
            "PR-AUC; the ROC-AUC CI excludes zero, but the difference is small. "
            if ctx_ci_includes_zero else
            "The PR-AUC CI excludes zero, so the difference is detectable on PR-AUC; "
            "the ROC-AUC CI " + ("includes zero. " if ctx_roc_ci_includes_zero else
                                 "also excludes zero. ")))
        + "The point estimate is small in absolute terms; the CI is the basis for the "
        + "detectability statement. The contrast is descriptive: context variables "
        f"are not randomised, and the context-only model is "
        f"{'near chance here' if ctx_at_chance else 'imperfect'}, so we do not claim a "
        f"context-adjusted effect."
    )
    parts.append(ctx_table)
    parts.append(_fig("figures/f08_context_comparison.png",
                      "Context-only, context-plus-volume, and context-plus-behaviour models on "
                      "the same held-out split.", "fig:context", "0.72\\linewidth"))
    parts.append(
        f"Module-level PR-AUC ranges from {f(mod_pr.min())} to {f(mod_pr.max())}; "
        f"subgroup calibration slopes range from {f(sub_cal.min())} to {f(sub_cal.max())}, and "
        f"subgroup recall ranges from {f(sub_recall.min())} to {f(sub_recall.max())}. "
        + (f"The lowest module PR-AUC is {_tex(worst_mod['group'])} "
           f"({f(worst_mod['pr_auc'])}); the lowest subgroup recall is "
           f"{_tex(worst_sub['group_type'])}:{_tex(worst_sub['group'])} "
           f"({f(worst_sub['recall'])}). " if worst_mod is not None and worst_sub is not None
           else "")
        + f"Groups with fewer than 200 test rows are flagged low-n; groups with fewer than five "
        f"rows, a single outcome class, or a missing value are omitted, so counts need not sum "
        f"to the test n. These are performance differences, not evidence that the model causes "
        f"them; per-group tables and figures are in Appendix~\\ref{{app:subgroups}}."
    )

    parts.append(r"\subsection{Cutoff curve and updating}")
    if best_cut == 28 and np.isfinite(cut_delta) and cut_delta <= 0:
        cut_text = (f"Table~\\ref{{tab:cutoff}} and Figure~\\ref{{fig:cutoff}} show M3 as the "
                    f"observation window grows. The curve is not monotone in this run: the best "
                    f"PR-AUC is near day 28, and day 56 changes it by {f(cut_delta)}. This is "
                    f"consistent with early behaviour carrying most of the available signal once "
                    f"the cohort is fixed, and with later features adding noise as well as information.")
    else:
        cut_text = (f"Table~\\ref{{tab:cutoff}} and Figure~\\ref{{fig:cutoff}} show M3 as the "
                    f"observation window grows. The best PR-AUC is at day {best_cut}; day 56 "
                    f"changes it by {f(cut_delta)} relative to day 28. The curve is "
                    f"{'monotone' if cut_monotone else 'not monotone'} in this run, so the "
                    f"cutoff choice is not innocuous.")
    parts.append(cut_text)
    parts.append(cutoff_table)
    parts.append(_fig("figures/f09_cutoff_curve.png",
                      "M3 PR-AUC versus cutoff for the fixed and at-risk cohorts. The at-risk "
                      "filter is a sensitivity analysis, not the main cohort.",
                      "fig:cutoff", "0.72\\linewidth"))
    if len(traj):
        if div_week is not None:
            div_note = (f" Because week {div_week} lies inside W1 (days 0--27), the matched "
                        f"groups were already diverging during the baseline window; the "
                        f"same-engagement premise is approximate, not exact."
                        if div_week <= 3 else
                        f" The divergence appears after W1, in the follow-up window.")
            tw_text = (f"Matched weekly trajectories (Table~\\ref{{tab:trajectory}}, "
                       f"Figure~\\ref{{fig:traj}}) show a first stable divergence at week "
                       f"{div_week}." + div_note)
        else:
            tw_text = (f"Matched weekly trajectories (Table~\\ref{{tab:trajectory}}, "
                       f"Figure~\\ref{{fig:traj}}) show no stable divergence in weeks 0--8 "
                       f"under the two-consecutive-week rule; any isolated week with a positive "
                       f"interval is still reported in the table and is not treated as a stable "
                       f"divergence.")
    else:
        tw_text = "Matched weekly trajectories are not available in this run."
    parts.append(tw_text)
    parts.append(_fig("figures/f12_trajectory_divergence.png",
                      "Matched weekly trajectories, absolute log-clicks and normalized share. "
                      "The divergence week is descriptive, not causal.",
                      "fig:traj", "0.95\\linewidth"))
    if len(pos_valid):
        upd_text = (f"Table~\\ref{{tab:updating}} compares W1-only, W2-only, all-features, "
                    f"and stacking (B1) at day 56. {len(pos_valid)} of "
                    f"{len(valid_inc)} valid updating methods have a 95\\% interval for the "
                    f"increment above zero; the largest PR-AUC is {_tex(best_inc['method'])} "
                    f"({f(best_inc['pr_auc'])}, Δ "
                    f"{f(best_inc['delta_pr_auc_vs_28'])}). B2 is excluded from the table and "
                    f"reported in Appendix Table~\\ref{{tab:b2_failed}} as a failed "
                    f"specification; it is not evidence about sequential updating. "
                    f"Even where the interval excludes zero, the increment is small relative "
                    f"to the between-specification spread, so we treat it as provisional.")
    else:
        upd_text = (f"Table~\\ref{{tab:updating}} compares W1-only, W2-only, all-features, "
                    f"and stacking (B1) at day 56. None of the updating methods beats W1-only "
                    f"by a convincing margin: the intervals for the increment mostly include "
                    f"zero or are negative. B2 is excluded from the table and reported in "
                    f"Appendix Table~\\ref{{tab:b2_failed}} as a failed specification. We report "
                    f"a null result rather than select a favourable cutoff.")
    parts.append(upd_text)
    parts.append(upd_table)

    parts.append(r"\subsection{Competing risks}")
    parts.append(
        f"The three-class model keeps Withdrawn separate (Table~\\ref{{tab:threeclass}}). "
        f"The lowest F1 among the three classes is {_tex(hardest_class)} ({f(hardest_f1)}); the "
        f"precision/recall trade-off differs by class. Landmark cause-specific Cox models at "
        f"days 28 and 56 use only features available at the landmark and are evaluated by "
        f"test concordance (Appendix Table~\\ref{{tab:landmark}}); we do not interpret the "
        f"landmark coefficients causally. The cumulative-incidence curves in Figure~\\ref{{fig:cif}} make the "
        f"competition explicit: by the end of the course most enrolments have either passed or "
        f"withdrawn, and a binary 'Fail' label would hide that distinction."
    )
    parts.append(t3_table)
    parts.append(_fig("figures/f10_cumulative_incidence.png",
                      "Cumulative incidence of Pass/Distinction, Fail, and Withdrawn. Withdrawn "
                      "is a competing event, not a missing label.",
                      "fig:cif", "0.72\\linewidth"))

    parts.append(r"\subsection{Diagnostics and placebo checks}")
    if len(nc):
        pre_role = ("above the no-skill PR-AUC and therefore not a null"
                    if pre_above_noskill else "not above the no-skill PR-AUC here")
        placebo_role = ("post-outcome persistence; a placebo failure, not a leak test"
                        if placebo_above_noskill else "at the no-skill rate here")
        expo_txt = ""
        if expo_roc:
            expo_txt = (f" The non-assessment-exposure diagnostic reaches ROC-AUC "
                        f"{f(expo_roc.get('non_assessment_w2_change', float('nan')))} "
                        f"(PR-AUC {f(expo_pr.get('non_assessment_w2_change', float('nan')))}); "
                        f"if predictive, it indicates generic engagement or background rather "
                        f"than a learning-strategy signal.")
        ctl = (f"The random-label leak check is close to chance, which argues against a gross "
               f"leakage bug. The pre-start-only positive control is {f(pre_start_auc)} "
               f"({pre_role}); the future-window control is {f(placebo_auc)} ({placebo_role}). "
               f"The repeater negative-control outcome reaches ROC-AUC {f(placebo_roc)}. "
               f"These diagnostics bound interpretation; they do not identify a clean null or "
               f"prove the absence of leakage. Table~\\ref{{tab:negctl}} lists each diagnostic "
               f"with its role and interpretation." + expo_txt)
    else:
        ctl = ""
    parts.append(spec_line + " " + ctl)
    if len(quiz_delta) or len(expo_diff) or len(bins_diff) or np.isfinite(placebo_roc):
        parts.append(
            f"Measurement sensitivity: quiz inclusion changes M1/M3 PR-AUC by "
            f"{f(quiz_delta.get('M1', float('nan')))}/{f(quiz_delta.get('M3', float('nan')))}. "
            f"The exposure-definition CEM gives W2 click differences from {f(expo_diff.min())} "
            f"to {f(expo_diff.max())}; the seven CEM bin grids range from {f(bins_diff.min())} "
            f"to {f(bins_diff.max())}. These measurement and matching comparisons are "
            f"diagnostics, not causal evidence."
            if (len(quiz_delta) or len(expo_diff) or len(bins_diff) or np.isfinite(placebo_roc))
            else "")
    parts.append(nc_table)
    parts.append(_fig("figures/f11_specification_curve.png",
                      "Specification curve: predictive specifications (left) and matching-method "
                      "effects on W2 click change (right).",
                      "fig:spec", "0.95\\linewidth"))

    parts.append(r"\section{Conclusion and Limitations}")
    parts.append(
        "The pipeline answers the research question with a qualified yes: early behavioural "
        "structure adds information beyond total clicks in the main split, but the "
        f"context-adjusted M1c-minus-M0c difference is {f(ctx_delta)} PR-AUC "
        f"(95\\% CI [{f(ctx_delta_lo)}, {f(ctx_delta_hi)}])"
        + ((" with no detectable behavioural increment beyond context on either PR-AUC or ROC-AUC"
            if ctx_ci_includes_zero and ctx_roc_ci_includes_zero else
            " with no detectable behavioural increment beyond context on PR-AUC; the ROC-AUC "
            "difference is small"
            if ctx_ci_includes_zero else
            " with a small detectable PR-AUC difference"))
        + ", and the overall increment is small, unstable across extrapolation settings, and "
        "sensitive to the operationalisation of engagement. "
        "The most defensible conclusions are negative and methodological. First, row-level "
        "splitting is not acceptable; grouping by student changes the evaluation and removes a "
        "design-level leak"
        + ("; this rehearsal has zero student overlap, so it does not estimate "
           "the size of that leak. " if draft else
           (f": a row-level split would share {int(leak_students)} students "
            f"({pct(leak_share)} of test rows) between train and test. " if leak_known else ". "))
        + "Second, Withdrawn cannot be treated as a missing label: the click stream "
        "does not stop at unregistration, and the three outcomes have different mechanisms. "
        "Third, a richer model is not automatically better; calibration and external validation "
        "matter more than a small PR-AUC gain on a random split."
    )
    parts.append(
        "Limitations are substantial. The data are observational; matching balances only measured "
        "W1 variables, and W1 engagement is already post-baseline, so the matched differences are "
        "associations, not causal effects. Clicks are a weak proxy for study time; OULAD has no "
        "session timestamps and no reliable time-on-task measure. "
        + (("This rehearsal uses 500 enrolments and the test sets are small, so the point "
            "estimates are noisy. ") if draft else
           ("The full cohort is larger, but the held-out test set and the temporal fold still "
            "limit precision. "))
        + (("The residual raw imbalance in active days and the limited common support argue for "
            "conservative language. ") if (np.isfinite(raw_smd) and raw_smd > 0.1) or
           (np.isfinite(cem_support) and cem_support < 0.5) else
           (f"CEM diagnostics are acceptable on the coarsened design (raw max |SMD| {f(raw_smd)}, "
            f"treated support {f(cem_support)}), but matching is still descriptive. "))
        + "The optional TMA appendix analyses were not attempted: the +21/+28 window "
        "overlaps W1/W2 and conditions on continued enrolment, the +14 anchor varies by "
        "presentation, and first-TMA scores are not available at the early cutoff. They are "
        "alternative, post-hoc estimands, not early-warning evidence. "
        "Finally, the project does not attempt a formal Bayesian "
        "prior-to-posterior model; the updating analysis is a predictive engineering comparison, "
        "not a posterior inference."
    )
    parts.append(
        f"Table~\\ref{{tab:limitations}} lists the main threats and residual risks. "
        f"{missing_unreg} Withdrawn enrolments have no date\\_unregistration and are treated as "
        f"course-end events; the T3 held-out test has n={t3_n}, not the T1 test n. "
        f"The day-28 zero-click share is {pct(zero_share)} in T1 but {pct(zero_all)} in the "
        f"full queue. Main-text page count excludes references, disclosures, and the appendix."
    )
    parts.append(lim_table)
    parts.append(r"\label{endmain}")
    parts.append(r"\section*{Disclosures}")
    parts.append(
        "External collaborators: none. Data: OULAD is a public dataset; no restricted or "
        "personal data were used beyond the public release. AIGC: Claude Code (Anthropic) was "
        "used to scaffold the analysis code, execute the staged pipeline, generate tables and "
        "figures, and draft this report. The pipeline's outputs are generated by the committed "
        "code and logged with configuration hashes; the group is responsible for reviewing the "
        "final submission. No references were fabricated; every cited source is listed below."
    )
    parts.append(r"\begin{thebibliography}{99}")
    refs = [
        "Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171.",
        "Wu, Jia & Pham (2026). Early prediction of at-risk students with the Open University Learning Analytics Dataset. Frontiers in Education, 11, 1917092. The reported 0.7747 ROC-AUC is for day-28 prediction of Fail-or-Withdrawn with 22 features including early scores, not a pure clickstream Pass/Distinction-vs-Fail baseline.",
        "Le, Abel & Laforge (2026). LEAP: leakage-aware early prediction. arXiv:2605.25794 (preprint).",
        "Gasevic, D., et al. (2016). Learning analytics should not promote one size fits all. Internet and Higher Education, 28, 68-84.",
        "Iacus, S. M., King, G., & Porro, G. (2012). Causal inference without balance checking: Coarsened exact matching. Political Analysis, 20(1), 1-24.",
        "VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. PLOS ONE, 10(3), e0118432.",
        "Van Calster, B., et al. (2019). Calibration: The Achilles heel of predictive analytics. BMC Medicine, 17, 230.",
        "Hennig, C. (2007). Cluster-wise assessment of cluster stability. Computational Statistics & Data Analysis, 52(1), 258-271.",
        "Monti, S., et al. (2003). Consensus clustering: A resampling-based method. Machine Learning, 52, 91-118.",
        "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.",
        "Kovanovic, V., et al. (2015). Does time-on-task estimation matter? LAK '15, 184-193.",
    ]
    for i, r in enumerate(refs, start=1):
        parts.append(r"\bibitem{ref" + str(i) + "} " + _tex(r))
    parts.append(r"\end{thebibliography}")

    # Appendix (excluded from the 6-10 page main text).
    parts.append(r"\appendix")
    parts.append(r"\section{Landmark Cox}")
    parts.append(lm_table)
    parts.append(
        "The landmark-28 model uses only W1 features; the landmark-56 model uses W1 and W2 "
        "features available at day 56. Both exclude events at or before the landmark and report "
        "held-out test concordance, addressing the static-W2 leakage in the baseline appendix. "
        "The table reports concordance; coefficient-level HR and 95\\% CI are in "
        "\\texttt{t10\\_landmark\\_cox.csv}."
    )
    parts.append(r"\section{Module and subgroup metrics}\label{app:subgroups}")
    parts.append(mod_table)
    parts.append(sub_table)
    parts.append(_fig("figures/f08_module_heatmap.png",
                      "Per-module held-out metrics (M3). Column-normalized for display; "
                      "cell labels are raw values.", "fig:modheat", "0.86\\linewidth"))
    parts.append(_fig("figures/f08_subgroup_calibration.png",
                      "Subgroup calibration slopes (M3, held-out test).", "fig:subcal",
                      "0.95\\linewidth"))
    parts.append(r"\section{Trajectory and clustering diagnostics}")
    parts.append(traj_table)
    parts.append(_fig("figures/f07_pca_scatter.png",
                      "PCA diagnostic: PC1 x PC2, coloured by the chosen k-means partition. "
                      "The 8-component solution explains the variance shown; this is a "
                      "diagnostic, not evidence that clusters are real.", "fig:pca",
                      "0.72\\linewidth"))
    parts.append(_fig("figures/f07_silhouette.png",
                      "Silhouette distribution for the chosen k (sampled).", "fig:sil",
                      "0.72\\linewidth"))
    parts.append(r"\section{Sensitivity tables}")
    parts.append(sens_table)
    parts.append(r"\section{B2 offset (failed specification)}")
    parts.append(b2_table)
    parts.append(
        "B2 was attempted with a fixed logit(pW1) offset, both without and with an "
        "intercept. The selected fit remains severely over-confident; it is retained here "
        "only for transparency and is not a sequential-updating result."
    )

    parts.append(r"\end{document}")
    return "\n".join(parts)


def compile_pdf(tex_path: Path, out_dir: Path, runs=2):
    logs = []
    for _ in range(runs):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            tex_path.name], cwd=out_dir, capture_output=True, text=True)
        logs.append(r.stdout + "\n" + r.stderr)
        if r.returncode != 0:
            return False, "\n".join(logs)
    pdf = out_dir / (tex_path.stem + ".pdf")
    return pdf.exists(), "\n".join(logs)


def pdf_pages(pdf: Path) -> int:
    try:
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        m = re.search(r"Pages:\s+(\d+)", r.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return -1


def build_markdown(ctx) -> str:
    t, stages = load_tables(ctx)
    models = t.get("t08_model_metrics", pd.DataFrame())
    lines = [f"# Same Engagement, Different Outcomes — Group 9", ""]
    if ctx.mode == "simulation":
        lines += ["> **SIMULATION DRAFT.** 500-enrolment rehearsal run; not the final submission.", ""]
    lines += ["## Problem", "",
              "Among students with similar first-four-week VLE engagement, which temporal and "
              "resource-use patterns are associated with passing rather than failing? Withdrawn "
              "students are analysed as a separate outcome.", ""]
    lines += ["## Data and leakage controls", "",
              "- Unit: enrolment (module x presentation x student); 32,593 enrolments in full OULAD.",
              "- Strict windows: W1 days 0-27; W2 days 28 to min(55, first TMA-1), with a fixed "
              "28-55 fallback when the first TMA is before week 4.",
              "- No score, banked flag, final result, or post-cutoff click in strict features.",
              "- Grouped splits; temporal and LOMO folds are student-disjoint.", ""]
    lines += ["## Main results", ""]
    for m in MODELS:
        lines.append(f"- {m}: PR-AUC {f(_mv(models,m,'main_test','pr_auc'))}, "
                     f"ROC-AUC {f(_mv(models,m,'main_test','roc_auc'))}, "
                     f"Brier {f(_mv(models,m,'main_test','brier'))}.")
    lines += ["", "M4 includes assessment scores and is an upper bound only.", ""]
    lines += ["## Reproducibility", "",
              "See README.md. Every stage writes run.jsonl, a stage summary, a handoff note, "
              "and an atomic STATUS.json update.", ""]
    return "\n".join(lines)


def build_readme(ctx) -> str:
    return f"""# OULAD — Same Engagement, Different Outcomes

Group 9 course project. The unit is an enrolment (`code_module`, `code_presentation`,
`id_student`). The primary target T1 is Fail (positive, 1) vs Pass/Distinction (0), with
Withdrawn excluded and analysed separately in T3 / competing risks. PR-AUC is read against
the Fail prevalence (no-skill value).

## Data

Download the public OULAD archive from UCI:

```bash
mkdir -p data/raw/oulad
curl -L --fail --retry 3 -o data/raw/oulad/oulad.zip \\
  "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
shasum -a 256 data/raw/oulad/oulad.zip
unzip -o data/raw/oulad/oulad.zip -d data/raw/oulad/
```

The archive must yield seven CSVs: `courses.csv`, `assessments.csv`, `vle.csv`,
`studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`,
`studentVle.csv`. The raw archive is not included in the ZIP.

## Environment

Python 3.10+; the recorded environment is in `REQUIREMENTS.lock` and
`runs/{ctx.run_id}/env.json`. Core dependencies: pandas, numpy, scikit-learn,
scipy, pyyaml, jsonschema, pyarrow, matplotlib, seaborn, statsmodels.
`lifelines` is optional; cause-specific Cox models use `statsmodels`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r REQUIREMENTS.lock
```

## Reproducibility checklist

1. Record the raw archive hash and seven CSV row counts (S00 `manifest.json`,
   `DATA_DOWNLOAD.md`).
2. Verify `config/config.sha256` matches the frozen config; every event records
   `config_hash`.
3. Use the recorded `run_id`; do not edit `runs/<run_id>/`.
4. Run `python3 scripts/validate_run.py --run-id <run_id>` and require 0 errors /
   0 warnings before treating outputs as final.
5. `STATUS.final.json` is written by S12 and included in the ZIP.

## Frozen configuration

`config/config.yaml` is copied from `docs/contracts/config.example.yaml`; its sha256 is in
`config/config.sha256`. Do not edit it in place. Any change requires `CHANGE_REQUEST.md`
and a new `run_id`.

## One-command workflow

```bash
make test     # pytest -q
make sim      # S00-S12 on the 500-enrolment sample
make validate RUN_ID=<sim-run-id>
make full     # S00-S12 on all 32,593 enrolments
make validate RUN_ID=<full-run-id>
```

Each stage invocation enforces the previous-stage gate:

```bash
export PYTHONHASHSEED=0
RUN_ID="sim-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
for S in S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
  python3 scripts/run_stage.py --stage "$S" --mode simulation --run-id "$RUN_ID" \\
    --config config/config.yaml || break
done
python3 scripts/validate_run.py --run-id "$RUN_ID"
```

After the simulation is green, run the full data with a new `run-` id:

```bash
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
for S in S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
  python3 scripts/run_stage.py --stage "$S" --mode full --run-id "$RUN_ID" \\
    --config config/config.yaml || break
done
python3 scripts/validate_run.py --run-id "$RUN_ID"
```

## Outputs

- `runs/<run_id>/run.jsonl` — append-only event log.
- `runs/<run_id>/stages/Sxx.stage.json` — stage summary.
- `runs/<run_id>/handoff/Sxx.md` — handoff note.
- `STATUS.json` — atomic run state.
- `outputs/tables/` — cohort, balance, stability, model, cutoff, competing-risk,
  specification and diagnostic tables.
- `outputs/figures/` — FIG 01-09 style figures.
- `Report.pdf`, `REPORT.md`, `README.md`, `Poster.pdf`, `Group_09_Project.zip`.
- `Group_09_Project.zip` is the only submission archive; `Poster.pdf` is submitted
  separately. `STATUS.final.json` is included in the ZIP.

Simulation outputs are written under `outputs/sim/`, `data/interim/sim/`, and
`data/processed/sim/` so they cannot overwrite full-run artifacts.

## Known limitations

Observational associations only; matching balances measured W1 variables but cannot remove
unmeasured confounding. Clicks are a weak proxy for study time. The simulation sample and
small held-out test sets make point estimates noisy. Withdrawn is not a missing label.
The full run has more power but the same design limitations.

TMA-based appendix analyses were not attempted: the TMA+21/+28 window overlaps W1/W2 and
conditions on continued enrolment, the TMA+14 anchor varies by presentation, and first-TMA
scores are not available at the early cutoff. These are alternative, post-hoc estimands, not
early-warning evidence.
"""


def build_poster(ctx) -> str:
    t, stages = load_tables(ctx)
    models = t.get("t08_model_metrics", pd.DataFrame())
    s06 = stages.get("S06", {}).get("metrics", {})
    raw_smd = float(s06.get("raw_max_abs_smd", float("nan")))
    cem_smd = float(s06.get("binned_max_abs_smd", float("nan")))
    cem_support = float(s06.get("common_support_treated", float("nan")))
    cem_n = s06.get("n_matched", "--")
    ev_p = float(s06.get("e_value", float("nan")))
    s07p = stages.get("S07", {}).get("metrics", {})
    chosen_kp = s07p.get("k_chosen", "--") if isinstance(s07p, dict) else "--"
    chosen_arip = float(s07p.get("mean_ari", float("nan")))
    chosen_jp = float(s07p.get("min_cluster_jaccard", float("nan")))
    draft = ctx.mode == "simulation"
    pr = _mv(models, "M1", "main_test", "pr_auc")
    roc = _mv(models, "M1", "main_test", "roc_auc")
    roc_cv = _mv(models, "M1", "group_kfold", "roc_auc")
    roc_temp = _mv(models, "M1", "temporal", "roc_auc")
    roc_lomo = _mv(models, "M1", "lomo", "roc_auc")
    nc = t.get("t11_negative_controls", pd.DataFrame())
    rand = float(nc.loc[nc["control"].str.contains("random"), "value"].iloc[0]) if len(nc) else float("nan")
    pre_start_auc = (float(nc.loc[nc["control"] == "pre_start_only", "value"].iloc[0])
                     if len(nc) and (nc["control"] == "pre_start_only").any() else float("nan"))
    placebo_auc = (float(nc.loc[nc["control"].str.contains("placebo"), "value"].iloc[0])
                   if len(nc) and nc["control"].str.contains("placebo").any() else float("nan"))
    pr_m3 = _mv(models, "M3", "main_test", "pr_auc")
    pr_m5 = _mv(models, "M5", "main_test", "pr_auc")
    ctxp = t.get("t08_context_metrics", pd.DataFrame())
    ctx_m1 = (float(ctxp.loc[ctxp["model"] == "M-1_context", "pr_auc"].iloc[0])
              if len(ctxp) and (ctxp["model"] == "M-1_context").any() else float("nan"))
    ctx_m1c = (float(ctxp.loc[ctxp["model"] == "M1c_context+M1", "pr_auc"].iloc[0])
               if len(ctxp) and (ctxp["model"] == "M1c_context+M1").any() else float("nan"))
    ctx_m0c = (float(ctxp.loc[ctxp["model"] == "M0c_context+M0", "pr_auc"].iloc[0])
               if len(ctxp) and (ctxp["model"] == "M0c_context+M0").any() else float("nan"))
    ctx_roc_p = (float(ctxp.loc[ctxp["model"] == "M-1_context", "roc_auc"].iloc[0])
                 if len(ctxp) and (ctxp["model"] == "M-1_context").any() else float("nan"))
    _m0p = models[(models["model"] == "M0") & (models["split"] == "main_test") &
                  (models["metric"] == "pr_auc")] if len(models) else pd.DataFrame()
    no_skill_p = float(_m0p["positive_rate"].iloc[0]) if len(_m0p) else float("nan")
    s01m = stages.get("S01", {}).get("metrics", {})
    counts = {k: int(s01m[k]) for k in ("Pass", "Fail", "Withdrawn", "Distinction")
              if k in s01m}
    flowp = t.get("t02_cohort_flow", pd.DataFrame())
    c28p = flowp[flowp["cutoff"] == 28].iloc[0] if len(flowp) else None
    if draft and c28p is not None:
        n_total_p = int(c28p["n_total"])
        n_t1_p = int(c28p["n_t1"])
        n_withdrawn_p = int(c28p["n_withdrawn"])
        base_t1_p = float(c28p["base_rate_t1"])
    elif counts:
        n_total_p = sum(counts.values())
        n_passdist = counts.get("Pass", 0) + counts.get("Distinction", 0)
        n_t1_p = n_passdist + counts.get("Fail", 0)
        n_withdrawn_p = counts.get("Withdrawn", 0)
        base_t1_p = counts.get("Fail", 0) / max(n_t1_p, 1)  # Fail share
    else:
        n_total_p, n_t1_p, n_withdrawn_p, base_t1_p = 0, 0, 0, float("nan")
    incp = t.get("t09_incremental", pd.DataFrame())
    b2p = (incp[(incp["cutoff"] == 56) & (incp["cohort"] == "fixed") &
                (incp["method"] == "B2_offset")]
           if len(incp) else pd.DataFrame())
    b2_status_p = str(b2p["status"].iloc[0]) if len(b2p) else "not available"
    b2_cal_p = float(b2p["calibration_slope"].iloc[0]) if len(b2p) else float("nan")
    quizp = t.get("t11_quiz_sensitivity", pd.DataFrame())
    quiz_m1_p = (float(quizp.loc[quizp["model"] == "M1", "delta_pr_auc"].iloc[0])
                 if len(quizp) and (quizp["model"] == "M1").any() else float("nan"))
    quiz_m3_p = (float(quizp.loc[quizp["model"] == "M3", "delta_pr_auc"].iloc[0])
                 if len(quizp) and (quizp["model"] == "M3").any() else float("nan"))
    expop = t.get("t11_exposure_sensitivity", pd.DataFrame())
    binsp = t.get("t11_matching_bins", pd.DataFrame())
    placebop = t.get("t11_placebo_outcomes", pd.DataFrame())
    expo_lo_p = float(expop["d_clicks_diff"].min()) if len(expop) else float("nan")
    expo_hi_p = float(expop["d_clicks_diff"].max()) if len(expop) else float("nan")
    bins_lo_p = float(binsp["d_clicks_diff"].min()) if len(binsp) else float("nan")
    bins_hi_p = float(binsp["d_clicks_diff"].max()) if len(binsp) else float("nan")
    placebo_roc_p = (float(placebop["roc_auc"].iloc[0]) if len(placebop) else float("nan"))
    divp = t.get("t12_divergence", pd.DataFrame())
    divw = (divp["divergence_week"].iloc[0] if len(divp) else None)
    if divw is None or (isinstance(divw, float) and not np.isfinite(divw)):
        divw = "none in weeks 0--8"
    else:
        divw = str(int(divw))
        if int(divw) <= 3:
            divw += " (inside W1; early similarity is approximate)"
        else:
            divw += " (after W1)"
    draft_block = (r"\block{SIMULATION DRAFT}{This poster is from the 500-enrolment "
                   r"rehearsal run; numbers will change in the full run.}" if draft else "")
    left_extra = (
        r"\block{Cohort definitions}{" "\n"
        r"\small T1 = Fail (positive) vs Pass/Distinction, Withdrawn excluded (n="
        + f"{n_t1_p:,}" + r"; Fail rate " + f(100 * base_t1_p, 1) + r"\%). "
        r"T2 counts Fail or Withdrawn as positive; T3 keeps three classes. "
        r"The main cohort is not filtered on \texttt{date\_unregistration}; at-risk filtering "
        r"is a sensitivity analysis." "\n"
        r"}" "\n"
        r"\block{Matching and clustering}{" "\n"
        r"\small CEM matched " + str(cem_n) + r" T1 enrolments; coarsened max $|$SMD$|$ "
        + f(cem_smd) + r", raw max $|$SMD$|$ " + f(raw_smd) + r", treated support "
        + f(cem_support) + r", E-value " + f(ev_p) + r". k-means selected k="
        + str(chosen_kp) + r" (bootstrap ARI " + f(chosen_arip) + r", minimum Jaccard "
        + f(chosen_jp) + r"); clusters are descriptive." "\n"
        r"}" "\n"
    )
    mid_extra = (
        r"\block{Diagnostics and placebo checks}{" "\n"
        r"\small Quiz inclusion changes M1/M3 PR-AUC by " + f(quiz_m1_p) + r"/"
        + f(quiz_m3_p) + r". Exposure-definition CEM gives W2 click differences from "
        + f(expo_lo_p) + r" to " + f(expo_hi_p) + r"; seven bin grids range from "
        + f(bins_lo_p) + r" to " + f(bins_hi_p) + r". The repeater placebo reaches ROC-AUC "
        + f(placebo_roc_p) + r"." "\n"
        r"}" "\n"
    )
    ctx_bullet = (
        "Context-only PR-AUC " + f(ctx_m1) + " (ROC-AUC " + f(ctx_roc_p) +
        "; no-skill " + f(no_skill_p) + "); context+volume " + f(ctx_m0c) +
        "; context+M1 " + f(ctx_m1c) + "."
    )
    pre_note = ("a strong signal, not a clean control"
                if np.isfinite(pre_start_auc) and np.isfinite(no_skill_p) and
                pre_start_auc > no_skill_p else "not a clean control")
    placebo_note = ("a placebo failure, not a leak test"
                    if np.isfinite(placebo_auc) and np.isfinite(no_skill_p) and
                    placebo_auc > no_skill_p else "close to no-skill")
    right_extra = (
        r"\block{What changes the answer}{" "\n"
        r"\small Pre-start-only PR-AUC " + f(pre_start_auc) + r" is " + pre_note
        + r"; the future-window control is " + f(placebo_auc) + r" (" + placebo_note
        + r"). Random-label ROC-AUC is " + f(rand) + r". Context-only PR-AUC is "
        + f(ctx_m1) + r" vs no-skill " + f(no_skill_p) + r" (ROC-AUC " + f(ctx_roc_p)
        + r"), so the context contrast is descriptive." "\n"
        r"}" "\n"
        r"\block{Reproducibility}{" "\n"
        r"\small Frozen config hash " + _tex(str(ctx.cfg.hash_ref)[:12]) + r"\ldots; full "
        r"hash in \texttt{config/config.sha256}. Unit tests, staged S00--S12 logs, "
        r"\texttt{STATUS.final.json} and \texttt{validate\_run.py} output are included in the "
        r"submission archive." "\n"
        r"}" "\n"
    )
    poster = r"""\documentclass[25pt, a0paper, portrait, margin=0mm]{tikzposter}
\usepackage{graphicx}
\usepackage{booktabs}
\title{Same Engagement, Different Outcomes}
\author{Group 9: Jiayang Liu, Yanbin Chen, Yanshun Wang, Yuming Zhang}
\institute{Data Mining Course Project}
\begin{document}
\maketitle
""" + draft_block + r"""
\begin{columns}
\column{0.32}
\block{Motivation}{
Online learners can click the same amount and finish differently. We ask which temporal and
resource-use patterns distinguish Pass/Distinction from Fail among students with similar
first-month engagement. Withdrawn is a separate outcome, not a missing label.
}
\block{Data and leakage audit}{
OULAD: 32,593 enrolments, 10.65M VLE rows. Dates are relative to course start and include
negative days. The click stream does \textbf{not} stop at \texttt{date\_unregistration}:
1,272 withdrawn students have 29,050 post-unregistration click rows. We therefore keep the
main cohort and treat at-risk filtering as a sensitivity analysis.
}
\block{Design}{
\small W1 = days 0--27. W2 = 28 to min(55, first TMA$-$1), with a fixed 28--55 fallback when the
first TMA is before week 4. Strict features exclude scores, banked flags, final results and
post-cutoff clicks. Splits are by student; temporal and LOMO folds are student-disjoint.
CEM uses module $\times$ presentation exact strata and coarsened W1 clicks, active days and
entropy. Clustering uses W1 profiles with 300 bootstrap refits; models M0--M5 form a feature
ladder. A specification curve and diagnostics test robustness.
}
@@LEFT@@
\column{0.36}
\block{Main results}{
\begin{itemize}
\item M1 held-out test: PR-AUC %.3f, ROC-AUC %.3f.
\item Grouped CV ROC-AUC %.3f; 2013$\to$2014 %.3f; LOMO %.3f.
\item M3 strict PR-AUC %.3f; M5 %.3f; M4 (scores) is an upper bound.
\item %s
\item Matched trajectory divergence week: %s.
\item Random-label control ROC-AUC %.3f, close to chance.
\item CEM balance is exact on coarsened bins; raw max $|$SMD$|$ is %.3f and treated
support is %.3f.
\end{itemize}
}
\block{Figures}{
\includegraphics[width=\linewidth]{figures/f12_trajectory_divergence.png}\\
\includegraphics[width=\linewidth]{figures/f09_cutoff_curve.png}\\
\includegraphics[width=0.48\linewidth]{figures/f10_cumulative_incidence.png}
\includegraphics[width=0.48\linewidth]{figures/f07_pca_scatter.png}
}
@@MID@@
\column{0.32}
\block{Takeaways}{
\begin{itemize}
\item Total clicks are an entry point; distribution and composition add a small,
setting-dependent increment.
\item Row-level splitting leaks students; grouping changes the answer.
\item Withdrawn must be modelled separately.
\item Calibration and external validation matter more than a small random-split gain.
\end{itemize}
}
\block{Limitations}{
\small Observational associations only; measured W1 matching cannot remove unmeasured confounding.
Clicks are a weak proxy for study time. The rehearsal uses 500 enrolments. Pre-start and
future-window diagnostics are not null, so they are not clean leak detectors.
}
\block{Disclosures}{
Public OULAD data. No external collaborators. Claude Code (Anthropic) was used for code
scaffolding, pipeline execution, figure generation and drafting; the group is responsible for
the final submission.
}
@@RIGHT@@
\end{columns}
\block{Bottom line}{
\small Total clicks are an entry point, not an explanation. Early-window behaviour adds a
small and setting-dependent predictive increment; calibration, external validation and the
separate treatment of Withdrawn matter more than a single random-split score. All reported
contrasts are associations, not causal effects.
}
"""
    poster = poster % (pr, roc, roc_cv, roc_temp, roc_lomo, pr_m3, pr_m5,
                       ctx_bullet, divw, rand, raw_smd, cem_support)
    poster = (poster.replace("@@LEFT@@", left_extra)
                    .replace("@@MID@@", mid_extra)
                    .replace("@@RIGHT@@", right_extra))
    s02m = stages.get("S02", {}).get("metrics", {})
    poster = poster.replace(
        "1,272 withdrawn students have 29,050 post-unregistration click rows.",
        f"{int(s02m.get('students_with_post_click', 0))} withdrawn students have "
        f"{int(s02m.get('post_unregistration_click_rows', 0))} post-unregistration click rows.")
    if not draft:
        poster = poster.replace("The rehearsal uses 500 enrolments. ", "")
    return poster + r"\end{document}"


def citation_scan(text: str):
    """Reject literature claims, but not coincidental values from our own tables."""
    errors, warnings = [], []
    flat = text.replace("\n", " ")
    for bad in REJECTED_PHRASES:
        if bad in flat:
            errors.append(f"rejected phrase present: {bad}")
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    for bad in REJECTED_NUMBERS:
        for sent in sentences:
            if bad not in sent:
                continue
            if any(tok.lower() in sent.lower() for tok in CITATION_TOKENS):
                errors.append(f"rejected cited number present: {bad}")
            # Coincidental values from our own tables are not literature claims.
    for sent in sentences:
        if "0.7747" in sent:
            if ("Fail-or-Withdrawn" not in sent or
                    ("score" not in sent.lower() and "early score" not in sent.lower())):
                warnings.append("0.7747 appears without target/feature qualifiers")
    return errors, warnings
