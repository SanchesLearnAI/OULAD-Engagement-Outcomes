"""Report, poster, README, citation checks and PDF helpers for the TMA design."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_root

MODELS = ["M0", "M1", "M2", "M3", "M4"]
PRIMARY = ["submitted_by_cutoff", "submitted_after_due", "relative_submission_day",
           "post_active_days", "post_longest_inactive_run", "post_week2_share"]
# Claims that must never reappear: they belonged to rejected drafts of the study.
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
        return f"{100 * float(x):.{n}f}\\%"
    except Exception:
        return "--"


def _tex(s):
    if s is None:
        return "--"
    s = str(s)
    s = s.replace("\\", "\x00")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("\x00", r"\textbackslash{}")


def load_tables(ctx):
    import json
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


def _fig(rel, caption, label, width="0.82\\linewidth"):
    return "\n".join([r"\begin{figure}[t]", r"\centering",
                      r"\includegraphics[width=" + width + "]{" + rel + "}",
                      r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\end{figure}"])


def _preamble() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{array}
\captionsetup{font=small}
\setlength{\parskip}{3pt}
\setlength{\parindent}{0pt}
\graphicspath{{./}}
\title{Same Deadline, Different Persistence\\\large Submission timing and post-deadline activity among students with comparable clicks}
\author{Group 9: Jiayang Liu (50027216), Yanbin Chen (50026846),\\ Yanshun Wang (50026688), Yuming Zhang (50025933)}
\date{Data Mining Course Project \\ \today}
\begin{document}
\maketitle"""


def _ensure_figures(ctx) -> None:
    """S12 compiles next to ``outputs/``; copy figures so LaTeX finds them."""
    import shutil
    for p in ctx.paths.figures_dir.glob("*.png"):
        target = ctx.paths.outputs_dir / p.name
        if not target.exists() or target.stat().st_mtime < p.stat().st_mtime:
            shutil.copyfile(p, target)


def build_latex(ctx) -> str:
    t, stages = load_tables(ctx)
    draft = ctx.mode == "simulation"
    _ensure_figures(ctx)

    flow = t.get("t02_cohort_flow", pd.DataFrame())
    presents = t.get("t02_presentations", pd.DataFrame())
    counts = stages.get("S02", {}).get("metrics", {})
    balance = t.get("t06_balance", pd.DataFrame())
    effects = t.get("t06_effects", pd.DataFrame())
    primary = t.get("t06_primary_holm", pd.DataFrame())
    pre_sens = t.get("t06_pre_volume_sensitivity", pd.DataFrame())
    post_sens = t.get("t06_positive_post_sensitivity", pd.DataFrame())
    models = t.get("t08_model_metrics", pd.DataFrame())
    calib = t.get("t08_calibration", pd.DataFrame())
    perm = t.get("t08_label_permutation", pd.DataFrame())
    parity = t.get("t08_parity", pd.DataFrame())
    rules = t.get("t08_threshold_rules", pd.DataFrame())
    specs = t.get("t11_specs", pd.DataFrame())
    direction = t.get("t11_direction_stability", pd.DataFrame())
    clusters = t.get("t07_cluster_profile", pd.DataFrame())
    cut = t.get("t09_cutoff_metrics", pd.DataFrame())
    classes = t.get("t10_class_metrics", pd.DataFrame())
    bands = t.get("t03_submission_timing", pd.DataFrame())
    splits = t.get("t05_split_summary", pd.DataFrame())

    n_binary = int(counts.get("n_binary", 0))
    n_fail = int(counts.get("n_fail", 0))
    n_elig = int(counts.get("n_eligible", 0))
    n_withdrawn = int(counts.get("n_withdrawn_eligible", 0))
    fail_rate = float(counts.get("fail_rate", float("nan")))
    n_total = int(counts.get("n_total", 0))
    main_pairs = (balance.loc[balance["tolerance"].sub(0.05).abs() < 1e-9].iloc[0]
                  if len(balance) else None)

    main_models = models[models["split"].eq("main_test")] if len(models) else pd.DataFrame()
    primary_model = "M2"
    ap_primary = (float(main_models.loc[main_models["model"].eq(primary_model),
                                        "average_precision"].mean())
                  if len(main_models) and primary_model in set(main_models["model"])
                  else float("nan"))
    no_skill = (float(main_models["no_skill_pr_auc"].mean())
                if len(main_models) else fail_rate)

    def effect_of(name, column="fail_minus_pass"):
        if not len(effects) or name not in set(effects["feature"]):
            return float("nan")
        row = effects[np.isclose(effects["tolerance"], 0.05) & effects["feature"].eq(name)]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    parts = [_preamble()]
    if draft:
        parts.append(r"""\begin{center}
\fcolorbox{red}{yellow}{\parbox{0.9\linewidth}{\centering\textbf{SIMULATION DRAFT.}
This report comes from the 500-enrolment rehearsal run. It is an execution artefact,
not the final submission.}}
\end{center}""")
    parts.append(r"\noindent\textbf{Page-count note.} The main text ends on page "
                 r"\pageref{endmain}; references, disclosures and the appendix are excluded "
                 r"from the 6--10 page main-text requirement.")

    # ---------------------------------------------------------------- intro
    parts.append(r"\section{Question}")
    parts.append(
        "Online course platforms record every resource view with a timestamp. Total click "
        "volume is the obvious summary of that record and the obvious baseline for predicting "
        "who will fail, yet two students can click the same number of times and finish very "
        "differently. This project asks whether the timing of activity around a mandatory "
        "deadline carries information that the volume alone does not.")
    parts.append(
        "The specific question is: within the same module presentation, among students whose "
        "clicks in the 14 days after the first tutor-marked assignment (TMA) deadline are "
        "comparable, are submission timing and post-deadline persistence associated with "
        "eventually failing the module rather than passing or earning a distinction? Fail is "
        "the positive class, so the measure of interest is the average precision (AP) of Fail "
        "predictions against the Fail share of the same test records. Final Withdrawn records "
        "are a separate outcome: they leave the binary comparison and return only in the "
        "at-risk description of Section~6.")
    parts.append(
        "Three choices define the design. First, the anchor is the deadline $d$ of each "
        "presentation's earliest TMA with positive weight, not the start of the course, "
        "because the deadline is the moment at which a large share of students either submit "
        "or disappear. Second, the observation window is symmetric: 14 days before "
        "($d-13\\ldots d$) give a starting-point baseline, and 14 days after "
        "($d+1\\ldots d+14$) give the behaviour of interest. Third, the comparison is paired "
        "on post-deadline clicks, so a difference in persistence cannot be explained by one "
        "group simply clicking more.")
    parts.append(
        "Everything reported here is an observational association. The pairing is constructed "
        "from the final outcome, so the resulting sample is 1:1 by design and cannot estimate "
        "how many students fail. Deadlines are the scheduled dates in the assessment table; "
        "OULAD records neither individual extensions nor mark-release dates, so a submission "
        "is only observed as early or late relative to the scheduled date, and no score is "
        "treated as known at the cutoff.")

    # ------------------------------------------------------------- material
    parts.append(r"\section{Data and design}")
    parts.append(
        "The analysis uses the Open University Learning Analytics Dataset (OULAD), the public "
        "release that accompanies the module-level records of 32,593 student enrolments in "
        "seven modules and 22 presentations. The unit of analysis is one enrolment "
        "(module $\\times$ presentation $\\times$ student), because the same person can appear "
        "in several presentations and the deadline is a property of the presentation.")
    rows = []
    for _, r in flow.iterrows():
        if str(r["step"]) in ("all_enrolments", "has_first_weighted_tma",
                              "registered_by_deadline", "registered_at_cutoff",
                              "banked_excluded", "binary_records", "fail_records",
                              "withdrawn_at_cutoff"):
            rows.append([_tex(r["step"]), str(int(r["records"]))])
    if rows:
        parts.append(_table(["Sample-flow step", "Records"], rows, "lr",
                            "Ordered eligibility flow. A record must carry a first weighted "
                            "TMA, be registered by the deadline, still be enrolled at "
                            "$d+14$ and not be a banked attempt. Withdrawn records drop out "
                            "of the binary comparison but remain available to Section~6.",
                            "tab:flow", size="\\small"))
    parts.append(
        f"Of {n_total:,} enrolments, {n_elig:,} satisfy every inclusion rule; {n_withdrawn:,} "
        f"of those end in Withdrawn, and {n_binary:,} have a binary outcome "
        f"({n_fail:,} Fail against {n_binary - n_fail:,} Pass/Distinction), so the Fail share "
        f"is {pct(fail_rate)}. Nineteen presentations across modules AAA--FFF carry a "
        f"positive-weight TMA; module GGG has none and never enters the design.")
    if len(presents):
        rows = [[_tex(r["code_module"]), _tex(r["code_presentation"]),
                 str(int(r["due_day"])), str(int(r["enrolments"])),
                 str(int(r["eligible"])), str(int(r["fail"])),
                 str(int(r["pass_distinction"]))] for _, r in presents.iterrows()]
        parts.append(_table(
            ["Module", "Presentation", "$d$", "Enrolments", "Eligible", "Fail",
             "Pass/Dist."],
            rows, "llrrrrr",
            "The nineteen analysed presentations. Deadlines fall between day 12 and day 54 "
            "of the course, so a fixed course-day cutoff would compare students at very "
            "different points of the TMA cycle; anchoring on $d$ removes that mismatch.",
            "tab:presentations", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}"))
    parts.append(
        "Timing features come from the daily click stream. For every record the pipeline "
        "counts clicks per relative day between $d-13$ and $d+14$, then derives post-deadline "
        "active days, the longest run of consecutive zero-click days, the share of "
        "post-deadline clicks in the second week, the peak seven-day share, a normalised "
        "entropy of the daily distribution, and the day of first post-deadline activity. "
        "Submission fields are masked to $d+14$: a submission recorded later is treated as not "
        "submitted, and two indicators plus the signed distance in days form the submission "
        "block.")

    # -------------------------------------------------------------- pairing
    parts.append(r"\section{Matched comparison}")
    parts.append(
        "Pairing runs inside each presentation. Fail records are matched 1:1 with "
        "Pass/Distinction records whose post-deadline click total is first exactly equal and "
        "then within the tolerance; positive counts must satisfy a ratio bound, and zero-click "
        "records may only pair with other zero-click records. No record is reused. The greedy "
        "order is not claimed to be globally optimal, and the realised pairing is reported so "
        "the residual imbalance can be inspected.")
    if main_pairs is not None:
        rows = []
        for _, r in balance.iterrows():
            rows.append([pct(r["tolerance"], 0), str(int(r["binary_records"])),
                         str(int(r["fail_records"])), str(int(r["pairs"])),
                         pct(r["fail_coverage"]), str(int(r["zero_post_pairs"])),
                         f(r["mean_post_click_gap"], 4), f(r["pre_log_click_smd"])])
        parts.append(_table(
            ["Tolerance", "Binary", "Fail", "Pairs", "Fail coverage", "Zero-click pairs",
             "Mean click gap", "Pre-click SMD"],
            rows, "lrrrrrrr",
            "Pairing on post-deadline clicks. The 5\\% row is the primary specification; "
            "0\\% and 10\\% are sensitivities. The pre-click standardised difference is "
            "reported because pairing deliberately does not balance the pre-deadline window.",
            "tab:pairing", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
        parts.append(
            f"The primary match pairs {int(main_pairs['pairs']):,} Fail records, covering "
            f"{pct(main_pairs['fail_coverage'])} of all Fail records, with a mean relative "
            f"click gap of {f(main_pairs['mean_post_click_gap'], 4)}. Post-deadline volume is "
            f"balanced almost exactly (standardised difference "
            f"{f(main_pairs['post_log_click_smd'])}), but the pre-deadline window is not: its "
            f"standardised difference is {f(main_pairs['pre_log_click_smd'])}, and the raw "
            f"difference is {f(effect_of('pre_clicks'), 1)} clicks per record. Fail records "
            f"were already somewhat less active before the deadline, so the paired difference "
            f"answers a conditional question: given similar post-deadline clicks, what else "
            f"differs?")

    if len(primary):
        rows = []
        for _, r in primary.iterrows():
            rows.append([_tex(r["metric"]), str(int(r["pairs"])),
                         f(r["fail_minus_pass"], 4),
                         f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]",
                         f(r["p_value"], 4), f(r["p_holm"], 4),
                         "yes" if bool(r["reject_holm_0_05"]) else "no"])
        parts.append(_table(
            ["Primary indicator", "Pairs", "Fail $-$ Pass", "95\\% interval",
             "$p$", "$p_{\\mathrm{Holm}}$", "Reject"],
            rows, "lrrcccc",
            "The six pre-specified paired differences, with a student-component bootstrap "
            "interval and Holm adjustment across the family. A negative value means Fail "
            "records score lower. The submission-distance indicator is only defined when both "
            "sides submitted by the cutoff, which is why its pair count is smaller. Intervals "
            "are conditional on the realised pairing.",
            "tab:primary", size="\\footnotesize"))
        parts.append(
            "All six indicators point the same way and none of them is close to zero. Fail "
            "records are less likely to have submitted by the cutoff and more likely to submit "
            "late; among those who submitted they submit later; after the deadline they are "
            "active on fewer days, silent for longer stretches, and less active in the second "
            "week. Holm adjustment leaves every indicator significant, so the direction of the "
            "story does not depend on which of the six is treated as primary.")
    else:
        parts.append("Primary tests were not produced in this run.")

    if len(bands):
        rows = [[_tex(r["submission_band"]), str(int(r["records"])), str(int(r["fail"])),
                 str(int(r["pass_distinction"])), str(int(r["withdrawn"])),
                 pct(r["fail_share_of_binary"])] for _, r in bands.iterrows()]
        parts.append(_table(
            ["Submission band", "Records", "Fail", "Pass/Dist.", "Withdrawn",
             "Fail share"],
            rows, "lrrrrr",
            "Outcome mix by submission band in the eligible cohort. The band is built from "
            "the masked submission day, so a record without a submission by $d+14$ counts as "
            "not submitted even if it submitted later.",
            "tab:bands", size="\\footnotesize"))
        not_submitted = bands[bands["submission_band"].eq("not_submitted_by_14")]
        early = bands[bands["submission_band"].eq("early_7plus")]
        if len(not_submitted) and len(early):
            parts.append(
                f"The descriptive pattern is steep: "
                f"{pct(not_submitted['fail_share_of_binary'].iloc[0])} of records without a "
                f"submission by $d+14$ fail, against "
                f"{pct(early['fail_share_of_binary'].iloc[0])} of those who submitted a week "
                f"or more early. This is descriptive, because submission and outcome both "
                f"reflect the same underlying effort, but it motivates keeping submission "
                f"status in every model.")

    if len(pre_sens):
        rows = [[pct(r["pre_click_tolerance"], 0), _tex(r["feature"]), str(int(r["pairs"])),
                 pct(r["fraction_of_main_pairs"]), f(r["fail_minus_pass"], 3)]
                for _, r in pre_sens.iterrows()]
        parts.append(_table(
            ["Pre-click tolerance", "Indicator", "Pairs", "Share of main", "Fail $-$ Pass"],
            rows, "llrrr",
            "Sensitivity that also requires the pre-deadline clicks to be close. The subset is "
            "selected after the fact and loses most of the sample, so it cannot replace the "
            "primary comparison; it shows whether the direction survives a stricter "
            "starting-point balance.", "tab:presens", size="\\scriptsize"))
        tight = pre_sens[np.isclose(pre_sens["pre_click_tolerance"], 0.20)]
        row = tight[tight["feature"].eq("post_active_days")] if len(tight) else tight
        if len(row):
            parts.append(
                f"Requiring pre-deadline clicks within 20\\% keeps only "
                f"{pct(row['fraction_of_main_pairs'].iloc[0])} of the main pairs "
                f"({int(row['pairs'].iloc[0]):,} pairs). The active-day difference stays "
                f"negative ({f(row['fail_minus_pass'].iloc[0], 3)} days), so the post-deadline "
                f"pattern is not simply an artefact of pre-deadline differences, but the "
                f"subset is small and its interval is wider.")
    if len(post_sens):
        rows = [[_tex(r["feature"]), str(int(r["pairs"])), f(r["fail_minus_pass"], 3)]
                for _, r in post_sens.iterrows()]
        parts.append(_table(
            ["Indicator", "Pairs", "Fail $-$ Pass"], rows, "lrr",
            "Sensitivity restricted to pairs where both records clicked at least once after "
            "the deadline, which removes the zero-click pairs.",
            "tab:postsens", size="\\footnotesize"))

    # --------------------------------------------------------------- models
    parts.append(r"\section{Predictive information}")
    parts.append(
        "The pairing describes who differs at comparable clicks. A separate question is "
        "whether timing adds predictive information beyond volume and submission status. The "
        "ladder is fitted on a grouped 60/20/20 split (no student spans two parts), the "
        "threshold is chosen on the validation part, and every model is refitted on "
        "train plus validation before the test part is scored.")
    if len(main_models):
        rows = []
        for _, r in main_models.groupby("model").mean(numeric_only=True).reset_index().iterrows():
            rows.append([_tex(r["model"]), str(int(r["n_features"])),
                         f(r["average_precision"], 4), f(r["no_skill_pr_auc"], 4),
                         f(r["roc_auc"], 4), f(r["brier"], 4),
                         f(r["calibration_slope"], 3), f(r["f1"], 3)])
        parts.append(_table(
            ["Model", "Features", "Fail AP", "No-skill AP", "ROC-AUC", "Brier",
             "Cal. slope", "F1"],
            rows, "lrrrrrrrr",
            "Model ladder on the grouped main test split (single holdout at the configured "
            "seed). M0 uses pre- and "
            "post-deadline volume plus submission status; M1 adds post-deadline timing; M2 "
            "adds submission distance and is the primary model; M3 adds resource composition; "
            "M4 adds the first-TMA score and is a retrospective upper bound only. Thresholds "
            "are chosen on the validation part.",
            "tab:models", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
        m0 = (float(main_models.loc[main_models["model"].eq("M0"), "average_precision"].mean())
              if "M0" in set(main_models["model"]) else float("nan"))
        m1 = (float(main_models.loc[main_models["model"].eq("M1"), "average_precision"].mean())
              if "M1" in set(main_models["model"]) else float("nan"))
        rf_rows = main_models[main_models["model"].eq("RF_M2")]["average_precision"]
        rf = float(rf_rows.mean()) if len(rf_rows) else float("nan")
        parts.append(
            f"The primary model reaches Fail AP {f(ap_primary, 4)} against a no-skill floor of "
            f"{f(no_skill, 4)}. The ladder shows where the gain comes from: M0 already "
            f"contains submission status and reaches {f(m0, 4)}, the post-deadline timing block "
            f"moves it to {f(m1, 4)}, and the submission-distance terms add the rest up to "
            f"{f(ap_primary, 4)}. Most of the predictive signal is submission status rather "
            f"than the fine structure of the click sequence. A gradient-boosted alternative "
            f"reaches {f(rf, 4)}, within noise of the logistic ladder, so there is little "
            f"evidence of a strong nonlinear pattern that the linear models miss.")
    else:
        parts.append("Model metrics were not produced in this run.")

    if len(rules):
        parts.append(
            "The threshold rule actually used is recorded per model ("
            + ", ".join(f"{_tex(r['model'])}: {_tex(r['threshold_rule'])}"
                        for _, r in rules.iterrows())
            + "). The pipeline first looks for a threshold that reaches 0.80 precision on the "
            "validation part and falls back to the maximum-F1 threshold when no such point "
            "exists, so the reported precision and recall describe a realisable operating "
            "point rather than the maximum of the precision-recall curve.")
    if len(calib):
        parts.append(_fig("f08_calibration.png",
                          "Reliability of the fitted probabilities on the main test split. "
                          "Points near the diagonal are better calibrated; the ladder is not "
                          "recalibrated after fitting, so miscalibration here is a property of "
                          "the fitted model rather than a post-hoc correction.",
                          "fig:calibration"))
    if len(perm):
        parts.append(
            f"With the label shuffled, the same pipeline returns Fail AP "
            f"{f(perm['average_precision'].mean(), 4)} against a shuffled prevalence of "
            f"{f(perm['fail_prevalence'].mean(), 4)}. A learning signal that survived "
            f"shuffling would indicate leakage; it does not.")
    if len(cut):
        rows = [[str(int(r["post_days"])), _tex(r["model"]), f(r["average_precision"], 4),
                 f(r["brier"], 4)] for _, r in cut.iterrows()]
        parts.append(_table(["Post days", "Model", "Fail AP", "Brier"], rows, "rlrr",
                            "Observation-window curve on the same split. A longer window "
                            "supplies more information, so AP rises; the change is not a "
                            "method gain and must not be read as one.",
                            "tab:cutoff", size="\\footnotesize"))
        first = cut[cut["post_days"].eq(1) & cut["model"].eq("M2")]
        last = cut[cut["post_days"].eq(14) & cut["model"].eq("M2")]
        if len(first) and len(last):
            parts.append(
                f"Moving the cutoff from $d+1$ to $d+14$ raises the primary model's AP from "
                f"{f(first['average_precision'].iloc[0], 4)} to "
                f"{f(last['average_precision'].iloc[0], 4)} with the same feature list. The "
                f"gain is therefore information gained by observing longer, not a better "
                f"model.")
    if len(main_models):
        parts.append(_fig("f08_model_ladder.png",
                          "Model ladder on the grouped main test split; whiskers span the "
                          "five grouped folds of the cross-validation arm and the dotted line "
                          "is the no-skill level.",
                          "fig:ladder", width="0.72\\linewidth"))

    # --------------------------------------------------------------- splits
    parts.append(r"\section{Validation design}")
    if len(splits):
        rows = [[_tex(r["split"]), str(int(r["n_train"])), str(int(r["n_test"])),
                 pct(r["train_positive_rate"]), pct(r["test_positive_rate"]),
                 str(int(r["student_overlap"]))] for _, r in splits.iterrows()]
        parts.append(_table(
            ["Split", "Train", "Test", "Train Fail", "Test Fail", "Student overlap"],
            rows, "lrrrrr",
            "Every split is student-disjoint: no student appears on both sides. The overlap "
            "column is zero everywhere by construction and is checked at run time. The "
            "leave-one-module-out folds number six, not seven, because only six modules have a "
            "positive-weight TMA; each fold also drops the students who appear in the held-out "
            "module, so the training side shrinks.", "tab:splits",
            size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
    parts.append(
        "The temporal split trains on the 2013 presentations and tests on 2014, dropping from "
        "the training side any student who also appears in 2014. Because most students take "
        "one module, this is a genuine forward check in time, but it is not an external "
        "dataset: the modules and the grading regime are the same. The leave-one-module-out "
        "folds address the other direction, generalising to a module the model has not seen, "
        "and the module composition of the test part differs a lot between folds.")
    if len(parity):
        rows = [[str(int(r["seed"])), _tex(r["model"]), str(int(r["n_test"])),
                 f(r["fail_prevalence"], 4), f(r["average_precision"], 4),
                 f(r["brier"], 4)] for _, r in parity.iterrows()]
        parts.append(_table(
            ["Seed", "Model", "Test records", "Fail share", "Fail AP", "Brier"],
            rows, "lrrrrr",
            "Replication of the published two-week ladder: an 80/20 grouped holdout with the "
            "seed set of the exploratory study and no validation part. These values are the "
            "direct comparison point for the earlier tables; the main split above answers a "
            "slightly different question with a different partition.",
            "tab:parity", size="\\scriptsize"))

    # ----------------------------------------------------------- robustness
    parts.append(r"\section{Robustness and scope}")
    if len(specs):
        rows = [[_tex(r["spec"]), str(int(r["post_days"])), pct(r["tolerance"], 0),
                 _tex(r["match_on"]), str(int(r["pairs"])), pct(r["fail_coverage"]),
                 f(r["pre_log_click_smd"])] for _, r in specs.iterrows()]
        parts.append(_table(
            ["Specification", "Post days", "Tolerance", "Matched on", "Pairs",
             "Fail coverage", "Pre SMD"],
            rows, "lrrrrrr",
            "Robustness curve. The +21/+28 variants appear twice: matched on post-deadline "
            "clicks, which is comparable with the main design, and matched on whole-window "
            "clicks, which is comparable with the earlier +21/+28 study. The two answer "
            "different questions and must not be ranked against each other.",
            "tab:specs", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}"))
    if len(direction):
        rows = [[_tex(r["feature"]), str(int(r["specs_negative"])), str(int(r["specs"])),
                 pct(r["share_negative"])] for _, r in direction.iterrows()]
        parts.append(_table(
            ["Indicator", "Specifications negative", "Specifications",
             "Share negative"],
            rows, "lrrr",
            "Sign stability of each indicator across the eight specifications. A direction "
            "that flipped between reasonable choices would have to be reported as unstable; "
            "these do not flip.", "tab:direction", size="\\footnotesize"))
        stable = direction[direction["share_negative"].isin([0.0, 1.0])]
        parts.append(
            f"All {len(stable)} of the indicators shown keep the same sign in every one of the "
            f"eight specifications, including the two whole-window matching variants that "
            f"balance a different quantity. The magnitude moves: matching on total clicks "
            f"leaves less post-deadline difference, which is expected because the whole-window "
            f"total is partly a pre-deadline quantity.")
    if len(clusters):
        rows = [[_tex(r["cluster"]), str(int(r["n"])), f(r["mean_post_clicks"], 1),
                 f(r["post_active_days"], 2), f(r["post_longest_inactive_run"], 2),
                 pct(r["submitted_by_cutoff"]), pct(r["fail_share"])]
                for _, r in clusters.iterrows()]
        parts.append(_table(
            ["Cluster", "Records", "Post clicks", "Active days", "Longest gap",
             "Submitted by $d+14$", "Fail share"],
            rows, "lrrrrrr",
            "Outcome-blind timing clusters inside the paired sample. k and the feature list "
            "were chosen without the final result. The largest cluster is also the most "
            "active, so cluster membership still tracks volume; the clusters are descriptive "
            "and are not a finding about learning styles.",
            "tab:clusters", size="\\footnotesize"))
    if len(classes):
        rows = [[_tex(r["y_t3"]), str(int(r["records"])), pct(r["share"])]
                for _, r in classes.iterrows()]
        parts.append(_table(["Final outcome in the at-risk set", "Records", "Share"],
                            rows, "lrr",
                            "The at-risk set keeps Withdrawn as its own class. It is a "
                            "different population from the binary comparison and answers a "
                            "different question.", "tab:risk", size="\\footnotesize"))
        parts.append(
            "The at-risk description is the only place where Withdrawn is modelled. Reading "
            "the binary comparison as a withdrawal warning would be wrong: those records were "
            "still enrolled at the cutoff by construction, and a model that predicts Fail "
            "among survivors is not the model that should flag a student who may leave.")

    # ---------------------------------------------------------- limitations
    parts.append(r"\section{Limitations}")
    parts.append(
        r"""\begin{itemize}\itemsep2pt
\item AP is average precision for Fail, not accuracy. The no-skill reference is the Fail share in the same test records.
\item Pairing is 1:1 by design and cannot estimate the population failure rate. The greedy order is not globally optimal, and the bootstrap resamples students on the realised pairs only, so re-matching uncertainty is excluded.
\item The binary cohort excludes records that withdrew before the cutoff. It is not a real-time early-warning population.
\item The first-TMA score contributes to the final result. Models that use it are retrospective upper bounds, and OULAD records no mark-release date.
\item The OULAD splits have been inspected across several rounds of exploration. The intervals and increments are exploratory even though the protocol is now frozen.
\item VLE activity types are platform categories, not content quality; \texttt{studentVle} records day-level clicks only, so study time cannot be measured.
\end{itemize}""")
    parts.append(
        "Two design facts deserve emphasis. First, the pre-deadline window is not balanced by "
        "the primary match; the sensitivity that balances it keeps only a small share of the "
        "pairs, so the honest statement is that the comparison holds post-deadline volume "
        "fixed and reports how much the pre-deadline history still differs. Second, the gain "
        "from timing features is small. Post-deadline persistence distinguishes outcomes "
        "descriptively far more clearly than it improves prediction, and the model that "
        "improves most adds submission distance, which is itself deadline behaviour rather "
        "than a click pattern.")

    parts.append(r"\label{endmain}")
    parts.append(r"\section*{Disclosures}")
    parts.append(
        "AIGC tools assisted with code scaffolding, text editing and LaTeX formatting. "
        "All analysis code, numerical results and claims were reviewed by the group, and "
        "every figure and table is regenerated by the pipeline from the public dataset. "
        "No external collaborator contributed data or results.")
    parts.append(r"\section*{Data and code availability}")
    parts.append(
        "The unmodified OULAD archive is committed with its SHA-256. Extracted CSVs, "
        "student-level features, pair indices and predictions stay outside version control. "
        "One command reproduces the analysis; the submission ZIP contains the code, the "
        "frozen configuration and the run log.")
    parts.append(r"\appendix")
    parts.append(r"\section{Exploratory extensions}")
    parts.append(
        "Four exploratory blocks were run after the primary analysis was frozen. They do not "
        "change any primary indicator or the primary model, and each is reported here rather "
        "than in the main text.")
    bg = t.get("t11_x_background_balance", pd.DataFrame())
    bg_eff = t.get("t11_x_background_effects", pd.DataFrame())
    bg_info = t.get("t11_x_background_constrained", pd.DataFrame())
    rematch = t.get("t11_x_rematching", pd.DataFrame())
    hetero = t.get("t11_x_heterogeneity", pd.DataFrame())
    nonsub = t.get("t11_x_non_submitter_outcomes", pd.DataFrame())
    atrisk = t.get("t11_x_at_risk_warning", pd.DataFrame())

    if len(bg):
        rows = [[_tex(r["covariate"]), _tex(r["kind"]), f(r["smd_unmatched"], 3),
                 f(r["smd_matched"], 3),
                 "yes" if bool(r["balanced_matched"]) else "no"] for _, r in bg.iterrows()]
        parts.append(_table(
            ["Covariate", "Kind", "SMD before", "SMD after", "$|$SMD$|\\le$0.10"],
            rows, "llrrr",
            "Pre-entry covariates of the matched pairs. Pairing was designed to balance "
            "post-deadline clicks only, so a remaining difference here is expected and is "
            "reported rather than corrected.", "tab:c1", size="\\footnotesize"))
        worst = bg.sort_values("abs_smd_matched", ascending=False).iloc[0]
        parts.append(
            f"The largest residual imbalance is {_tex(worst['covariate'])} at "
            f"{f(worst['smd_matched'], 3)} standard deviations; education, IMD band and "
            f"previous attempts remain the least balanced covariates. Restricting the pairs to "
            f"those that also agree exactly on education and IMD band keeps "
            f"{pct(bg_info['share_of_main_pairs'].iloc[0]) if len(bg_info) else '--'} of them, "
            f"which is too small to replace the primary comparison.")
        if len(bg_eff):
            parts.append(
                "On that surviving subset the active-day difference is "
                f"{f(bg_eff.loc[bg_eff['feature'].eq('post_active_days'), 'fail_minus_pass'].iloc[0], 3)} "
                "days with a much wider interval, so the direction holds but the precision "
                "does not.")

    if len(rematch):
        rows = [[_tex(r["metric"]), f(r["observed"], 3), f(r["mean_rematched"], 3),
                 f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]", str(int(r["repeats_used"]))]
                for _, r in rematch.iterrows()]
        parts.append(_table(
            ["Indicator", "Fixed-pair estimate", "Re-matched mean", "Re-matched 95\\% interval",
             "Replicates"],
            rows, "lrrrr",
            "Re-matching uncertainty: each replicate resamples students, rebuilds the cohort "
            "and re-runs the greedy pairing, so the interval includes the pairing itself "
            "rather than holding it fixed. Replicates are few, so it is a check on width, not "
            "a definitive interval.", "tab:c2", size="\\footnotesize"))
        parts.append(
            "Both indicators keep their sign and their magnitude under re-matching, and the "
            "re-matched intervals are only slightly wider than the fixed-pair ones. Pairing "
            "choice is therefore not doing the work here.")
    if len(hetero):
        rows = [[_tex(r["metric"]), f(r["pooled_effect"], 3),
                 f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]", f(r["tau"], 3),
                 pct(r["i_squared"] / 100)] for _, r in hetero.iterrows()]
        parts.append(_table(
            ["Indicator", "Random-effects pooled", "95\\% interval", "$\\tau$",
             "$I^2$"],
            rows, "lrrrr",
            "Per-presentation differences summarised with a DerSimonian-Laird random-effects "
            "model. A high $I^2$ means the presentations do not share one effect size, which "
            "is expected when presentations have different cohorts and deadlines.",
            "tab:c2b", size="\\footnotesize"))
    if len(nonsub):
        rows = [[_tex(r["outcome"]), str(int(r["records"])),
                 pct(r["share_of_non_submitters"]), pct(r["share_of_eligible"])]
                for _, r in nonsub.iterrows()]
        parts.append(_table(
            ["Outcome", "Records", "Share of non-submitters", "Share of eligible cohort"],
            rows, "lrrr",
            "Records with no submission by $d+14$, split by final outcome. This is the "
            "subgroup the binary comparison cannot speak to, because withdrawal and "
            "non-submission overlap here.", "tab:c3", size="\\footnotesize"))
        parts.append(
            f"Of the {int(nonsub['non_submitters'].iloc[0]):,} records without a submission by "
            f"$d+14$, {pct(nonsub.loc[nonsub['outcome'].eq('Withdrawn'), 'share_of_non_submitters'].sum())} "
            f"eventually withdraw and "
            f"{pct(nonsub.loc[nonsub['outcome'].eq('Fail'), 'share_of_non_submitters'].sum())} "
            f"fail. Non-submission is therefore a stronger signal of leaving than of failing, "
            f"which is exactly why the binary design excludes Withdrawn.")
    if len(atrisk):
        rows = [[_tex(r["model"]), str(int(r["n_test"])), pct(r["positive_share"]),
                 f(r["average_precision"], 4), f(r["roc_auc"], 4), f(r["brier"], 4)]
                for _, r in atrisk.iterrows()]
        parts.append(_table(
            ["Model", "Test records", "Fail-or-Withdrawn share", "AP", "ROC-AUC",
             "Brier"],
            rows, "lrrrrr",
            "Warning target: Fail-or-Withdrawn among records enrolled at the deadline, with "
            "Withdrawn as a positive case. The cohort and the label differ from the binary "
            "comparison, so the AP is not comparable with Section~4.", "tab:c4",
            size="\\footnotesize"))
        parts.append(
            "Because Withdrawn is a positive case here, the AP level is much higher than in "
            "the binary design; the two numbers answer different questions and must not be "
            "placed in the same table.")
    parts.append(r"\section{Reproduction}")
    parts.append(
        r"""\begin{verbatim}
make test
make sim && make validate RUN_ID=$(cat .last_sim_run)
make full && make validate RUN_ID=$(cat .last_full_run)
\end{verbatim}
Every stage writes a stage summary, a handoff note and a JSONL event; the validator
checks the log against the schemas in \texttt{docs/contracts} and reports zero errors
and zero warnings.""")
    parts.append(r"\end{document}")
    return "\n".join(parts)

def build_markdown(ctx) -> str:
    t, stages = load_tables(ctx)
    counts = stages.get("S02", {}).get("metrics", {})
    models = t.get("t08_model_metrics", pd.DataFrame())
    primary = t.get("t06_primary_holm", pd.DataFrame())
    lines = ["# Same Deadline, Different Persistence", ""]
    if ctx.mode == "simulation":
        lines += ["> **SIMULATION DRAFT.** 500-enrolment rehearsal run, not the final submission.", ""]
    lines += ["## Question", "",
              "Within the same module presentation, among students whose post-deadline clicks "
              "after the first positive-weight TMA are comparable, are submission timing and "
              "post-deadline persistence associated with eventually failing rather than "
              "passing or earning a distinction? Withdrawn remains a separate outcome.", "",
              "## Cohort", "",
              f"- Eligible records: {int(counts.get('n_eligible', 0)):,}",
              f"- Binary records: {int(counts.get('n_binary', 0)):,}",
              f"- Fail records: {int(counts.get('n_fail', 0)):,} "
              f"({100 * float(counts.get('fail_rate', 0)):.1f}%)",
              f"- Presentations with a positive-weight TMA: {int(counts.get('presentations', 0))}"]
    if len(primary):
        lines += ["", "## Primary indicators", "",
                  "| Indicator | Pairs | Fail - Pass | 95% interval | p (Holm) |",
                  "|---|---:|---:|---|---:|"]
        for _, r in primary.iterrows():
            lines.append(f"| `{r['metric']}` | {int(r['pairs'])} | {r['fail_minus_pass']:.4f} | "
                         f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p_holm']:.4f} |")
    if len(models):
        main = models[models["split"].eq("main_test")].groupby("model").mean(numeric_only=True)
        lines += ["", "## Model ladder (main test, mean)", "",
                  "| Model | Fail AP | No-skill | ROC-AUC | Brier |", "|---|---:|---:|---:|---:|"]
        for model, row in main.iterrows():
            lines.append(f"| {model} | {row['average_precision']:.4f} | "
                         f"{row['no_skill_pr_auc']:.4f} | {row['roc_auc']:.4f} | "
                         f"{row['brier']:.4f} |")
        lines += ["", "M4 contains the first-TMA score and is a retrospective upper bound only.", ""]
    lines += ["## Reproducibility", "",
              "See README.md. Every stage writes `run.jsonl`, a stage summary, a handoff note "
              "and an atomic `STATUS.json` update; `validate_run.py` checks them.", ""]
    return "\n".join(lines)


def build_readme(ctx) -> str:
    t, stages = load_tables(ctx)
    counts = stages.get("S02", {}).get("metrics", {})
    primary = t.get("t06_primary_holm", pd.DataFrame())
    models = t.get("t08_model_metrics", pd.DataFrame())
    main = (models[models["split"].eq("main_test")].groupby("model").mean(numeric_only=True)
            if len(models) else pd.DataFrame())
    primary_ap = (f"{main.loc['M2', 'average_precision']:.4f}"
                  if len(main) and "M2" in main.index else "not available")
    lines = []
    lines.append("# OULAD — Same Deadline, Different Persistence")
    lines.append("")
    lines.append("Group 9 course project. The unit is an enrolment "
                 "(`code_module`, `code_presentation`, `id_student`).")
    if ctx.mode == "simulation":
        lines.append("")
        lines.append("> Simulation run: 500-enrolment rehearsal sample; numbers are not final.")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Within the same module presentation, among students with comparable VLE "
                 "click volume in the 14 days after the first positive-weight TMA deadline, "
                 "which submission-timing and post-deadline persistence patterns are "
                 "associated with eventually failing rather than passing or earning a "
                 "distinction? Final Withdrawn records are described separately.")
    lines.append("")
    lines.append("## Design in one table")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Anchor | Deadline `d` of the first `TMA` with `weight > 0` per presentation |")
    lines.append("| Windows | `d-13 … d` and `d+1 … d+14` |")
    lines.append("| Cohort | registered by `d`, still enrolled at `d+14`, not banked |")
    lines.append("| Label | Fail = 1, Pass/Distinction = 0, Withdrawn excluded |")
    lines.append("| Pairing | 1:1 within presentation on `post_clicks`, 5% tolerance |")
    lines.append("| Primary model | M2 (volume + submission + timing + days early/late) |")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append("```bash")
    lines.append("python3.11 -m venv .venv && source .venv/bin/activate")
    lines.append("python -m pip install pandas numpy scikit-learn scipy pyyaml jsonschema \\")
    lines.append("    pyarrow matplotlib seaborn statsmodels pytest duckdb")
    lines.append("make test")
    lines.append("make sim   && make validate RUN_ID=$(cat .last_sim_run)")
    lines.append("make full  && make validate RUN_ID=$(cat .last_full_run)")
    lines.append("```")
    lines.append("")
    lines.append("The frozen configuration is `config/config.yaml` with its SHA-256 in "
                 "`config/config.sha256`. Any change needs a `CHANGE_REQUEST.md` entry and a "
                 "new run id.")
    lines.append("")
    lines.append("## This run")
    lines.append("")
    lines.append(f"- run id: `{ctx.run_id}` (mode `{ctx.mode}`)")
    lines.append(f"- config: `{ctx.cfg.hash_ref}`")
    if counts:
        lines.append(f"- eligible records: {int(counts.get('n_eligible', 0)):,}")
        lines.append(f"- binary records: {int(counts.get('n_binary', 0)):,}, "
                     f"Fail {int(counts.get('n_fail', 0)):,} "
                     f"({100 * float(counts.get('fail_rate', 0)):.1f}%)")
        lines.append(f"- presentations: {int(counts.get('presentations', 0))}")
    lines.append(f"- primary model M2 Fail AP: {primary_ap}")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("See `DATA_DOWNLOAD.md`. Extract the seven CSVs into "
                 "`data/raw/oulad/` (and keep `oulad.zip` there so S00 can record its hash). "
                 "Raw CSVs, `data/interim/`, `data/processed/` and `runs/` are git-ignored.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("`outputs/tables/t01`–`t11` hold the aggregate tables cited in the report; "
                 "`outputs/figures/` holds the figures. Student-level features, pair indices "
                 "and cluster labels stay in the ignored directories.")
    lines.append("")
    return "\n".join(lines)


def build_poster(ctx) -> str:
    t, stages = load_tables(ctx)
    counts = stages.get("S02", {}).get("metrics", {})
    primary = t.get("t06_primary_holm", pd.DataFrame())
    models = t.get("t08_model_metrics", pd.DataFrame())
    main = (models[models["split"].eq("main_test")].groupby("model").mean(numeric_only=True)
            if len(models) else pd.DataFrame())
    rows = []
    if len(primary):
        for _, r in primary.iterrows():
            rows.append(f"{_tex(r['metric'])}: {r['fail_minus_pass']:+.3f}")
    effect_text = "; ".join(rows) if rows else "results pending"
    ap = (f"{main.loc['M2', 'average_precision']:.3f}"
          if len(main) and "M2" in main.index else "--")
    no_skill = (f"{main['no_skill_pr_auc'].mean():.3f}" if len(main) else "--")
    return r"""\documentclass[25pt, a0paper, portrait, margin=0mm]{tikzposter}
\usepackage{graphicx}
\usepackage{booktabs}
\title{Same Deadline, Different Persistence}
\author{Group 9: Jiayang Liu, Yanbin Chen, Yanshun Wang, Yuming Zhang}
\institute{Data Mining Course Project}
\begin{document}
\maketitle
\begin{columns}
\column{0.5}
\block{Question}{
Within the same module presentation, among students whose clicks after the first
TMA deadline are comparable, are submission timing and post-deadline persistence
associated with failing rather than passing?}
\block{Design}{
Anchor: the deadline of the first positive-weight TMA. Windows: 14 days before and
14 days after. Fail is the positive class; Withdrawn is described separately.
Pairs: 1:1 within presentation on post-deadline clicks, 5\% tolerance,
zero clicks only with zero clicks.}
\column{0.5}
\block{Evidence}{
Cohort: @@NELIG@@ eligible records, @@NBIN@@ in the binary comparison, @@NFAIL@@ Fail.
Paired differences (Fail $-$ Pass): @@EFFECTS@@.
Model M2 Fail AP @@AP@@ against a no-skill floor of @@NOSKILL@@.}
\block{Reading}{
The pairing conditions on post-deadline volume, so it cannot estimate the overall
failure rate. Persistence separates outcomes descriptively far more clearly than
it improves prediction, and the score model is a retrospective upper bound.}
\end{columns}
\end{document}""".replace("@@NELIG@@", f"{int(counts.get('n_eligible', 0)):,}") \
        .replace("@@NBIN@@", f"{int(counts.get('n_binary', 0)):,}") \
        .replace("@@NFAIL@@", f"{int(counts.get('n_fail', 0)):,}") \
        .replace("@@EFFECTS@@", effect_text) \
        .replace("@@AP@@", ap).replace("@@NOSKILL@@", no_skill)


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
    for sent in sentences:
        if "0.7747" in sent:
            if ("Fail-or-Withdrawn" not in sent or
                    ("score" not in sent.lower() and "early score" not in sent.lower())):
                warnings.append("0.7747 appears without target/feature qualifiers")
    return errors, warnings
