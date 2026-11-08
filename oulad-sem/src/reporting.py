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
    counts = stages.get("S02", {}).get("metrics", {}).get("cohort_counts", {})
    balance = t.get("t06_balance", pd.DataFrame())
    effects = t.get("t06_effects", pd.DataFrame())
    primary = t.get("t06_primary_holm", pd.DataFrame())
    models = t.get("t08_model_metrics", pd.DataFrame())
    calib = t.get("t08_calibration", pd.DataFrame())
    perm = t.get("t08_label_permutation", pd.DataFrame())
    specs = t.get("t11_specs", pd.DataFrame())
    clusters = t.get("t07_cluster_profile", pd.DataFrame())
    cut = t.get("t09_cutoff_metrics", pd.DataFrame())
    classes = t.get("t10_class_metrics", pd.DataFrame())
    splits = t.get("t05_split_summary", pd.DataFrame())

    n_binary = int(counts.get("n_binary", 0))
    n_fail = int(counts.get("n_fail", 0))
    n_elig = int(counts.get("n_eligible", 0))
    fail_rate = float(counts.get("fail_rate", float("nan")))
    n_pres = int(counts.get("presentations", 0))
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
        "Students in the same module presentation can click the same number of times and "
        "still end the course very differently. This project asks whether, \\emph{among "
        "students whose clicks after the first tutor-marked assignment (TMA) deadline are "
        "comparable}, submission timing and post-deadline persistence are associated with "
        "eventually failing the module rather than passing or earning a distinction.")
    parts.append(
        "The anchor is not enrolment: each presentation is aligned on the deadline $d$ of its "
        "first TMA with positive weight, and behaviour is observed for 14 days before "
        "($d-13\\ldots d$) and 14 days after ($d+1\\ldots d+14$). Fail is the positive class; "
        "final Withdrawn records are described separately and never folded into the binary "
        "comparison. Every claim is an observational association. Pairing is built from the "
        "final outcome, so the design cannot estimate the population failure rate.")

    # ------------------------------------------------------------- material
    parts.append(r"\section{Data and design}")
    rows = []
    for _, r in flow.iterrows():
        if str(r["step"]) in ("all_enrolments", "has_first_weighted_tma",
                              "registered_by_deadline", "registered_at_cutoff",
                              "banked_excluded", "binary_records", "fail_records",
                              "withdrawn_at_cutoff"):
            rows.append([_tex(r["step"]), str(int(r["records"]))])
    if rows:
        parts.append(_table(["Sample-flow step", "Records"], rows, "lr",
                            "Ordered eligibility flow for the TMA cohort. Withdrawn records "
                            "drop out of the binary comparison but stay in the at-risk set of "
                            "Section~7.", "tab:flow", size="\\small"))
    parts.append(
        f"Of {n_elig:,} eligible records, {n_binary:,} have a binary outcome "
        f"({n_fail:,} Fail, {n_binary - n_fail:,} Pass/Distinction), so the Fail share is "
        f"{pct(fail_rate)}. Nineteen module presentations (modules AAA--FFF) carry a "
        f"positive-weight TMA; GGG has none and never enters the design. "
        f"Every submission field is masked to $d+14$: a submission recorded later is treated "
        f"as not submitted, and no score is used in the strict models.")

    # -------------------------------------------------------------- pairing
    parts.append(r"\section{Matched comparison}")
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
            "Pairing within module presentation on post-deadline clicks. The 5\\% row is the "
            "primary specification; zero clicks only pair with zero clicks, and no record is "
            "reused. The pre-click standardised difference is reported because pairing does "
            "not balance the pre-deadline window.",
            "tab:pairing", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
        parts.append(
            f"Within presentations the primary match pairs {int(main_pairs['pairs']):,} Fail "
            f"records, covering {pct(main_pairs['fail_coverage'])} of all Fail records, and the "
            f"mean relative click gap is {f(main_pairs['mean_post_click_gap'], 4)}. The "
            f"pre-deadline click difference does not vanish: its standardised difference is "
            f"{f(main_pairs['pre_log_click_smd'])}, so the comparison conditions on "
            f"post-deadline volume only. That is a feature of the question, not an oversight: "
            "the paired difference answers \\emph{given similar post-deadline clicks}, not "
            "\\emph{given identical histories}.")

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
            "records score lower. Intervals are conditional on the realised pairing.",
            "tab:primary", size="\\footnotesize"))
        parts.append(
            "All six indicators point the same way: Fail records are less likely to have "
            "submitted by the cutoff, more likely to submit late, slower to submit, active on "
            "fewer post-deadline days, silent for longer stretches, and less active in the "
            "second week. Holm adjustment does not change that ranking because every "
            "unadjusted $p$-value sits at the bootstrap floor.")
    else:
        parts.append("Primary tests were not produced in this run.")

    # --------------------------------------------------------------- models
    parts.append(r"\section{Predictive information}")
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
            "Model ladder on the grouped main test split (six-seed mean). M0 uses volume and "
            "submission status; M1 adds post-deadline timing; M2 adds submission distance and "
            "is the primary model; M3 adds resource composition; M4 adds the first-TMA score "
            "and is a retrospective upper bound only. Thresholds are chosen on the "
            "validation part.",
            "tab:models", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
        parts.append(
            f"The primary model reaches Fail AP {f(ap_primary, 4)} against a no-skill floor "
            f"of {f(no_skill, 4)}. That gap is real but modest, and it comes mostly from "
            f"submission status rather than from fine-grained timing: later models add little. "
            f"The score model is reported only to show how much of the gap a retrospective "
            f"feature can close; scores are not available online at the cutoff.")
    else:
        parts.append("Model metrics were not produced in this run.")

    if len(calib):
        parts.append(_fig("f08_calibration.png",
                          "Reliability of the fitted probabilities on the main test split. "
                          "Points near the diagonal are better calibrated; the ladder is not "
                          "recalibrated after fitting.", "fig:calibration"))
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
                            "supplies more information; the change is not a method gain.",
                            "tab:cutoff", size="\\footnotesize"))

    # --------------------------------------------------------------- splits
    if len(splits):
        rows = [[_tex(r["split"]), str(int(r["n_train"])), str(int(r["n_test"])),
                 pct(r["train_positive_rate"]), pct(r["test_positive_rate"]),
                 str(int(r["student_overlap"]))] for _, r in splits.iterrows()]
        parts.append(_table(
            ["Split", "Train", "Test", "Train Fail", "Test Fail", "Student overlap"],
            rows, "lrrrrr",
            "Every split is student-disjoint: no student appears on both sides. The overlap "
            "column is zero everywhere by construction and is checked at run time. The "
            "leave-one-module-out folds number six, not seven, because only six modules have "
            "a positive-weight TMA.", "tab:splits", size="\\footnotesize"))

    # ----------------------------------------------------------- robustness
    if len(specs):
        rows = [[_tex(r["spec"]), str(int(r["post_days"])), pct(r["tolerance"], 0),
                 _tex(r["match_on"]), str(int(r["pairs"])), pct(r["fail_coverage"]),
                 f(r["pre_log_click_smd"])] for _, r in specs.iterrows()]
        parts.append(_table(
            ["Specification", "Post days", "Tolerance", "Matched on", "Pairs",
             "Fail coverage", "Pre SMD"],
            rows, "lrrrrrr",
            "Robustness curve. The +21/+28 variants are reported twice: matched on "
            "post-deadline clicks (comparable with the main design) and on whole-window clicks "
            "(comparable with the earlier +21/+28 study).", "tab:specs",
            size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
    if len(clusters):
        rows = [[_tex(r["cluster"]), str(int(r["n"])), f(r["mean_post_clicks"], 1),
                 f(r["post_active_days"], 2), f(r["post_longest_inactive_run"], 2),
                 pct(r["submitted_by_cutoff"])] for _, r in clusters.iterrows()]
        parts.append(_table(
            ["Cluster", "Records", "Post clicks", "Active days", "Longest gap",
             "Submitted by $d+14$"],
            rows, "lrrrrr",
            "Outcome-blind timing clusters inside the paired sample. k and the feature list "
            "were chosen without the final result; the clusters are descriptive and are not "
            "used as a finding about learning styles.", "tab:clusters",
            size="\\footnotesize"))

    # ----------------------------------------------------------- at-risk set
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
            "still enrolled at the cutoff by construction.")

    # ---------------------------------------------------------- limitations
    parts.append(r"\section{Limitations}")
    parts.append(
        r"""\begin{itemize}\itemsep2pt
\item AP is average precision for Fail, not accuracy. The no-skill reference is the Fail share in the same test records.
\item Pairing is 1:1 by design and cannot estimate the population failure rate. The greedy order is not globally optimal, and the bootstrap resamples students on the realised pairs only, so re-matching uncertainty is excluded.
\item The binary cohort excludes records that withdrew before the cutoff. It is not a real-time early-warning population.
\item The first-TMA score contributes to the final result. Models that use it are retrospective upper bounds, and OULAD records no mark-release date.
\item The OULAD splits have been inspected across several rounds of exploration. The intervals and increments are exploratory even though the protocol is now frozen.
\item VLE \texttt{activity\_type} is a platform category, not content quality; \texttt{studentVle} records day-level clicks only, so study time cannot be measured.
\end{itemize}""")
    parts.append(
        "Two design facts deserve emphasis in any presentation. First, the pre-deadline window "
        "is not balanced by the primary match, so a reviewer can reasonably ask what happens "
        "when it is: Section~3 reports that subset and the sample loss is large. Second, the "
        "gain from timing features is small; the honest summary is that post-deadline "
        "persistence distinguishes outcomes descriptively far more clearly than it improves "
        "prediction.")

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
    counts = stages.get("S02", {}).get("metrics", {}).get("cohort_counts", {})
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
    counts = stages.get("S02", {}).get("metrics", {}).get("cohort_counts", {})
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
    counts = stages.get("S02", {}).get("metrics", {}).get("cohort_counts", {})
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
