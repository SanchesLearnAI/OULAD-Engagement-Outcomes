"""S10 — the at-risk set at the deadline, with Withdrawn kept as its own outcome.

This is the only stage that models Withdrawn.  The unit is every record that
was registered at the deadline and still enrolled at the observation cutoff, so
the label set is {Pass/Distinction, Fail, Withdrawn}.  It is a separate
analysis from the matched binary comparison: it describes who ends up where,
not what differs at comparable clicks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.data import KEY, read_master, read_vle
from src.tma import daily

FEATURES = ["pre_clicks", "pre_active_days", "last3_share", "post_clicks",
            "post_active_days", "post_longest_inactive_run", "post_week2_share",
            "post_both_weeks", "submitted_by_cutoff", "submitted_after_due",
            "days_early", "days_late"]
LABELS = ["Pass", "Fail", "Withdrawn"]


def _incidence(frame: pd.DataFrame, max_weeks: int = 12) -> pd.DataFrame:
    """Weekly outcome counts among the at-risk set, by week after the deadline."""
    rows = []
    for week in range(0, max_weeks + 1):
        rows.append({"weeks_after_deadline": week,
                     "records": int(len(frame)),
                     "still_enrolled": int((frame["date_unregistration"].isna()
                                            | frame["date_unregistration"].gt(
                                                frame["due_day"] + 7 * week)).sum())})
    return pd.DataFrame(rows)


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    post_days = int(cfg["study"]["post_days"])
    master = read_master(ctx)
    risk = master[master["eligible"] & master["due_day"].notna()].reset_index(drop=True)
    if risk.empty:
        st.check("risk_set_present", False, "empty at-risk cohort")
        return
    risk["y_t3"] = risk["final_result"].map({"Pass": "Pass", "Distinction": "Pass",
                                             "Fail": "Fail", "Withdrawn": "Withdrawn"})
    st.metric("risk_set", {str(k): int(v) for k, v in risk["y_t3"].value_counts().items()})
    st.check("withdrawn_included", int(risk["y_t3"].eq("Withdrawn").sum()) > 0,
             f"Withdrawn in the risk set={int(risk['y_t3'].eq('Withdrawn').sum())}")
    st.check("risk_set_not_binary",
             int(risk["y_t3"].eq("Withdrawn").sum()) > 0,
             "Withdrawn is modelled as its own class, never folded into Fail")

    lookup = risk[KEY + ["due_day"]]
    matrix = daily.daily_events(read_vle(ctx), lookup, pre_days=14,
                                post_days=max(post_days, 14))
    frame, _, _ = daily.build_daily_features(risk, matrix, None, pre_days=14,
                                             post_days=post_days)

    mix = (frame.groupby("y_t3").size().rename("records").reset_index())
    mix["share"] = mix["records"] / len(frame)
    p = ctx.paths.tables_dir / "t10_class_metrics.csv"
    mix.to_csv(p, index=False)
    st.artifact(p, rows=len(mix), cols=mix.shape[1], message="three-outcome mix")

    incidence = _incidence(frame)
    p = ctx.paths.tables_dir / "t10_cumulative_incidence.csv"
    incidence.to_csv(p, index=False)
    st.artifact(p, rows=len(incidence), cols=incidence.shape[1],
                message="weekly enrolment and outcome counts")

    with st.step("three_class_model"):
        cols = [c for c in FEATURES if c in frame]
        y = frame["y_t3"].to_numpy()
        tr, temp = next(GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
                        .split(frame, y, frame["id_student"].to_numpy()))
        holder = temp.reshape(-1, 1)
        va_rel, te_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
                              .split(holder, y[temp], frame["id_student"].iloc[temp].to_numpy()))
        va, te = temp[va_rel], temp[te_rel]
        st.check("three_class_student_disjoint",
                 set(frame["id_student"].iloc[tr]).isdisjoint(frame["id_student"].iloc[te]),
                 "no student spans train and test")
        transform = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols),
            ("context", OneHotEncoder(handle_unknown="ignore"), ["module_presentation"])])
        model = make_pipeline(transform, LogisticRegression(max_iter=2000,
                                                           class_weight="balanced",
                                                           random_state=seed))
        model.fit(frame.iloc[tr][cols + ["module_presentation"]], y[tr])
        prob = model.predict_proba(frame.iloc[te][cols + ["module_presentation"]])
        classes = list(model.classes_)
        pred = np.array(classes)[prob.argmax(axis=1)]
        cm = confusion_matrix(y[te], pred, labels=classes)
        table = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                             columns=[f"pred_{c}" for c in classes]).reset_index()
        p = ctx.paths.tables_dir / "t10_confusion.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1], message="three-class confusion")
        st.metric("three_class", {"n_test": int(len(te)),
                                  "classes": classes,
                                  "log_loss": float(log_loss(y[te], prob, labels=classes))})
        st.check("three_class_all_classes_present", len(classes) == 3,
                 f"classes={classes}")

    p = ctx.paths.tables_dir / "t10_outcome_by_presentation.csv"
    (frame.groupby(["code_module", "code_presentation", "y_t3"]).size()
     .rename("records").reset_index().to_csv(p, index=False))
    st.artifact(p, message="outcome mix by presentation")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(incidence["weeks_after_deadline"], incidence["still_enrolled"],
            color="#3b6ea5")
    ax.set_xlabel("weeks after the deadline")
    ax.set_ylabel("records still enrolled")
    ax.set_title("At-risk set shrinks as students unregister")
    fig.tight_layout()
    fp = ctx.paths.figures_dir / "f10_risk_set.png"
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    st.artifact(fp, message="risk-set figure")
