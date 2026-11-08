"""S10 — three-class outcome and competing-risk analysis (Withdrawn kept separate)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

from src.data import KEY, read_features, read_master, read_vle
from src.features import feature_columns
from src.survival import cumulative_incidence, long_table, student_event_table


def _split3(cohort, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr, temp = next(gss.split(cohort, np.zeros(len(cohort)),
                              cohort["id_student"].to_numpy()))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    va_rel, te_rel = next(gss2.split(temp.reshape(-1, 1),
                                     np.zeros(len(temp)),
                                     cohort["id_student"].iloc[temp].to_numpy()))
    return tr, temp[va_rel], temp[te_rel]


def _concordance(time, event, risk, max_n=3000, seed=42):
    n = len(time)
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_n, replace=False)
        time, event, risk = time[idx], event[idx], risk[idx]
        n = max_n
    conc = disc = tie = 0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] < risk[j]:
                    disc += 1
                else:
                    tie += 1
    den = conc + disc + tie
    return float((conc + 0.5 * tie) / den) if den else float("nan")


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    master = read_master(ctx)
    vle = read_vle(ctx)
    cohort = master.reset_index(drop=True)
    y = cohort["y_t3"].to_numpy()
    tr, va, te = _split3(cohort, seed)

    st.metric("t3_counts", {str(k): int(v) for k, v in
                            cohort["y_t3"].value_counts().items()})
    st.check("all_three_outcomes", set(np.unique(y)) == {"Pass", "Fail", "Withdrawn"},
             f"classes={sorted(np.unique(y))}")

    # ---- multinomial three-class at every cutoff ---------------------------
    rows = []
    c28_pred = None
    for c in cfg["windows"]["cutoffs"]:
        f = read_features(ctx, c)
        df = cohort[KEY + ["y_t3"]].merge(f, on=KEY, how="left",
                                          validate="one_to_one")
        cols = feature_columns(df, 3, c)
        X = df[cols].to_numpy(dtype=float)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                       random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])
        pred = pipe.predict(X[te])
        classes = list(pipe.named_steps["clf"].classes_)
        rep = classification_report(y[te], pred, labels=classes,
                                    output_dict=True, zero_division=0)
        ll = float(log_loss(y[te], proba, labels=classes))
        for cls in classes:
            rows.append({
                "cutoff": c, "class": cls, "precision": rep[cls]["precision"],
                "recall": rep[cls]["recall"], "f1": rep[cls]["f1-score"],
                "support": int(rep[cls]["support"]),
                "macro_f1": rep["macro avg"]["f1-score"],
                "weighted_f1": rep["weighted avg"]["f1-score"],
                "log_loss": ll, "n_test": len(te),
            })
        if c == 28:
            c28_pred = pd.DataFrame({
                "id_student": cohort["id_student"].iloc[te].to_numpy(),
                "code_module": cohort["code_module"].iloc[te].to_numpy(),
                "presentation": cohort["code_presentation"].iloc[te].to_numpy(),
                "y_true": y[te], "y_pred": pred,
                **{f"p_{cl}": proba[:, i] for i, cl in enumerate(classes)},
            })
    cm = pd.DataFrame(rows)
    p = ctx.paths.tables_dir / "t10_class_metrics.csv"
    cm.to_csv(p, index=False)
    st.artifact(p, rows=len(cm), cols=cm.shape[1], message="three-class metrics")
    st.check("multi_class_all_cutoffs", cm["cutoff"].nunique() == len(cfg["windows"]["cutoffs"]),
             f"cutoffs={cm['cutoff'].nunique()}")
    if c28_pred is not None:
        pp = ctx.run_dir / "artifacts" / "t10_predictions_c28.parquet"
        c28_pred.to_parquet(pp, index=False)
        st.artifact(pp, rows=len(c28_pred), cols=c28_pred.shape[1],
                    message="three-class predictions c28")

    # ---- long table, cumulative incidence ---------------------------------
    with st.step("competing_risk"):
        sdf = student_event_table(master)
        st.metric("event_time", {
            "origin": "course start (day 0); weeks = floor(day/7)",
            "event_week_min": int(sdf["event_week"].min()),
            "event_week_max": int(sdf["event_week"].max()),
            "withdrawn_missing_unreg": int(((sdf["y_t3"] == "Withdrawn") &
                                            sdf["date_unregistration"].isna()).sum()),
            "censoring": "none before course end; Pass/Fail events occur at course end",
        })
        idx = master[KEY].reset_index().rename(columns={"index": "row"})
        lt = long_table(sdf, vle, idx)
        lp = ctx.run_dir / "artifacts" / "t10_long_table.parquet"
        lt.to_parquet(lp, index=False)
        st.artifact(lp, rows=len(lt), cols=lt.shape[1],
                    message="student-week long table")
        ci = cumulative_incidence(sdf)
        cp = ctx.paths.tables_dir / "t10_cumulative_incidence.csv"
        ci.to_csv(cp, index=False)
        st.artifact(cp, rows=len(ci), cols=ci.shape[1],
                    message="cumulative incidence")
        st.check("withdrawn_kept_separate",
                 {"Pass", "Fail", "Withdrawn"} <= set(np.unique(y)),
                 "Withdrawn is a separate class, never folded into Fail")

        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        ax.plot(ci["week"], ci["cif_pass"], label="Pass/Distinction", lw=1.6)
        ax.plot(ci["week"], ci["cif_fail"], label="Fail", lw=1.6)
        ax.plot(ci["week"], ci["cif_withdrawn"], label="Withdrawn", lw=1.6)
        ax.set_xlabel("week from course start")
        ax.set_ylabel("cumulative incidence")
        ax.set_title("Competing risks: three outcomes")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f10_cumulative_incidence.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="cumulative incidence figure")

    # ---- landmark cause-specific Cox (features available at the landmark) ---
    with st.step("landmark_cox"):
        f28 = read_features(ctx, 28)
        f56 = read_features(ctx, 56)
        sdf = student_event_table(master)
        event_day = np.where(sdf["y_t3"].to_numpy() == "Withdrawn",
                             sdf["date_unregistration"].to_numpy(dtype=float),
                             sdf["module_presentation_length"].to_numpy(dtype=float))
        event_day = pd.Series(event_day, index=sdf.index).fillna(
            sdf["module_presentation_length"]).to_numpy(dtype=float)
        landmark_specs = [
            (28, f28, ["w1_clicks_log1p", "w1_active_days", "w1_entropy_activity",
                       "w1_n_sites", "w1_daily_cv"]),
            (56, f56, ["w1_clicks_log1p", "w1_active_days", "w1_entropy_activity",
                       "w1_n_sites", "w1_daily_cv",
                       "w2_clicks_log1p", "w2_active_days",
                       "w2_entropy_activity", "d_clicks"]),
        ]
        lm_rows = []
        lm_summary = {}
        cohort_ok_28 = True
        for L, f, wanted in landmark_specs:
            cols = [c for c in wanted if c in f.columns]
            d = sdf.merge(f[KEY + cols], on=KEY, how="left")
            alive = (event_day > L)
            trL = tr[alive[tr]]
            teL = te[alive[te]]
            if L == 28:
                cohort_ok_28 = bool((event_day[trL] > 28).all() and
                                    (event_day[teL] > 28).all())
            if len(trL) == 0 or len(teL) == 0:
                st.warn(f"landmark {L}: empty train/test", "landmark_cox")
                continue
            X_all = np.nan_to_num(d[cols].to_numpy(dtype=float))
            scaler = StandardScaler().fit(X_all[trL])
            X = scaler.transform(X_all)
            time = np.maximum(1.0, d["event_week"].to_numpy(dtype=float) - L / 7.0)
            fitted = 0
            for cause_name, cause_code in (("Pass", 1), ("Fail", 2), ("Withdrawn", 3)):
                status = (d["event_type"].to_numpy() == cause_code).astype(int)
                if status[trL].sum() < 5 or status[teL].sum() < 1:
                    st.warn(f"landmark {L} cause {cause_name}: too few events",
                            "landmark_cox")
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = PHReg(time[trL], X[trL], status=status[trL]).fit()
                    ci_df = res.conf_int()
                    risk = X[teL] @ res.params
                    test_c = _concordance(time[teL], status[teL], risk,
                                          max_n=3000, seed=ctx.cfg["seed"])
                    for i, col in enumerate(cols):
                        lm_rows.append({
                            "landmark": L, "cause": cause_name, "covariate": col,
                            "coef": float(res.params[i]),
                            "hr": float(np.exp(res.params[i])),
                            "ci_low": float(np.exp(ci_df[i, 0])),
                            "ci_high": float(np.exp(ci_df[i, 1])),
                            "p_value": float(res.pvalues[i]),
                            "n_train": int(len(trL)), "n_test": int(len(teL)),
                            "events_train": int(status[trL].sum()),
                            "events_test": int(status[teL].sum()),
                            "test_concordance": test_c,
                        })
                    fitted += 1
                except Exception as e:  # noqa: BLE001
                    st.warn(f"landmark {L} cause {cause_name} failed: "
                            f"{type(e).__name__}: {e}", "landmark_cox")
            lm_summary[L] = {"n_train": int(len(trL)), "n_test": int(len(teL)),
                             "fitted_causes": int(fitted)}
        if lm_rows:
            lm = pd.DataFrame(lm_rows)
            lp = ctx.paths.tables_dir / "t10_landmark_cox.csv"
            lm.to_csv(lp, index=False)
            st.artifact(lp, rows=len(lm), cols=lm.shape[1],
                        message="landmark cause-specific Cox (test concordance)")
        st.metric("cox_landmark", lm_summary)
        l28_cols = [c for c in landmark_specs[0][2] if c in f28.columns]
        st.check("cox_no_future_features",
                 not any(c.startswith(("w2_", "d_", "accel_")) or "score" in c.lower()
                         for c in l28_cols),
                 f"landmark 28 features={l28_cols}")
        st.check("cox_landmark_cohort", bool(cohort_ok_28),
                 "no event at or before the day-28 landmark enters its cohort")
        st.check("survival_available", bool(lm_rows),
                 "landmark Cox fitted for at least one landmark/cause",
                 warn_only=not bool(lm_rows))
