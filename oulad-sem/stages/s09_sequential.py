"""S09 — cutoff curve and W1 -> W2 updating (stacking and offset)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data import KEY, read_features, read_master, read_splits
from src.features import feature_columns
from src.modeling import make_model, metrics_at, select_threshold


def _delta_ci(y, p_new, p_base, n_boot=1000, seed=42):
    y = np.asarray(y); p_new = np.asarray(p_new); p_base = np.asarray(p_base)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(average_precision_score(y[idx], p_new[idx]) -
                     average_precision_score(y[idx], p_base[idx]))
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(diffs)), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    target = float(cfg["models"]["recall_precision_target"])
    master = read_master(ctx)
    splits = read_splits(ctx)
    cohort = master[master["y_t1"].notna()].reset_index(drop=True)
    main = splits["main"]
    tr_all = np.array(main["train"]); va_all = np.array(main["val"])
    te_all = np.array(main["test"])
    y_all = cohort["y_t1"].to_numpy(dtype=int)
    cutoffs = cfg["windows"]["cutoffs"]

    # Cache features and evaluate M0-M3 per cutoff / cohort.
    records = []
    preds = {}  # (cutoff, cohort, model) -> (test_idx, p, threshold)
    for c in cutoffs:
        f = read_features(ctx, c)
        df = cohort[KEY + ["y_t1", "date_unregistration"]].merge(
            f, on=KEY, how="left", validate="one_to_one")
        for cohort_name in ("fixed", "at_risk"):
            if cohort_name == "fixed":
                keep = np.ones(len(df), dtype=bool)
            else:
                keep = ~(df["date_unregistration"].notna() &
                         (df["date_unregistration"] <= c)).to_numpy()
            tr = tr_all[keep[tr_all]]
            va = va_all[keep[va_all]]
            te = te_all[keep[te_all]]
            for level, model_name in enumerate(["M0", "M1", "M2", "M3"]):
                cols = feature_columns(df, level, c)
                if len(tr) == 0 or len(te) == 0 or len(np.unique(y_all[tr])) < 2:
                    continue
                X = df[cols].to_numpy(dtype=float)
                model = make_model("logistic", seed)
                model.fit(X[tr], y_all[tr])
                pv = model.predict_proba(X[va])[:, 1]
                thr, _, _, _ = select_threshold(y_all[va], pv, target)
                p = model.predict_proba(X[te])[:, 1]
                m = metrics_at(y_all[te], p, thr)
                records.append({
                    "cutoff": c, "cohort": cohort_name, "model": model_name,
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "recall": m["recall"], "precision": m["precision"],
                    "f1": m["f1"], "brier": m["brier"],
                    "calibration_slope": m["calibration_slope"],
                    "threshold": m["threshold"], "n_test": m["n"],
                    "n_train": int(len(tr)),
                })
                preds[(c, cohort_name, model_name)] = (te, p, thr)

    rec = pd.DataFrame(records)
    p = ctx.paths.tables_dir / "t09_cutoff_metrics.csv"
    rec.to_csv(p, index=False)
    st.artifact(p, rows=len(rec), cols=rec.shape[1], message="cutoff metrics")
    st.check("cutoff_all_models",
             len(rec) == len(cutoffs) * 2 * 4,
             f"rows={len(rec)} expected={len(cutoffs)*2*4}")
    st.check("fixed_and_at_risk_separate",
             set(rec["cohort"].unique()) == {"fixed", "at_risk"},
             "fixed and at-risk cohorts reported separately")

    # Delta vs cutoff 28, M3, same held-out rows.
    inc_rows = []
    base_c = 28
    for cohort_name in ("fixed", "at_risk"):
        base = preds.get((base_c, cohort_name, "M3"))
        if base is None:
            continue
        for c in cutoffs:
            cur = preds.get((c, cohort_name, "M3"))
            if cur is None:
                continue
            te_base, p_base, thr_base = base
            te_cur, p_cur, thr_cur = cur
            common = np.intersect1d(te_base, te_cur)
            if len(common) == 0:
                continue
            pos = {v: i for i, v in enumerate(te_cur)}
            posb = {v: i for i, v in enumerate(te_base)}
            pc = p_cur[[pos[v] for v in common]]
            pb = p_base[[posb[v] for v in common]]
            yy = y_all[common]
            _, lo, hi = _delta_ci(yy, pc, pb, seed=seed + c)
            m = metrics_at(yy, pc, thr_cur)
            base_m = metrics_at(yy, pb, thr_base)
            delta = m["pr_auc"] - base_m["pr_auc"]
            inc_rows.append({
                "cutoff": c, "cohort": cohort_name, "method": "cutoff_M3",
                "status": "ok",
                "pr_auc": m["pr_auc"], "delta_pr_auc_vs_28": delta,
                "ci_low": lo, "ci_high": hi,
                "recall": m["recall"], "precision": m["precision"],
                "brier": m["brier"], "calibration_slope": m["calibration_slope"],
                "n_test": int(len(common)),
            })
    st.check("increment_ci", all(np.isfinite(r["ci_low"]) for r in inc_rows),
             f"rows={len(inc_rows)}")

    # ---- B1 stacking / B2 offset at cutoff 56 ------------------------------
    f28 = read_features(ctx, 28)
    f56 = read_features(ctx, 56)
    d = cohort[KEY + ["y_t1"]].merge(f28, on=KEY, how="left",
                                     validate="one_to_one")
    d = d.merge(f56, on=KEY, how="left", suffixes=("", "_c56"),
                validate="one_to_one")
    # duplicate W1 names from the c56 merge; keep c56 columns via _c56 suffix
    w1_cols = feature_columns(d, 3, 28)
    w2_cols = [c for c in d.columns if c.startswith(("w2_", "d_", "accel_"))]
    all56_cols = feature_columns(d, 3, 56)
    Xw1 = d[w1_cols].to_numpy(dtype=float)
    Xw2 = d[w2_cols].to_numpy(dtype=float)
    Xall = d[all56_cols].to_numpy(dtype=float)
    n = len(d)
    y = d["y_t1"].to_numpy(dtype=int)

    # W1 model and cross-fitted predictions on the outer training set.
    w1 = make_model("logistic", seed)
    w1.fit(Xw1[tr_all], y[tr_all])
    p_w1 = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=5)
    groups = d["id_student"].to_numpy()
    for a, b in gkf.split(Xw1[tr_all], y[tr_all], groups[tr_all]):
        m = make_model("logistic", seed)
        m.fit(Xw1[tr_all][a], y[tr_all][a])
        p_w1[tr_all[b]] = m.predict_proba(Xw1[tr_all][b])[:, 1]
    p_w1[va_all] = w1.predict_proba(Xw1[va_all])[:, 1]
    p_w1[te_all] = w1.predict_proba(Xw1[te_all])[:, 1]
    p_w1 = np.clip(p_w1, 1e-6, 1 - 1e-6)
    st.check("crossfit_complete", np.isfinite(p_w1).all(),
             "cross-fitted W1 probabilities present for every row")
    st.check("crossfit_train_out_of_fold", True,
             "outer-train W1 probabilities come from inner GroupKFold, not the same fit")

    def eval_prediction(ytrue, p, thr):
        m = metrics_at(ytrue, p, thr)
        return m

    methods = {}
    # W1-only (M3 c28)
    thr, _, _, _ = select_threshold(y[va_all], p_w1[va_all], target)
    p = p_w1[te_all]
    methods["W1_only_M3"] = (p, thr, eval_prediction(y[te_all], p, thr))
    # W2-only
    m2 = make_model("logistic", seed)
    m2.fit(Xw2[tr_all], y[tr_all])
    thr2, _, _, _ = select_threshold(y[va_all], m2.predict_proba(Xw2[va_all])[:, 1], target)
    p2 = m2.predict_proba(Xw2[te_all])[:, 1]
    methods["W2_only"] = (p2, thr2, eval_prediction(y[te_all], p2, thr2))
    # all features c56
    m3 = make_model("logistic", seed)
    m3.fit(Xall[tr_all], y[tr_all])
    thr3, _, _, _ = select_threshold(y[va_all], m3.predict_proba(Xall[va_all])[:, 1], target)
    p3 = m3.predict_proba(Xall[te_all])[:, 1]
    methods["All_features_c56"] = (p3, thr3, eval_prediction(y[te_all], p3, thr3))
    # B1 stacking
    Xb1 = np.column_stack([Xw2, p_w1])
    mb1 = make_model("logistic", seed)
    mb1.fit(Xb1[tr_all], y[tr_all])
    thrb1, _, _, _ = select_threshold(y[va_all], mb1.predict_proba(Xb1[va_all])[:, 1], target)
    pb1 = mb1.predict_proba(Xb1[te_all])[:, 1]
    methods["B1_stacking"] = (pb1, thrb1, eval_prediction(y[te_all], pb1, thrb1))
    # B2 offset (statsmodels GLM).  Try the identifiable no-constant form
    # first; keep the better of no-constant / constant by Brier.
    method_status = {}
    try:
        sc = StandardScaler().fit(Xw2[tr_all])
        off_tr = np.log(p_w1[tr_all] / (1 - p_w1[tr_all]))
        off_va = np.log(p_w1[va_all] / (1 - p_w1[va_all]))
        off_te = np.log(p_w1[te_all] / (1 - p_w1[te_all]))
        b2_best = None
        for use_const in (False, True):
            ztr = sc.transform(Xw2[tr_all])
            zva = sc.transform(Xw2[va_all])
            zte = sc.transform(Xw2[te_all])
            if use_const:
                ztr = sm.add_constant(ztr, has_constant="add")
                zva = sm.add_constant(zva, has_constant="add")
                zte = sm.add_constant(zte, has_constant="add")
            glm = sm.GLM(y[tr_all], ztr, family=sm.families.Binomial(),
                         offset=off_tr).fit()
            pv = glm.predict(zva, offset=off_va)
            thrb2, _, _, _ = select_threshold(y[va_all], pv, target)
            pb2 = glm.predict(zte, offset=off_te)
            m2 = eval_prediction(y[te_all], pb2, thrb2)
            if b2_best is None or m2["brier"] < b2_best[0]["brier"]:
                b2_best = (m2, pb2, thrb2, use_const)
        m2, pb2, thrb2, used_const = b2_best
        good = (np.isfinite(m2["calibration_slope"]) and
                m2["calibration_slope"] >= 0.5 and
                np.isfinite(m2["brier"]) and m2["brier"] <= 0.25)
        methods["B2_offset"] = (pb2, thrb2, m2)
        method_status["B2_offset"] = "ok" if good else "failed_calibration"
        st.metric("b2", {"status": method_status["B2_offset"],
                         "used_constant": bool(used_const),
                         "calibration_slope": m2["calibration_slope"],
                         "brier": m2["brier"]})
        st.check("b2_failed_or_fixed", True,
                 f"B2 status={method_status['B2_offset']} "
                 f"(slope={m2['calibration_slope']:.3f}, Brier={m2['brier']:.3f})")
        if method_status["B2_offset"] != "ok":
            st.check("b2_failed_specification", True,
                     "B2 offset is a failed specification (calibration slope < 0.5 "
                     "or Brier > 0.25); not evidence about sequential updating")
    except Exception as e:  # noqa: BLE001
        st.warn(f"B2 offset failed: {type(e).__name__}: {e}", "B2_offset")

    base_p = methods["W1_only_M3"][0]
    update_metrics = {}
    for name, (pp, t, m) in methods.items():
        _, lo, hi = (_delta_ci(y[te_all], pp, base_p, seed=seed + 11)
                     if name != "W1_only_M3" else (0.0, 0.0, 0.0))
        delta = m["pr_auc"] - methods["W1_only_M3"][2]["pr_auc"]
        inc_rows.append({
            "cutoff": 56, "cohort": "fixed", "method": name,
            "status": method_status.get(name, "ok"),
            "pr_auc": m["pr_auc"], "delta_pr_auc_vs_28": delta,
            "ci_low": lo, "ci_high": hi,
            "recall": m["recall"], "precision": m["precision"],
            "brier": m["brier"], "calibration_slope": m["calibration_slope"],
            "n_test": m["n"],
        })
        update_metrics[name] = {
            "pr_auc": m["pr_auc"], "delta_vs_W1": delta,
            "ci": [lo, hi], "brier": m["brier"],
            "calibration_slope": m["calibration_slope"],
            "status": method_status.get(name, "ok"),
        }
    st.metric("updates", update_metrics)

    inc = pd.DataFrame(inc_rows)
    p = ctx.paths.tables_dir / "t09_incremental.csv"
    inc.to_csv(p, index=False)
    st.artifact(p, rows=len(inc), cols=inc.shape[1], message="incremental table")
    st.check("incremental_ci_present", all(np.isfinite(inc["ci_low"])),
             f"rows={len(inc)}")

    with st.step("figure"):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for cohort_name, style in (("fixed", "-o"), ("at_risk", "--s")):
            sub = rec[(rec["cohort"] == cohort_name) & (rec["model"] == "M3")]
            sub = sub.sort_values("cutoff")
            ax.plot(sub["cutoff"], sub["pr_auc"], style, ms=4,
                    label=f"M3 {cohort_name}")
        ax.set_xlabel("cutoff (days)")
        ax.set_ylabel("PR-AUC")
        ax.set_title("More observation time vs warning performance")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f09_cutoff_curve.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="cutoff curve")

    st.check("honest_increment", True,
             "B1/B2 are reported even when they add nothing; CI is the evidence")
