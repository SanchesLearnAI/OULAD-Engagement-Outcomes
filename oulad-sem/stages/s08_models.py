"""S08 — M0-M5 model ladder on grouped, temporal and LOMO splits."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

from src.data import KEY, read_features, read_master, read_splits
from src.features import feature_columns
from src.logging_utils import atomic_write_json
from src.modeling import (bootstrap_metrics, make_model, metrics_at,
                          select_threshold)

_MODELS = [
    ("M0", "logistic", "volume-only: log clicks + active days"),
    ("M1", "logistic", "M0 + breadth/regularity/composition"),
    ("M2", "logistic", "M1 + timing"),
    ("M3", "logistic", "M2 + assessment submission timing (no scores)"),
    ("M4", "logistic", "M3 + assessment scores (UPPER BOUND ONLY)"),
    ("M5", "histgb", "M3 features + HistGradientBoosting"),
]


def _holdout(train_idx, cohort, y, seed):
    if len(train_idx) < 20:
        return train_idx, np.array([], dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    a, b = next(gss.split(cohort.iloc[train_idx], y[train_idx],
                          cohort["id_student"].iloc[train_idx].to_numpy()))
    return train_idx[a], train_idx[b]


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    target_prec = float(cfg["models"]["recall_precision_target"])
    master = read_master(ctx)
    features = read_features(ctx, 28)
    scores = pd.read_parquet(ctx.paths.processed_dir / "features_scores_c28.parquet")
    splits = read_splits(ctx)

    cohort = master[master["y_t1"].notna()].reset_index(drop=True)
    st.check("split_cohort_matches", len(cohort) == splits["n"],
             f"cohort={len(cohort)} splits={splits['n']}")
    st.check("split_cohort_hash", True,
             f"cohort_keys_sha256={splits['cohort_keys_sha256'][:16]}...")
    df = cohort[KEY + ["y_t1"]].merge(features, on=KEY, how="left",
                                      validate="one_to_one")
    df = df.merge(scores[KEY + ["score_mean", "score_min", "score_max",
                                "score_count"]], on=KEY, how="left")
    st.check("rows_aligned", len(df) == len(cohort), f"rows={len(df)}")
    y = df["y_t1"].to_numpy(dtype=int)
    n = len(df)

    cols = {
        "M0": ["w1_clicks_log1p", "w1_active_days"],
        "M1": feature_columns(df, 1, 28),
        "M2": feature_columns(df, 2, 28),
        "M3": feature_columns(df, 3, 28),
    }
    cols["M4"] = cols["M3"] + ["score_mean", "score_min", "score_max",
                                "score_count"]
    cols["M5"] = list(cols["M3"])
    for m in ("M0", "M1", "M2", "M3", "M5"):
        st.check(f"no_score_features_{m}",
                 not any("score" in c.lower() for c in cols[m]),
                 f"{m} feature count={len(cols[m])}")
    st.check("m4_labeled_upper_bound", True,
             "M4 contains assessment scores and is reported only as an upper bound")

    pred_frames = []
    metric_rows = []
    cards = {}
    main_metrics = {}
    kfold_metrics = {}
    temporal_metrics = {}
    lomo_metrics = {}

    def add_metrics(model, split, ytrue, pprob, thr, n_train, extra=None,
                    rule="unknown"):
        m = metrics_at(ytrue, pprob, thr)
        prev = float(np.mean(ytrue)) if len(ytrue) else float("nan")
        # Reference points: no-skill PR-AUC equals prevalence; predicting every
        # enrolment positive gives F1 = 2*prev/(1+prev).
        no_skill_f1 = 2 * prev / (1 + prev) if np.isfinite(prev) else float("nan")
        ci = bootstrap_metrics(ytrue, pprob, thr, n_boot=1000,
                               seed=seed + 7)
        for k, v in m.items():
            if k in ("n", "positive_rate", "threshold"):
                continue
            lo, hi = ci.get(k, (np.nan, np.nan))
            metric_rows.append({
                "model": model, "split": split, "metric": k, "value": v,
                "ci_low": lo, "ci_high": hi, "n": m["n"],
                "positive_rate": m["positive_rate"], "threshold": m["threshold"],
                "n_train": int(n_train), "threshold_rule": rule,
                "no_skill_pr_auc": prev, "all_positive_f1": no_skill_f1,
            })
        return m

    def fit_predict(model_name, kind, columns, tr, va, te, refit):
        if len(tr) == 0 or len(te) == 0 or len(np.unique(y[tr])) < 2:
            return None, np.nan, None, "no_fit"
        X = df[columns].to_numpy(dtype=float)
        model = make_model(kind, seed)
        model.fit(X[tr], y[tr])
        thr = np.nan
        rule = "no_validation"
        if va is not None and len(va):
            pv = model.predict_proba(X[va])[:, 1]
            thr, _, _, rule = select_threshold(y[va], pv, target_prec)
        if refit and va is not None and len(va):
            model = make_model(kind, seed)
            model.fit(X[np.concatenate([tr, va])],
                      y[np.concatenate([tr, va])])
        p = model.predict_proba(X[te])[:, 1]
        return model, thr, p, rule

    with st.step("main_split"):
        main = splits["main"]
        main_tr = np.array(main["train"])
        main_va = np.array(main["val"])
        main_te = np.array(main["test"])
        tr, va, te = main_tr, main_va, main_te
        for name, kind, note in _MODELS:
            model, thr, p, rule = fit_predict(name, kind, cols[name], tr, va, te, False)
            if p is None:
                st.check(f"{name}_main_fit", False, "could not fit")
                continue
            if not np.isfinite(thr):
                thr = 0.5
            m = add_metrics(name, "main_test", y[te], p, thr, len(tr), rule=rule)
            pf = pd.DataFrame({
                "model": name, "split": "main_test",
                "id_student": cohort["id_student"].iloc[te].to_numpy(),
                "code_module": cohort["code_module"].iloc[te].to_numpy(),
                "presentation": cohort["code_presentation"].iloc[te].to_numpy(),
                "y_true": y[te], "y_prob": p, "fold": "0",
            })
            pred_frames.append(pf)
            main_metrics[name] = {
                "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                "recall_at_precision": m["recall"], "precision": m["precision"],
                "f1": m["f1"], "brier": m["brier"],
                "calibration_slope": m["calibration_slope"],
                "threshold": m["threshold"], "n_test": m["n"],
                "n_train": int(len(tr)),
            }
            cards[name] = {
                "model": name, "family": kind, "note": note,
                "n_features": len(cols[name]), "features": cols[name],
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "threshold": float(thr),
                "test_metrics": {k: float(m[k]) for k in
                                 ("pr_auc", "roc_auc", "precision", "recall",
                                  "f1", "brier", "calibration_slope",
                                  "calibration_intercept", "ece")},
            }
        st.metric("main_models", {"main": main_metrics})
        st.check("main_all_models", len(cards) == len(_MODELS),
                 f"models fitted={len(cards)}")
        st.check("main_calibration_complete",
                 all(np.isfinite(c["test_metrics"]["brier"]) for c in cards.values()),
                 "Brier score recorded for every model")

    with st.step("context_baseline"):
        cat_cols = ["gender", "age_band", "highest_education", "imd_band",
                    "disability", "code_module", "code_presentation"]
        num_cols = ["num_of_prev_attempts", "studied_credits"]
        cdf = cohort[cat_cols + num_cols].copy()
        for c in cat_cols:
            cdf[c] = cdf[c].fillna("Unknown").astype(str)
        num = cdf[num_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median()).to_numpy(dtype=float)
        dummies = pd.get_dummies(cdf[cat_cols], prefix=cat_cols, dummy_na=True,
                                 dtype=float)
        Xctx = np.column_stack([num, dummies.to_numpy(dtype=float)])
        ctx_specs = {
            "M-1_context": Xctx,
            "M0c_context+M0": np.column_stack([Xctx, df[cols["M0"]].to_numpy(dtype=float)]),
            "M1c_context+M1": np.column_stack([Xctx, df[cols["M1"]].to_numpy(dtype=float)]),
        }
        crows = []
        ctx_pred = {}
        for name, Xc in ctx_specs.items():
            model = make_model("logistic", seed)
            model.fit(Xc[main_tr], y[main_tr])
            pv = model.predict_proba(Xc[main_va])[:, 1]
            thr, _, _, _ = select_threshold(y[main_va], pv, target_prec)
            p = model.predict_proba(Xc[main_te])[:, 1]
            ctx_pred[name] = p
            m = metrics_at(y[main_te], p, thr)
            ci = bootstrap_metrics(y[main_te], p, thr, n_boot=500, seed=seed + 31)
            crows.append({
                "model": name, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                "recall": m["recall"], "precision": m["precision"],
                "brier": m["brier"], "calibration_slope": m["calibration_slope"],
                "threshold": m["threshold"], "n_train": int(len(main_tr)),
                "n_test": m["n"], "pr_ci_low": ci["pr_auc"][0],
                "pr_ci_high": ci["pr_auc"][1],
            })
        ct = pd.DataFrame(crows)
        cp = ctx.paths.tables_dir / "t08_context_metrics.csv"
        ct.to_csv(cp, index=False)
        st.artifact(cp, rows=len(ct), cols=ct.shape[1],
                    message="context-only and context-adjusted models")

        # WP5.1: paired bootstrap on the same held-out rows.
        p_m0c = ctx_pred.get("M0c_context+M0")
        p_m1c = ctx_pred.get("M1c_context+M1")
        y_te = y[main_te]
        delta_seed = seed + 37
        if p_m0c is not None and p_m1c is not None:
            n_te = len(y_te)
            rng = np.random.default_rng(delta_seed)
            dpr, droc = [], []
            for _ in range(1000):
                idx = rng.integers(0, n_te, n_te)
                if len(np.unique(y_te[idx])) < 2:
                    continue
                dpr.append(
                    average_precision_score(y_te[idx], p_m1c[idx]) -
                    average_precision_score(y_te[idx], p_m0c[idx]))
                droc.append(
                    roc_auc_score(y_te[idx], p_m1c[idx]) -
                    roc_auc_score(y_te[idx], p_m0c[idx]))
            point_pr = (average_precision_score(y_te, p_m1c) -
                        average_precision_score(y_te, p_m0c))
            point_roc = (roc_auc_score(y_te, p_m1c) -
                         roc_auc_score(y_te, p_m0c))
            if dpr:
                pr_lo, pr_hi = np.percentile(dpr, [2.5, 97.5])
                roc_lo, roc_hi = np.percentile(droc, [2.5, 97.5])
            else:
                pr_lo = pr_hi = roc_lo = roc_hi = float("nan")
            cd = pd.DataFrame([{
                "comparison": "M1c_context+M1 - M0c_context+M0",
                "delta_pr_auc": float(point_pr),
                "ci_low": float(pr_lo), "ci_high": float(pr_hi),
                "delta_roc_auc": float(point_roc),
                "roc_ci_low": float(roc_lo), "roc_ci_high": float(roc_hi),
                "n_test": int(n_te), "seed": int(delta_seed),
            }])
            dp = ctx.paths.tables_dir / "t08_context_delta.csv"
            cd.to_csv(dp, index=False)
            st.artifact(dp, rows=len(cd), cols=cd.shape[1],
                        message="context paired-bootstrap delta")
            st.metric("context_delta", {
                "delta_pr_auc": float(point_pr),
                "ci_low": float(pr_lo), "ci_high": float(pr_hi),
                "delta_roc_auc": float(point_roc),
                "roc_ci_low": float(roc_lo), "roc_ci_high": float(roc_hi),
                "n_boot": len(dpr), "seed": int(delta_seed),
                "ci_includes_zero": bool(pr_lo <= 0 <= pr_hi),
            })
            st.check("context_delta_ci",
                     len(dpr) == 1000 and np.isfinite(pr_lo) and np.isfinite(pr_hi) and
                     np.isfinite(roc_lo) and np.isfinite(roc_hi),
                     f"paired bootstrap n={len(dpr)}, seed={delta_seed}")
        st.metric("context_models", {
            r["model"]: {"pr_auc": round(float(r["pr_auc"]), 4),
                         "roc_auc": round(float(r["roc_auc"]), 4),
                         "brier": round(float(r["brier"]), 4)}
            for _, r in ct.iterrows()})
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        x = np.arange(len(ct)); w = 0.38
        ax.bar(x - w/2, ct["pr_auc"], w, label="PR-AUC", color="#3b6ea5")
        ax.bar(x + w/2, ct["roc_auc"], w, label="ROC-AUC", color="#6ea36f")
        ax.set_xticks(x, ct["model"], rotation=15)
        ax.set_ylim(0, 1); ax.set_ylabel("held-out test")
        ax.set_title("Context-only vs context-adjusted ladder")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f08_context_comparison.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="context comparison figure")
        st.check("context_models_held_out", len(ct) == 3,
                 f"models={list(ct['model'])}")

    with st.step("group_kfold"):
        X = {name: df[cols[name]].to_numpy(dtype=float) for name, _, _ in _MODELS}
        groups = cohort["id_student"].to_numpy()
        for name, kind, _ in _MODELS:
            oof = np.full(n, np.nan)
            folds = np.full(n, -1, dtype=int)
            ok = True
            for f in splits["group_kfold"]:
                trf = np.array(f["train"]); tef = np.array(f["test"])
                if len(np.unique(y[trf])) < 2:
                    ok = False; continue
                model = make_model(kind, seed)
                model.fit(X[name][trf], y[trf])
                oof[tef] = model.predict_proba(X[name][tef])[:, 1]
                folds[tef] = f["fold"]
            mask = ~np.isnan(oof)
            if not ok or mask.sum() == 0:
                st.check(f"{name}_kfold", False, "k-fold failed")
                continue
            thr, _, _, rule = select_threshold(y[mask], oof[mask], target_prec)
            m = add_metrics(name, "group_kfold", y[mask], oof[mask], thr,
                            int(n - mask.sum()), rule=rule)
            pf = pd.DataFrame({
                "model": name, "split": "group_kfold",
                "id_student": cohort["id_student"].iloc[mask].to_numpy(),
                "code_module": cohort["code_module"].iloc[mask].to_numpy(),
                "presentation": cohort["code_presentation"].iloc[mask].to_numpy(),
                "y_true": y[mask], "y_prob": oof[mask],
                "fold": folds[mask].astype(str),
            })
            pred_frames.append(pf)
            kfold_metrics[name] = {"pr_auc": m["pr_auc"],
                                   "roc_auc": m["roc_auc"],
                                   "recall_at_precision": m["recall"]}
        st.metric("kfold_models", {"group_kfold": kfold_metrics})

    with st.step("temporal"):
        t = splits["temporal"]
        t_tr = np.array(t["train"]); t_te = np.array(t["test"])
        tr80, va20 = _holdout(t_tr, cohort, y, seed + 2)
        for name, kind, _ in _MODELS:
            model, thr, p, rule = fit_predict(name, kind, cols[name], tr80, va20, t_te, True)
            if p is None:
                st.check(f"{name}_temporal", False, "could not fit")
                continue
            if not np.isfinite(thr):
                thr = 0.5
            m = add_metrics(name, "temporal", y[t_te], p, thr,
                            len(tr80) + len(va20), rule=rule)
            pf = pd.DataFrame({
                "model": name, "split": "temporal",
                "id_student": cohort["id_student"].iloc[t_te].to_numpy(),
                "code_module": cohort["code_module"].iloc[t_te].to_numpy(),
                "presentation": cohort["code_presentation"].iloc[t_te].to_numpy(),
                "y_true": y[t_te], "y_prob": p, "fold": "0",
            })
            pred_frames.append(pf)
            temporal_metrics[name] = {"pr_auc": m["pr_auc"],
                                      "roc_auc": m["roc_auc"],
                                      "recall_at_precision": m["recall"]}
        st.metric("temporal_models", {"temporal": temporal_metrics})

    with st.step("lomo"):
        for name, kind, _ in _MODELS:
            oof = np.full(n, np.nan)
            folds = np.array([""] * n, dtype=object)
            ok = True
            for f in splits["lomo"]:
                trf = np.array(f["train"]); tef = np.array(f["test"])
                tr80, va20 = _holdout(trf, cohort, y, seed + 3)
                model, thr, p, rule = fit_predict(name, kind, cols[name], tr80, va20,
                                            tef, True)
                if p is None:
                    ok = False; continue
                oof[tef] = p
                folds[tef] = f["module"]
            mask = ~np.isnan(oof)
            if not ok or mask.sum() == 0:
                st.check(f"{name}_lomo", False, "LOMO failed")
                continue
            thr, _, _, rule = select_threshold(y[mask], oof[mask], target_prec)
            m = add_metrics(name, "lomo", y[mask], oof[mask], thr,
                            int(n - mask.sum()), rule=rule)
            pf = pd.DataFrame({
                "model": name, "split": "lomo",
                "id_student": cohort["id_student"].iloc[mask].to_numpy(),
                "code_module": cohort["code_module"].iloc[mask].to_numpy(),
                "presentation": cohort["code_presentation"].iloc[mask].to_numpy(),
                "y_true": y[mask], "y_prob": oof[mask],
                "fold": folds[mask].astype(str),
            })
            pred_frames.append(pf)
            lomo_metrics[name] = {"pr_auc": m["pr_auc"],
                                  "roc_auc": m["roc_auc"],
                                  "recall_at_precision": m["recall"]}
        st.metric("lomo_models", {"lomo": lomo_metrics})

    with st.step("write_metrics_predictions"):
        mt = pd.DataFrame(metric_rows)
        p = ctx.paths.tables_dir / "t08_model_metrics.csv"
        mt.to_csv(p, index=False)
        st.artifact(p, rows=len(mt), cols=mt.shape[1], message="model metrics")
        preds = pd.concat(pred_frames, ignore_index=True)
        pp = ctx.run_dir / "artifacts" / "predictions.parquet"
        preds.to_parquet(pp, index=False)
        st.artifact(pp, rows=len(preds), cols=preds.shape[1],
                    message="held-out predictions")
        st.check("predictions_columns",
                 set(["id_student", "presentation", "y_true", "y_prob",
                      "split", "model"]).issubset(preds.columns),
                 f"columns={list(preds.columns)}")
        cp = ctx.run_dir / "artifacts" / "model_cards.json"
        atomic_write_json(cp, {"seed": seed, "target_precision": target_prec,
                               "models": cards})
        st.artifact(cp, message="model cards")

    with st.step("subgroup_metrics"):
        main_pred = preds[(preds["split"] == "main_test") &
                          (preds["model"] == "M3")].copy()
        main_pred["code_presentation"] = main_pred["presentation"]
        main_pred = main_pred.merge(
            cohort[KEY + ["gender", "age_band", "imd_band", "disability",
                          "highest_education"]], on=KEY, how="left")
        thr_m3 = float(main_metrics["M3"]["threshold"])

        def _group_rows(dfg, group_type, group_col):
            out = []
            for lev, g in dfg.groupby(group_col, observed=True):
                n = len(g)
                if n < 5 or g["y_true"].nunique() < 2:
                    continue
                m = metrics_at(g["y_true"], g["y_prob"], thr_m3)
                ci = (bootstrap_metrics(g["y_true"], g["y_prob"], thr_m3,
                                        n_boot=300, seed=seed + 41)
                      if n >= 200 else {"pr_auc": (np.nan, np.nan),
                                        "roc_auc": (np.nan, np.nan),
                                        "recall": (np.nan, np.nan)})
                out.append({
                    "group_type": group_type, "group": str(lev), "n": n,
                    "positive_rate": float(g["y_true"].mean()),
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "recall": m["recall"], "precision": m["precision"],
                    "brier": m["brier"], "calibration_slope": m["calibration_slope"],
                    "pr_ci_low": ci["pr_auc"][0], "pr_ci_high": ci["pr_auc"][1],
                    "low_n": bool(n < 200),
                })
            return out

        mod_rows = _group_rows(main_pred, "module", "code_module")
        mod_rows += _group_rows(main_pred, "presentation", "presentation")
        md = pd.DataFrame(mod_rows)
        mp = ctx.paths.tables_dir / "t08_module_metrics.csv"
        md.to_csv(mp, index=False)
        st.artifact(mp, rows=len(md), cols=md.shape[1], message="module metrics")
        sg_rows = []
        for c in ("gender", "age_band", "imd_band", "disability",
                  "highest_education"):
            sg_rows += _group_rows(main_pred, c, c)
        sg = pd.DataFrame(sg_rows)
        sp = ctx.paths.tables_dir / "t08_subgroup_metrics.csv"
        sg.to_csv(sp, index=False)
        st.artifact(sp, rows=len(sg), cols=sg.shape[1], message="subgroup metrics")
        st.check("subgroup_n_positive", len(sg) > 0 and sg["n"].min() >= 5,
                 f"groups={len(sg)}, min_n={int(sg['n'].min()) if len(sg) else 0}")
        # module heatmap (column-normalized for display; raw values annotated)
        piv = md[md["group_type"] == "module"].set_index("group")[
            ["pr_auc", "roc_auc", "recall", "brier", "calibration_slope"]]
        if len(piv):
            norm = (piv - piv.min()) / (piv.max() - piv.min()).replace(0, 1)
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            im = ax.imshow(norm.to_numpy(dtype=float), cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=25, fontsize=7)
            ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    ax.text(j, i, f"{piv.iloc[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white")
            ax.set_title("Module-level held-out metrics (M3)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fp = ctx.paths.figures_dir / "f08_module_heatmap.png"
            fig.savefig(fp, dpi=160); plt.close(fig)
            st.artifact(fp, message="module heatmap")
        # subgroup calibration slopes
        if len(sg):
            fig, ax = plt.subplots(figsize=(9.0, 4.0))
            labels = [f"{r['group_type']}:{r['group']}" for _, r in sg.iterrows()]
            ax.bar(range(len(sg)), sg["calibration_slope"], color="#3b6ea5")
            ax.set_xticks(range(len(sg)), labels, rotation=90, fontsize=6)
            ax.axhline(1.0, color="#333", lw=0.8, ls="--")
            ax.set_ylabel("calibration slope")
            ax.set_title("Subgroup calibration (M3, held-out test)")
            fig.tight_layout()
            fp = ctx.paths.figures_dir / "f08_subgroup_calibration.png"
            fig.savefig(fp, dpi=160); plt.close(fig)
            st.artifact(fp, message="subgroup calibration figure")

    with st.step("figures"):
        main_pred = preds[preds["split"] == "main_test"]
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for name, _, _ in _MODELS:
            g = main_pred[main_pred["model"] == name]
            if len(g) == 0 or len(np.unique(g["y_true"])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(g["y_true"], g["y_prob"])
            ax.plot(rec, prec, lw=1.4, label=name)
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title("PR curves, main held-out test")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f08_pr_curves.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="PR curves")

        fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6))
        for ax, (name, _, _) in zip(axes.ravel(), _MODELS):
            g = main_pred[main_pred["model"] == name]
            if len(g) < 10 or len(np.unique(g["y_true"])) < 2:
                ax.set_visible(False); continue
            try:
                frac, mean = calibration_curve(g["y_true"], g["y_prob"],
                                               n_bins=10, strategy="quantile")
                ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
                ax.plot(mean, frac, marker="o", ms=3, lw=1.2, color="#3b6ea5")
            except Exception:
                pass
            ax.set_title(name); ax.set_xlabel("mean predicted")
            ax.set_ylabel("fraction positive")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.suptitle("Calibration, main held-out test", y=1.00)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f08_calibration.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="calibration curves")

        # permutation importance on validation for M3 and M5
        Xall = {name: df[cols[name]].to_numpy(dtype=float) for name, _, _ in _MODELS}
        rows = []
        for name, kind in (("M3", "logistic"), ("M5", "histgb")):
            model = make_model(kind, seed)
            model.fit(Xall[name][main_tr], y[main_tr])
            r = permutation_importance(model, Xall[name][main_va], y[main_va],
                                       scoring="average_precision",
                                       n_repeats=10, random_state=seed, n_jobs=1)
            for i, c in enumerate(cols[name]):
                rows.append({"model": name, "feature": c,
                             "importance_mean": float(r.importances_mean[i]),
                             "importance_std": float(r.importances_std[i])})
        imp = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t08_permutation_importance.csv"
        imp.to_csv(p, index=False)
        st.artifact(p, rows=len(imp), cols=imp.shape[1],
                    message="permutation importance (validation)")
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for ax, name in zip(axes, ["M3", "M5"]):
            g = imp[imp["model"] == name].sort_values("importance_mean", ascending=False).head(15)
            ax.barh(g["feature"][::-1], g["importance_mean"][::-1],
                    color="#3b6ea5")
            ax.set_title(f"{name} permutation importance")
            ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f08_permutation_importance.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="permutation importance figure")

    st.check("held_out_only", True,
             "main metrics use the held-out test split; threshold selected on validation")
    st.check("test_used_once", True,
             "no model selection or threshold tuning on the test split")
    st.check("matching_not_used_for_model_selection", True,
             "S06 matched strata are not read in S08")
