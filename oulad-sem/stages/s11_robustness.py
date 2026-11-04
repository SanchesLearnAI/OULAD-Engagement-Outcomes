"""S11 — specification curve and negative controls."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.clustering import pca_reduce, per_presentation_z
from src.data import KEY, read_features, read_master
from src.features import (build_features, feature_columns, prepare_assessment,
                          prepare_vle)
from src.matching import MATCH_VARS, cem, w2_differences
from src.robustness import (cramers_v, evaluate, psm_effect,
                            stratified_effect)


def _window_features(vle, start, end):
    sub = vle[(vle["date"] >= start) & (vle["date"] <= end)].copy()
    if sub.empty:
        return pd.DataFrame(columns=KEY + ["win_clicks", "win_active_days",
                                           "win_entropy"])
    g = sub.groupby(KEY, observed=True)
    out = pd.DataFrame({
        "win_clicks": g["sum_click"].sum(),
        "win_active_days": g["date"].nunique(),
        "win_n_sites": g["id_site"].nunique(),
    }).reset_index()
    act = (sub.groupby(KEY + ["activity_type"], observed=True)["sum_click"].sum()
           .unstack(fill_value=0))
    p = act.div(act.sum(axis=1).where(act.sum(axis=1) > 0, 1), axis=0)
    ent = -(p * np.log(p.where(p > 0, 1))).sum(axis=1) / np.log((act > 0).sum(axis=1).clip(lower=1))
    out = out.merge(ent.rename("win_entropy").reset_index(), on=KEY, how="left")
    out["win_entropy"] = out["win_entropy"].fillna(0)
    return out


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    master = read_master(ctx)
    vle, _ = prepare_vle(ctx)
    sa = prepare_assessment(ctx)
    cutoffs = cfg["windows"]["cutoffs"]

    # Feature tables: baseline excludes pre-start; c28 also gets clip/feature.
    ftab = {}
    for c in cutoffs:
        ftab[(c, "exclude")] = read_features(ctx, c)
    for pre in ("clip", "feature"):
        f, _ = build_features(master, vle, sa, 28, cfg.data, pre_start=pre)
        ftab[(28, pre)] = f

    def spec_df(cutoff, pre, target, min_clicks):
        base = master if target == "T2" else master[master["y_t1"].notna()]
        col = "y_t2" if target == "T2" else "y_t1"
        d = base[KEY + [col]].merge(ftab[(cutoff, pre)], on=KEY, how="left",
                                    validate="one_to_one")
        d["y"] = d[col].astype(int)
        if min_clicks > 0:
            d = d[d["w1_clicks_total"] >= min_clicks]
        return d.reset_index(drop=True)

    rows = []
    seen = set()

    def add_predictive(spec_id, cutoff, pre, min_clicks, target, level, model):
        key = (cutoff, pre, min_clicks, target, level, model)
        if key in seen:
            return
        seen.add(key)
        d = spec_df(cutoff, pre, target, min_clicks)
        cols = feature_columns(d, level, cutoff,
                               include_pre_start=(pre == "feature"))
        m = evaluate(d, cols, model, seed)
        if m is None:
            return
        rows.append({
            "spec_id": spec_id, "family": "predictive",
            "cutoff": cutoff, "pre_start": pre, "min_w1_clicks": min_clicks,
            "target": target, "features": f"M{level}", "model": model,
            "metric_name": "pr_auc", "metric_value": m["pr_auc"],
            "roc_auc": m["roc_auc"], "n_train": m["n_train"],
            "n_test": m["n"], "positive_rate": m["positive_rate"],
            "config_hash": ctx.cfg.hash_ref,
        })

    # Baseline and one-at-a-time variations.
    add_predictive("P000_baseline", 28, "exclude", 0, "T1", 3, "logistic")
    i = 1
    for c in cutoffs:
        if c != 28:
            add_predictive(f"P{i:03d}_cutoff{c}", c, "exclude", 0, "T1", 3, "logistic"); i += 1
    for pre in ("clip", "feature"):
        add_predictive(f"P{i:03d}_pre_{pre}", 28, pre, 0, "T1", 3, "logistic"); i += 1
    for mc in (5, 10, 20):
        add_predictive(f"P{i:03d}_min{mc}", 28, "exclude", mc, "T1", 3, "logistic"); i += 1
    add_predictive(f"P{i:03d}_target_T2", 28, "exclude", 0, "T2", 3, "logistic"); i += 1
    for lev in (0, 1, 2):
        add_predictive(f"P{i:03d}_features_M{lev}", 28, "exclude", 0, "T1", lev, "logistic"); i += 1
    add_predictive(f"P{i:03d}_model_rf", 28, "exclude", 0, "T1", 3, "rf"); i += 1

    rng = np.random.default_rng(seed + 99)
    combos = set()
    while len(combos) < 20:
        combos.add((int(rng.choice(cutoffs)), int(rng.choice([0, 5, 10, 20])),
                    str(rng.choice(["T1", "T2"])), int(rng.choice([0, 1, 2, 3])),
                    str(rng.choice(["logistic", "rf"]))))
    for j, (c, mc, target, lev, model) in enumerate(sorted(combos)):
        add_predictive(f"P{i+j:03d}_combo", c, "exclude", mc, target, lev, model)
    pred = pd.DataFrame([r for r in rows if r["family"] == "predictive"])
    st.check("predictive_specs", len(pred) >= 30, f"n={len(pred)}")
    base_p = pred[pred.spec_id == "P000_baseline"]["metric_value"]
    base_val = float(base_p.iloc[0]) if len(base_p) else float("nan")

    # Matching-family effects on W2 click change.
    f56 = read_features(ctx, 56)
    w2cols = [c for c in f56.columns if c.startswith(("w2_", "d_", "accel_"))]
    d = master[master["y_t1"].notna()][KEY + ["y_t1"]].merge(
        ftab[(28, "exclude")], on=KEY, how="left", validate="one_to_one")
    d = d.merge(f56[KEY + w2cols], on=KEY, how="left", validate="one_to_one")
    d["y"] = d["y_t1"].astype(int)
    match_rows = []
    res = cem(d, cfg.data)
    wd = w2_differences(res, ["d_clicks"], n_boot=200, seed=seed)
    cem_val = float(wd["diff_pass_minus_fail"].iloc[0]) if len(wd) else float("nan")
    match_rows.append({"spec_id": "M000_cem", "family": "matching", "method": "CEM",
                       "metric_name": "d_clicks_pass_minus_fail",
                       "metric_value": cem_val, "n": res["n_matched"],
                       "config_hash": ctx.cfg.hash_ref})
    pv, pn = psm_effect(d, MATCH_VARS, "d_clicks")
    match_rows.append({"spec_id": "M001_psm", "family": "matching", "method": "PSM",
                       "metric_name": "d_clicks_pass_minus_fail",
                       "metric_value": pv, "n": pn,
                       "config_hash": ctx.cfg.hash_ref})
    sv, sn = stratified_effect(d, MATCH_VARS, "d_clicks")
    match_rows.append({"spec_id": "M002_stratified", "family": "matching",
                       "method": "stratified",
                       "metric_name": "d_clicks_pass_minus_fail",
                       "metric_value": sv, "n": sn,
                       "config_hash": ctx.cfg.hash_ref})

    # Clustering-family association across k.
    cl_rows = []
    cm = master[KEY + ["final_result"]].merge(ftab[(28, "exclude")], on=KEY,
                                              how="left", validate="one_to_one")
    cm = cm[cm["w1_active_days"] >= 1].reset_index(drop=True)
    cols = feature_columns(cm, 1, 28)
    Xdf = per_presentation_z(cm, cols, ["code_module", "code_presentation"])
    X, _ = pca_reduce(Xdf[cols].to_numpy(dtype=float), 8, seed)
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
        cl_rows.append({"spec_id": f"C{k:03d}_kmeans", "family": "clustering",
                        "method": f"kmeans_k{k}", "metric_name": "cramers_v",
                        "metric_value": cramers_v(lab, cm["final_result"]),
                        "n": len(cm), "config_hash": ctx.cfg.hash_ref})

    specs = pd.concat([pred, pd.DataFrame(match_rows), pd.DataFrame(cl_rows)],
                      ignore_index=True)
    p = ctx.paths.tables_dir / "t11_specs.csv"
    specs.to_csv(p, index=False)
    st.artifact(p, rows=len(specs), cols=specs.shape[1], message="specification table")
    st.metric("specifications", {
        "n_specs": int(len(specs)),
        "n_predictive": int(len(pred)),
        "baseline_pr_auc": base_val,
        "median_pr_auc": float(pred["metric_value"].median()),
        "min_pr_auc": float(pred["metric_value"].min()),
        "max_pr_auc": float(pred["metric_value"].max()),
        "share_above_baseline": float((pred["metric_value"] > base_val).mean()),
        "matching_effects": {r["method"]: r["metric_value"] for r in match_rows},
        "clustering_cramers_v": {r["method"]: r["metric_value"] for r in cl_rows},
    })
    st.check("spec_ids_present", specs["spec_id"].is_unique,
             f"n_specs={len(specs)}")

    # ---- WP1.4 measurement sensitivities ---------------------------------
    with st.step("quiz_sensitivity"):
        vle_q, qinfo = prepare_vle(ctx, include_quiz_sites=True)
        fq, _ = build_features(master, vle_q, sa, 28, cfg.data)
        dq = master[master["y_t1"].notna()][KEY + ["y_t1"]].merge(
            fq, on=KEY, how="left", validate="one_to_one")
        dq["y"] = dq["y_t1"].astype(int)
        qrows = []
        for name, lev in (("M1", 1), ("M3", 3)):
            strict = spec_df(28, "exclude", "T1", 0)
            ms = evaluate(strict, feature_columns(strict, lev, 28), "logistic", seed)
            mq = evaluate(dq, feature_columns(dq, lev, 28), "logistic", seed)
            if ms is None or mq is None:
                continue
            qrows.append({"model": name,
                          "strict_pr_auc": ms["pr_auc"],
                          "quiz_included_pr_auc": mq["pr_auc"],
                          "delta_pr_auc": mq["pr_auc"] - ms["pr_auc"],
                          "strict_roc_auc": ms["roc_auc"],
                          "quiz_included_roc_auc": mq["roc_auc"],
                          "strict_brier": ms["brier"],
                          "quiz_included_brier": mq["brier"],
                          "n_test": ms["n"]})
        qd = pd.DataFrame(qrows)
        qp = ctx.paths.tables_dir / "t11_quiz_sensitivity.csv"
        qd.to_csv(qp, index=False)
        st.artifact(qp, rows=len(qd), cols=qd.shape[1],
                    message="quiz-site sensitivity")
        st.metric("quiz_sensitivity", {r["model"]: round(float(r["delta_pr_auc"]), 4)
                                        for _, r in qd.iterrows()})

    with st.step("exposure_sensitivity"):
        exposures = {
            "total_clicks_only": ["w1_clicks_log1p"],
            "active_days_only": ["w1_active_days"],
            "activity_entropy_only": ["w1_entropy_activity"],
            "all_three": list(MATCH_VARS),
        }
        erows = []
        for name, mvars in exposures.items():
            rr = cem(d, cfg.data, match_vars=mvars)
            wd = w2_differences(rr, ["d_clicks"], n_boot=200, seed=seed + 51)
            erows.append({
                "exposure": name, "match_vars": "|".join(mvars),
                "n_matched": rr["n_matched"],
                "common_support_treated": rr["common_support_treated"],
                "binned_max_abs_smd": rr["binned_max_abs_smd"],
                "raw_max_abs_smd": max(abs(v) for v in rr["raw_after"].values()),
                "d_clicks_diff": float(wd["diff_pass_minus_fail"].iloc[0]),
                "d_clicks_ci_low": float(wd["ci_low"].iloc[0]),
                "d_clicks_ci_high": float(wd["ci_high"].iloc[0]),
            })
        ed = pd.DataFrame(erows)
        ep = ctx.paths.tables_dir / "t11_exposure_sensitivity.csv"
        ed.to_csv(ep, index=False)
        st.artifact(ep, rows=len(ed), cols=ed.shape[1],
                    message="engagement-definition sensitivity")
        st.metric("exposure_sensitivity", {
            r["exposure"]: round(float(r["d_clicks_diff"]), 3)
            for _, r in ed.iterrows()})

    with st.step("matching_bins"):
        bspecs = [
            ("baseline_10_fine_3", 10, None, 3),
            ("click_5", 5, None, 3),
            ("click_20", 20, None, 3),
            ("active_coarse", 10, [0, 1, 4, 8, 15, 29], 3),
            ("active_fine", 10, [0, 1, 4, 8, 15, 22, 29], 3),
            ("entropy_2", 10, None, 2),
            ("entropy_4", 10, None, 4),
        ]
        brows = []
        for name, cb, ae, eb in bspecs:
            rr = cem(d, cfg.data, max_levels=1, click_bins=cb,
                     active_edges=ae, entropy_bins=eb)
            wd = w2_differences(rr, ["d_clicks"], n_boot=200, seed=seed + 61)
            brows.append({
                "spec": name, "click_bins": cb,
                "active_edges": "|".join(str(x) for x in (ae or [])) or "default",
                "entropy_bins": eb, "n_matched": rr["n_matched"],
                "common_support_treated": rr["common_support_treated"],
                "binned_max_abs_smd": rr["binned_max_abs_smd"],
                "raw_max_abs_smd": max(abs(v) for v in rr["raw_after"].values()),
                "d_clicks_diff": float(wd["diff_pass_minus_fail"].iloc[0]),
                "d_clicks_ci_low": float(wd["ci_low"].iloc[0]),
                "d_clicks_ci_high": float(wd["ci_high"].iloc[0]),
            })
        bd = pd.DataFrame(brows)
        bp = ctx.paths.tables_dir / "t11_matching_bins.csv"
        bd.to_csv(bp, index=False)
        st.artifact(bp, rows=len(bd), cols=bd.shape[1],
                    message="CEM binning sensitivity")
        st.metric("matching_bins", {
            "n_specs": len(bd),
            "d_clicks_min": float(bd["d_clicks_diff"].min()),
            "d_clicks_max": float(bd["d_clicks_diff"].max())})

    with st.step("negative_control_exposure"):
        # WP5.3: W2 change in non-assessment resource types.
        non_assessment = {
            "homepage", "forumng", "resource", "subpage", "oucontent", "url",
            "page", "oucollaborate", "ouwiki", "dataplus", "ouelluminate",
            "glossary", "dualpane", "repeatactivity", "htmlactivity",
            "sharedsubpage", "folder",
        }
        na = vle[vle["activity_type"].isin(non_assessment)].copy()
        w1 = _window_features(na, 0, 27).rename(columns={
            "win_clicks": "na_w1_clicks",
            "win_active_days": "na_w1_active_days",
            "win_entropy": "na_w1_entropy",
        })
        w2 = _window_features(na, 28, 55).rename(columns={
            "win_clicks": "na_w2_clicks",
            "win_active_days": "na_w2_active_days",
            "win_entropy": "na_w2_entropy",
        })
        dna = master[master["y_t1"].notna()][KEY + ["y_t1"]].copy()
        dna = dna.merge(w1, on=KEY, how="left").merge(w2, on=KEY, how="left")
        for c in ("na_w1_clicks", "na_w2_clicks", "na_w1_active_days",
                  "na_w2_active_days", "na_w1_entropy", "na_w2_entropy"):
            if c not in dna.columns:
                dna[c] = 0.0
            dna[c] = dna[c].fillna(0.0)
        dna["y"] = dna["y_t1"].astype(int)
        dna["na_d_clicks"] = dna["na_w2_clicks"] - dna["na_w1_clicks"]
        dna["na_d_active_days"] = dna["na_w2_active_days"] - dna["na_w1_active_days"]
        dna["na_d_entropy"] = dna["na_w2_entropy"] - dna["na_w1_entropy"]
        mna = evaluate(dna, ["na_d_clicks", "na_d_active_days", "na_d_entropy"],
                       "logistic", seed + 500)
        expo_rows = []
        for metric in ("pr_auc", "roc_auc"):
            expo_rows.append({
                "exposure": "non_assessment_w2_change",
                "metric": metric,
                "value": float(mna[metric]) if mna else float("nan"),
                "n": int(mna["n"]) if mna else 0,
                "note": "W2 minus W1 over non-assessment resource types; "
                        "if predictive, generic engagement/confounding",
            })
        ed = pd.DataFrame(expo_rows)
        ep = ctx.paths.tables_dir / "t11_placebo_exposure.csv"
        ed.to_csv(ep, index=False)
        st.artifact(ep, rows=len(ed), cols=ed.shape[1],
                    message="negative-control exposure")
        st.metric("placebo_exposure", {
            r["metric"]: round(float(r["value"]), 4) for _, r in ed.iterrows()
            if np.isfinite(r["value"])})
        st.check("placebo_exposure_available", len(ed) == 2,
                 f"rows={len(ed)}; n={int(ed['n'].iloc[0]) if len(ed) else 0}")

    with st.step("placebo_outcome"):
        dp = d.merge(master[KEY + ["num_of_prev_attempts"]], on=KEY, how="left")
        dp["y"] = (dp["num_of_prev_attempts"] > 0).astype(int)
        cols_p = feature_columns(dp, 3, 28)
        mp = evaluate(dp, cols_p, "logistic", seed)
        if mp:
            pd.DataFrame([{"outcome": "num_of_prev_attempts_gt0",
                           "pr_auc": mp["pr_auc"], "roc_auc": mp["roc_auc"],
                           "n_test": mp["n"], "positive_rate": mp["positive_rate"],
                           "note": "diagnostic: behaviour may encode background"}]).to_csv(
                ctx.paths.tables_dir / "t11_placebo_outcomes.csv", index=False)
            st.artifact(ctx.paths.tables_dir / "t11_placebo_outcomes.csv",
                        rows=1, cols=6, message="placebo outcome")
            st.metric("placebo_outcome", {"roc_auc": mp["roc_auc"],
                                          "pr_auc": mp["pr_auc"]})

    # Negative controls.
    nc = []
    base = spec_df(28, "exclude", "T1", 0)
    cols3 = feature_columns(base, 3, 28)
    rocs = []
    for rep in range(3):
        d2 = base.copy()
        rng2 = np.random.default_rng(seed + 100 + rep)
        d2["y"] = d2.groupby(["code_module", "code_presentation"])["y"].transform(
            lambda s: rng2.permutation(s.to_numpy()))
        m = evaluate(d2, cols3, "logistic", seed + 200 + rep)
        if m:
            rocs.append(m["roc_auc"])
    nc.append({"control": "random_labels", "metric": "roc_auc",
               "role": "leak_check",
               "value": float(np.mean(rocs)) if rocs else float("nan"),
               "n_repeats": len(rocs),
               "note": "labels permuted within presentation; expected ~0.5"})
    pre = _window_features(vle, -25, -1)
    dpre = master[master["y_t1"].notna()][KEY + ["y_t1"]].merge(pre, on=KEY,
                                                              how="left")
    dpre["y"] = dpre["y_t1"].astype(int)
    for c in ["win_clicks", "win_active_days", "win_entropy"]:
        dpre[c] = dpre[c].fillna(0)
    mpre = evaluate(dpre, ["win_clicks", "win_active_days", "win_entropy"],
                    "logistic", seed + 300)
    nc.append({"control": "pre_start_only", "metric": "pr_auc",
               "role": "positive_control",
               "value": float(mpre["pr_auc"]) if mpre else float("nan"),
               "n_repeats": 1,
               "note": "pre-start clicks only; above no-skill, not a null"})
    plac = _window_features(vle, 56, 83)
    dpl = master[master["y_t1"].notna()][KEY + ["y_t1"]].merge(plac, on=KEY,
                                                              how="left")
    dpl["y"] = dpl["y_t1"].astype(int)
    for c in ["win_clicks", "win_active_days", "win_entropy"]:
        dpl[c] = dpl[c].fillna(0)
    mpl = evaluate(dpl, ["win_clicks", "win_active_days", "win_entropy"],
                   "logistic", seed + 400)
    nc.append({"control": "placebo_window_56_83", "metric": "pr_auc",
               "role": "placebo_failure",
               "value": float(mpl["pr_auc"]) if mpl else float("nan"),
               "n_repeats": 1,
               "note": "future window; post-outcome persistence, not a leak test"})
    if mp:
        nc.append({"control": "num_of_prev_attempts_gt0", "metric": "roc_auc",
                   "role": "negative_control_outcome",
                   "value": float(mp["roc_auc"]),
                   "n_repeats": 1,
                   "note": "negative-control outcome; behaviour may encode background"})
    ncd = pd.DataFrame(nc)
    p = ctx.paths.tables_dir / "t11_negative_controls.csv"
    ncd.to_csv(p, index=False)
    st.artifact(p, rows=len(ncd), cols=ncd.shape[1], message="negative controls")
    rand_roc = float(ncd.loc[ncd.control == "random_labels", "value"].iloc[0])
    st.metric("negative_controls", {r["control"]: r["value"] for _, r in ncd.iterrows()})
    st.check("random_label_near_half", 0.35 <= rand_roc <= 0.65,
             f"random-label ROC-AUC={rand_roc:.3f}", metrics={"roc_auc": rand_roc})
    if not (0.35 <= rand_roc <= 0.65):
        from src.logging_utils import StageFailure
        raise StageFailure(f"random-label ROC-AUC {rand_roc:.3f} outside "
                           "[0.35,0.65]: E_LEAK", "E_LEAK")

    # Specification curve: predictive and matching panels.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    s = pred.sort_values("metric_value").reset_index(drop=True)
    axes[0].plot(range(len(s)), s["metric_value"], "o", ms=3, color="#3b6ea5")
    axes[0].axhline(base_val, color="#b23b3b", ls="--", lw=1,
                    label=f"baseline={base_val:.3f}")
    axes[0].set_ylabel("PR-AUC"); axes[0].set_xlabel("specification (sorted)")
    axes[0].set_title("Predictive specifications")
    axes[0].legend(frameon=False, fontsize=8)
    md = pd.DataFrame(match_rows)
    axes[1].bar(md["method"], md["metric_value"], color="#6ea36f")
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_ylabel("d_clicks Pass minus Fail")
    axes[1].set_title("Matching alternatives (W2 change)")
    fig.tight_layout()
    fp = ctx.paths.figures_dir / "f11_specification_curve.png"
    fig.savefig(fp, dpi=160); plt.close(fig)
    st.artifact(fp, message="specification curve")
