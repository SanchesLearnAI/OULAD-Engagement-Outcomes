"""S11 — robustness: windows, tolerances, duplicate events and a spec curve.

Every row is a sensitivity analysis of the main design, not a new main result.
The +21/+28 windows are reported twice: with post-click matching (comparable to
the main design) and with whole-window click matching (comparable to the earlier
+21/+28 study).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_master, read_splits, read_tma_features, read_vle
from src.tma import daily, effects, modeling, pairing
from src.tma import exploration
from src.tma.anchor import binary_cohort

from sklearn.metrics import average_precision_score, brier_score_loss

SPECS = [
    ("primary", {"post_days": 14, "tolerance": 0.05, "match_on": "post_clicks",
                 "dedupe": False}),
    ("tolerance_0", {"post_days": 14, "tolerance": 0.00, "match_on": "post_clicks",
                     "dedupe": False}),
    ("tolerance_10", {"post_days": 14, "tolerance": 0.10, "match_on": "post_clicks",
                      "dedupe": False}),
    ("window_21_post", {"post_days": 21, "tolerance": 0.05, "match_on": "post_clicks",
                        "dedupe": False}),
    ("window_28_post", {"post_days": 28, "tolerance": 0.05, "match_on": "post_clicks",
                        "dedupe": False}),
    ("window_21_total", {"post_days": 21, "tolerance": 0.05, "match_on": "total_clicks",
                         "dedupe": False}),
    ("window_28_total", {"post_days": 28, "tolerance": 0.05, "match_on": "total_clicks",
                         "dedupe": False}),
    ("dedupe_events", {"post_days": 14, "tolerance": 0.05, "match_on": "post_clicks",
                       "dedupe": True}),
]
METRIC_SUMMARY = ["submitted_by_cutoff", "submitted_after_due", "post_active_days",
                  "post_longest_inactive_run", "post_week2_share"]


def _features(vle: pd.DataFrame, cohort: pd.DataFrame, post_days: int,
              dedupe: bool) -> pd.DataFrame:
    source = vle
    if dedupe:
        source = vle.drop_duplicates(subset=KEY + ["id_site", "date", "sum_click"])
    lookup = cohort[KEY + ["due_day"]]
    matrix = daily.daily_events(source, lookup, pre_days=14, post_days=post_days)
    frame, _, _ = daily.build_daily_features(cohort, matrix, None, pre_days=14,
                                             post_days=post_days)
    frame["total_clicks"] = frame["pre_clicks"] + frame["post_clicks"]
    return frame


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    repeats = int(cfg["clustering"].get("max_pairs", 800))
    master = read_master(ctx)
    cohort = binary_cohort(master)
    vle = read_vle(ctx)

    with st.step("spec_curve"):
        cache = {}
        rows, effect_rows = [], []
        for name, spec in SPECS:
            key = (spec["post_days"], spec["dedupe"])
            if key not in cache:
                cache[key] = _features(vle, cohort, spec["post_days"], spec["dedupe"])
            frame = cache[key]
            column = spec["match_on"]
            pairs = pairing.make_pairs(frame, spec["tolerance"], seed, column)
            if pairs.empty:
                continue
            balance = pairing.balance_row(frame, pairs, spec["tolerance"], column)
            eff = effects.effect_table(frame, pairs, METRIC_SUMMARY, seed, repeats)
            for _, r in eff.iterrows():
                effect_rows.append({"spec": name, "post_days": spec["post_days"],
                                    "tolerance": spec["tolerance"],
                                    "match_on": column, "dedupe": spec["dedupe"],
                                    **r.to_dict()})
            rows.append({"spec": name, "post_days": spec["post_days"],
                         "tolerance": spec["tolerance"], "match_on": column,
                         "dedupe": spec["dedupe"], "pairs": balance["pairs"],
                         "fail_coverage": balance["fail_coverage"],
                         "pre_log_click_smd": balance["pre_log_click_smd"],
                         "post_log_click_smd": balance["post_log_click_smd"]})
        table = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t11_specs.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1], message="specification curve")
        st.metric("specs", {"specifications": int(len(table)),
                            "pairs_min": int(table["pairs"].min()) if len(table) else 0,
                            "pairs_max": int(table["pairs"].max()) if len(table) else 0})
        st.check("primary_spec_present", "primary" in set(table["spec"]),
                 "the pre-specified primary specification is in the curve")

        effects_table = pd.DataFrame(effect_rows)
        p = ctx.paths.tables_dir / "t11_spec_effects.csv"
        effects_table.to_csv(p, index=False)
        st.artifact(p, rows=len(effects_table),
                    cols=effects_table.shape[1] if len(effects_table) else 0,
                    message="paired effects across specifications")

    with st.step("window_sensitivity"):
        window = table[table["spec"].str.startswith("window")].copy()
        p = ctx.paths.tables_dir / "t11_window_sensitivity.csv"
        window.to_csv(p, index=False)
        st.artifact(p, rows=len(window), cols=window.shape[1] if len(window) else 0,
                    message="+21/+28 window versions")
        st.check("window_versions_labelled",
                 set(window["match_on"]) <= {"post_clicks", "total_clicks"},
                 "each window is labelled by its matching variable")

    with st.step("negative_control"):
        # Persist the spec curve's sign stability: if a direction flips across
        # specifications, the report must say so instead of quoting one number.
        eff = pd.DataFrame(effect_rows)
        if len(eff):
            summary = (eff.assign(negative=eff["fail_minus_pass"] < 0)
                       .groupby("feature")["negative"]
                       .agg(["sum", "count"]).reset_index()
                       .rename(columns={"sum": "specs_negative", "count": "specs"}))
            summary["share_negative"] = summary["specs_negative"] / summary["specs"]
            p = ctx.paths.tables_dir / "t11_direction_stability.csv"
            summary.to_csv(p, index=False)
            st.artifact(p, rows=len(summary), cols=summary.shape[1],
                        message="sign stability across specifications")
            for feature in ("post_active_days", "post_longest_inactive_run"):
                row = summary[summary["feature"].eq(feature)]
                if len(row):
                    st.metric(f"direction_{feature}",
                              {"share_negative": float(row["share_negative"].iloc[0]),
                               "specs": int(row["specs"].iloc[0])})

    with st.step("exploration_c1_background"):
        flags = cfg.get("exploration", {}) or {}
        features = read_tma_features(ctx)
        main_pairs = pd.read_parquet(
            ctx.paths.outputs_dir / f"pairs_{int(round(float(cfg['study']['tolerance']) * 100)):02d}.parquet")
        pools = {col: ctx.paths.interim_dir / f"explore_{col}.csv" for col in ()}
        if flags.get("background_balance", False):
            # S04 keeps the pre-entry covariates on the feature table, so the row
            # order matches the pair indices directly.
            merged = features
            table = exploration.background_balance(merged, main_pairs)
            p = ctx.paths.tables_dir / "t11_x_background_balance.csv"
            table.to_csv(p, index=False)
            st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                        message="C1 pre-entry background of the pairs")
            if len(table):
                worst = float(table["abs_smd_matched"].max())
                st.metric("explore_background", {
                    "covariates": int(len(table)),
                    "max_abs_smd_matched": worst,
                    "covariates_within_0_10": int(table["balanced_matched"].sum())})
                st.check("explore_background_recorded", True,
                         f"max abs SMD after pairing={worst:.3f}")
            subset, info = exploration.constrained_repair(merged, main_pairs)
            p = ctx.paths.tables_dir / "t11_x_background_constrained.csv"
            info.to_csv(p, index=False)
            st.artifact(p, rows=len(info), cols=info.shape[1],
                        message="C1 pairs that also agree on education and IMD")
            if len(subset):
                eff = effects.effect_table(features, subset,
                                           ["post_active_days", "post_longest_inactive_run",
                                            "post_week2_share"], seed, repeats)
                eff.insert(0, "exact_on", "highest_education + imd_band")
                p = ctx.paths.tables_dir / "t11_x_background_effects.csv"
                eff.to_csv(p, index=False)
                st.artifact(p, rows=len(eff), cols=eff.shape[1],
                            message="C1 effects on the education/IMD-agreeing subset")

    with st.step("exploration_c2_rematching"):
        flags = cfg.get("exploration", {}) or {}
        if flags.get("rematching_uncertainty", False):
            repeats_cap = int(flags.get("rematching_repeats", 20))
            summaries, draws = [], []
            for metric in ("post_active_days", "post_longest_inactive_run"):
                summary, values = exploration.rematching_interval(
                    features, metric=metric,
                    tolerance=float(cfg["study"]["tolerance"]), seed=seed,
                    repeats=repeats_cap)
                summaries.append(summary)
                values["metric"] = metric
                draws.append(values)
            table = pd.concat(summaries, ignore_index=True)
            p = ctx.paths.tables_dir / "t11_x_rematching.csv"
            table.to_csv(p, index=False)
            st.artifact(p, rows=len(table), cols=table.shape[1],
                        message="C2 resample-then-repair intervals")
            values = pd.concat(draws, ignore_index=True)
            p = ctx.paths.tables_dir / "t11_x_rematching_draws.csv"
            values.to_csv(p, index=False)
            st.artifact(p, rows=len(values), cols=values.shape[1],
                        message="C2 bootstrap draws")
            st.metric("explore_rematching", {
                r["metric"]: {"observed": float(r["observed"]),
                              "ci_low": float(r["ci_low"]),
                              "ci_high": float(r["ci_high"]),
                              "repeats_used": int(r["repeats_used"])}
                for _, r in table.iterrows()})
            wide = float((table["ci_high"] - table["ci_low"]).max())
            st.check("explore_rematching_reported", True,
                     f"widest re-matched interval={wide:.3f}")

        if flags.get("heterogeneity", False):
            pres = t_present = pd.read_csv(
                ctx.paths.tables_dir / "t06_presentation_effects.csv")
            rows = []
            for metric in ("post_active_days", "post_longest_inactive_run",
                           "post_week2_share"):
                summary = exploration.heterogeneity(pres, metric)
                if len(summary):
                    rows.append(summary)
            table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            p = ctx.paths.tables_dir / "t11_x_heterogeneity.csv"
            table.to_csv(p, index=False)
            st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                        message="C2 random-effects summary across presentations")
            if len(table):
                st.metric("explore_heterogeneity", {
                    r["metric"]: {"pooled": float(r["pooled_effect"]),
                                  "i_squared": float(r["i_squared"])}
                    for _, r in table.iterrows()})

    with st.step("exploration_c3_non_submitters"):
        flags = cfg.get("exploration", {}) or {}
        if flags.get("non_submitters", False):
            outcomes = exploration.non_submitter_outcomes(master)
            p = ctx.paths.tables_dir / "t11_x_non_submitter_outcomes.csv"
            outcomes.to_csv(p, index=False)
            st.artifact(p, rows=len(outcomes), cols=outcomes.shape[1] if len(outcomes) else 0,
                        message="C3 outcomes among records without a submission by d+14")
            # Behaviour needs the three-outcome cohort, which S10 rebuilds.
            st.metric("explore_non_submitters", {
                "non_submitters": int(outcomes["non_submitters"].iloc[0]) if len(outcomes) else 0,
                "outcomes": int(len(outcomes))})
            st.check("explore_non_submitters_recorded", True,
                     "three-outcome split of the non-submitting subgroup")

    with st.step("exploration_c4_at_risk"):
        flags = cfg.get("exploration", {}) or {}
        if flags.get("at_risk_warning", False):
            risk = master[master["eligible"] & master["due_day"].notna()].reset_index(drop=True)
            risk["y_t3"] = risk["final_result"].map(
                {"Pass": "Pass", "Distinction": "Pass", "Fail": "Fail",
                 "Withdrawn": "Withdrawn"})
            risk = risk.sort_values(KEY).reset_index(drop=True)
            lookup = risk[KEY + ["due_day"]]
            matrix = daily.daily_events(read_vle(ctx), lookup, pre_days=14, post_days=14)
            risk_features, _, _ = daily.build_daily_features(
                risk, matrix, None, pre_days=14, post_days=14)
            # The warning target needs its own student-grouped partition: the
            # binary split cannot be reused because this cohort is larger.
            risk_features["y_fail"] = risk_features["y_t3"].isin(
                ["Fail", "Withdrawn"]).astype(int)
            from src.splits import build_tma_splits
            _, risk_splits, _ = build_tma_splits(risk_features, cfg.data, seed)
            data, table = exploration.at_risk_warning_set(
                risk_features, risk_splits, seed=seed, models=["M0", "M1", "M2"])
            p = ctx.paths.tables_dir / "t11_x_at_risk_warning.csv"
            table.to_csv(p, index=False)
            st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                        message="C4 Fail-or-Withdrawn prediction in the at-risk set")
            if len(table):
                st.metric("explore_at_risk", {
                    "records": int(len(data)),
                    "positive_share": float(data["y_fail"].mean()),
                    "n_test": int(table["n_test"].iloc[0]),
                    "ap_M2": float(table.loc[table["model"].eq("M2"),
                                             "average_precision"].mean()),
                    "models": int(len(table))})
                st.check("explore_at_risk_recorded", True,
                         "withdrawal is a positive case here, unlike the binary design")

    with st.step("exploration_c5_volume_clusters"):
        flags = cfg.get("exploration", {}) or {}
        if flags.get("volume_clusters", False):
            features = read_tma_features(ctx)
            k_lo, k_hi = cfg["clustering"]["k_range"]
            candidates, labels, best_k = exploration.volume_clusters(
                features["post_clicks"].to_numpy(), seed,
                k_range=(k_lo, k_hi),
                n_init=int(cfg["clustering"].get("n_init", 20)))
            candidates.to_csv(ctx.paths.tables_dir / "t11_x_volume_cluster_candidates.csv",
                              index=False)
            st.artifact(ctx.paths.tables_dir / "t11_x_volume_cluster_candidates.csv",
                        rows=len(candidates), cols=candidates.shape[1],
                        message="C5 cluster candidates")
            tolerance = float(cfg["study"]["tolerance"])
            main_pairs = pairing.make_pairs(features, tolerance, seed,
                                            str(cfg["study"]["match_on"]))
            within, crossed = exploration.within_cluster_pairs(main_pairs, labels)
            population, balance, cluster_effects = exploration.cluster_summary(
                features, within, labels, best_k,
                metrics=list(cfg["study"]["primary_metrics"]),
                seed=seed, repeats=repeats)
            for name, table in (("population", population), ("balance", balance),
                                ("effects", cluster_effects)):
                p = ctx.paths.tables_dir / f"t11_x_volume_cluster_{name}.csv"
                table.to_csv(p, index=False)
                st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                            message=f"C5 within-cluster {name}")
            st.metric("explore_volume_clusters", {
                "k": int(best_k), "pairs_total": int(len(main_pairs)),
                "pairs_within_cluster": int(len(within)),
                "pairs_crossing_clusters": int(crossed),
                "cluster_sizes": population.set_index("cluster")["records"].to_dict()})
            st.check("explore_volume_clusters_recorded", True,
                     f"k={best_k}; {crossed} pairs dropped because the two sides "
                     f"fell in different volume clusters")

    with st.step("exploration_c6_relative_gap"):
        flags = cfg.get("exploration", {}) or {}
        if flags.get("relative_gap", False):
            features = read_tma_features(ctx)
            # Exploratory arm only: the frozen ladder (S08) must not change.
            base = [c for c in modeling.columns_for(features, "M2") if c in features]
            candidate = base + ["relative_click_gap"]
            splits = read_splits(ctx)
            tr = np.array(splits["main"]["train"])
            va = np.array(splits["main"]["val"])
            te = np.array(splits["main"]["test"])
            rows = []
            preds = {}
            for name, cols in (("M2_frozen", base), ("M2_plus_relative_gap", candidate)):
                prob, thr, rule = modeling._fit(features, cols, tr, te, "logistic",
                                               seed, va,
                                               float(cfg["models"]["precision_target"]))
                preds[name] = prob
                rows.append({"model": name, "n_features": len(cols),
                             "average_precision": float(
                                 average_precision_score(features["y_fail"].to_numpy()[te],
                                                         prob)),
                             "brier": float(brier_score_loss(
                                 features["y_fail"].to_numpy()[te], prob)),
                             "threshold_rule": rule})
            table = pd.DataFrame(rows)
            p = ctx.paths.tables_dir / "t11_x_relative_gap.csv"
            table.to_csv(p, index=False)
            st.artifact(p, rows=len(table), cols=table.shape[1],
                        message="C6 relative pre/post click gap ablation")
            gap = (table.loc[table["model"].eq("M2_plus_relative_gap"),
                             "average_precision"].iloc[0]
                   - table.loc[table["model"].eq("M2_frozen"),
                               "average_precision"].iloc[0])
            st.metric("explore_relative_gap", {
                "delta_ap": float(gap),
                "gap_mean": float(features["relative_click_gap"].mean()),
                "both_zero_records": int(features["both_zero_clicks"].sum())})
            st.check("explore_relative_gap_recorded", True,
                     f"exploratory arm only; delta AP={gap:+.5f}")

    with st.step("figure"):
        eff = pd.DataFrame(effect_rows)
        if len(eff):
            pivot = eff.pivot_table(index="spec", columns="feature",
                                    values="fail_minus_pass", aggfunc="first")
            fig, ax = plt.subplots(figsize=(7.4, 3.8))
            order = pivot.index.tolist()
            x = np.arange(len(order))
            width = 0.8 / max(len(METRIC_SUMMARY), 1)
            for i, feature in enumerate([f for f in METRIC_SUMMARY if f in pivot.columns]):
                ax.bar(x + (i - 0.5) * width, pivot[feature], width, label=feature)
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(x, order, rotation=20, ha="right")
            ax.set_ylabel("Fail minus Pass/Distinction")
            ax.set_title("Primary directions across robustness specifications")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fp = ctx.paths.figures_dir / "f11_spec_curve.png"
            fig.savefig(fp, dpi=170)
            plt.close(fig)
            st.artifact(fp, message="specification curve figure")
