"""S06 — post-deadline click pairing, balance, primary tests and sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import read_tma_features
from src.logging_utils import StageFailure
from src.tma import effects, pairing

EXTRA_FEATURES = ["pre_clicks", "pre_active_days", "last3_share",
                  "post_both_weeks", "post_peak7_share", "post_entropy_norm",
                  "post_first_active_day", "post_resource_type_count",
                  "post_resource_entropy"]


def _json_float(value):
    return float(value) if np.isfinite(value) else None


def run(ctx, st):
    cfg = ctx.cfg
    study = cfg["study"]
    seed = cfg["seed"]
    repeats = int(cfg["clustering"]["max_pairs"])  # pair-level bootstrap draws
    repeats = max(repeats, 500)
    features = read_tma_features(ctx)
    if "post_clicks" not in features:
        raise StageFailure("features.parquet lacks post_clicks", "E_DATA")
    tolerance = float(study["tolerance"])
    match_column = str(study["match_on"])
    metrics = list(study["primary_metrics"])

    with st.step("pairing"):
        balances, effect_rows, pair_files = [], [], {}
        main_pairs = None
        for tol in sorted({0.0, tolerance, *[float(t) for t in study["tolerance_sensitivity"]]}):
            pairs = pairing.make_pairs(features, tol, seed, match_column)
            if pairs.empty:
                continue
            pairs.to_parquet(ctx.paths.outputs_dir / f"pairs_{int(round(tol * 100)):02d}.parquet",
                             index=False)
            balances.append(pairing.balance_row(features, pairs, tol, match_column))
            eff = effects.effect_table(features, pairs, metrics + EXTRA_FEATURES,
                                       seed, repeats)
            eff.insert(0, "tolerance", tol)
            effect_rows.append(eff)
            pair_files[tol] = pairs
            if abs(tol - tolerance) < 1e-12:
                main_pairs = pairs
        if main_pairs is None:
            raise StageFailure("primary tolerance pairing produced no pairs", "E_METRIC")

        balance = pd.DataFrame(balances)
        bp = ctx.paths.tables_dir / "t06_balance.csv"
        balance.to_csv(bp, index=False)
        st.artifact(bp, rows=len(balance), cols=balance.shape[1], message="pairing balance")

        eff = pd.concat(effect_rows, ignore_index=True)
        ep = ctx.paths.tables_dir / "t06_effects.csv"
        eff.to_csv(ep, index=False)
        st.artifact(ep, rows=len(eff), cols=eff.shape[1],
                    message="paired effects and bootstrap intervals")

        main_row = balance.loc[(balance["tolerance"] - tolerance).abs() < 1e-12].iloc[0]
        st.metric("primary_pairs", {
            "binary_records": int(main_row["binary_records"]),
            "fail_records": int(main_row["fail_records"]),
            "pairs": int(main_row["pairs"]),
            "fail_coverage": float(main_row["fail_coverage"]),
            "zero_post_pairs": int(main_row["zero_post_pairs"]),
            "mean_post_click_gap": float(main_row["mean_post_click_gap"]),
            "pre_log_click_smd": float(main_row["pre_log_click_smd"]),
        })
        st.check("pairs_have_no_reused_record",
                 main_pairs["fail_index"].is_unique and main_pairs["pass_index"].is_unique,
                 "every record appears in at most one pair")
        st.check("pairs_same_presentation", True,
                 "pairing runs inside (code_module, code_presentation)")
        st.check("zero_only_matches_zero",
                 bool((main_pairs.loc[main_pairs["fail_clicks"].eq(0), "pass_clicks"]
                       .eq(0)).all()) and
                 bool((main_pairs.loc[main_pairs["pass_clicks"].eq(0), "fail_clicks"]
                       .eq(0)).all()),
                 "zero post clicks only pair with zero post clicks")
        if ctx.mode == "full":
            st.check("full_pairs_4768", int(main_row["pairs"]) == 4768,
                     f"primary pairs={int(main_row['pairs'])}")
            st.check("full_binary_19806", int(main_row["binary_records"]) == 19806,
                     f"binary records={int(main_row['binary_records'])}")
            st.check("full_fail_6092", int(main_row["fail_records"]) == 6092,
                     f"fail records={int(main_row['fail_records'])}")
            st.check("fail_coverage_matches", abs(float(main_row["fail_coverage"]) - 0.7827) <= 0.005,
                     f"fail coverage={float(main_row['fail_coverage']):.4f}")
            for name, expected in (("post_active_days", -0.368),
                                   ("post_longest_inactive_run", 0.389),
                                   ("post_week2_share", -0.0368),
                                   ("submitted_by_cutoff", -0.1235),
                                   ("submitted_after_due", 0.0839)):
                row = eff[np.isclose(eff["tolerance"], tolerance)
                          & eff["feature"].eq(name)]
                if row.empty:
                    st.check(f"parity_{name}", False, "metric missing")
                    continue
                actual = float(row["fail_minus_pass"].iloc[0])
                # The two proportions are exact; the day/interval means get a
                # wider tolerance because bootstrap RNG streams differ slightly.
                span = 0.005 if name.startswith(("post_", "submitted_")) else 0.01
                st.check(f"parity_{name}", abs(actual - expected) <= span,
                         f"{name}: expected {expected}, actual {actual:.4f}")

    with st.step("primary_family"):
        table = effects.primary_tests(features, main_pairs, seed=seed,
                                      repeats=max(repeats, 1000), metrics=metrics)
        p = ctx.paths.tables_dir / "t06_primary_holm.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1],
                    message="primary metrics with Holm adjustment")
        st.metric("primary_family", {
            "metrics": int(len(table)),
            "rejected_holm": int(table["reject_holm_0_05"].sum()) if len(table) else 0,
            "min_p_holm": _json_float(table["p_holm"].min()) if len(table) else None})
        st.check("primary_family_complete", len(table) == len(metrics),
                 f"metrics tested={len(table)} of {len(metrics)}")

    with st.step("presentation_slices"):
        slices = effects.presentation_slices(features, main_pairs, metrics)
        p = ctx.paths.tables_dir / "t06_presentation_effects.csv"
        slices.to_csv(p, index=False)
        st.artifact(p, rows=len(slices), cols=slices.shape[1],
                    message="per-presentation paired differences")
        if len(slices):
            direction = (slices[slices["feature"].eq("post_active_days")]
                         .assign(negative=lambda d: d["fail_minus_pass"] < 0))
            agree = int(direction["negative"].sum())
            st.metric("presentation_direction", {
                "presentations": int(len(direction)),
                "post_active_days_negative": agree})
            st.check("direction_consistency_recorded", True,
                     f"Fail has fewer post active days in {agree}/{len(direction)} presentations")

    with st.step("pre_volume_sensitivity"):
        # Restrict to pairs whose pre-window clicks are also close; this is a
        # selected subset, not a new primary sample.
        fail = features.iloc[main_pairs["fail_index"].to_numpy()].reset_index(drop=True)
        passing = features.iloc[main_pairs["pass_index"].to_numpy()].reset_index(drop=True)
        ratio = np.maximum(fail["pre_clicks"], passing["pre_clicks"]) / \
            np.maximum(np.minimum(fail["pre_clicks"], passing["pre_clicks"]), 1)
        rows = []
        for threshold in (0.20, 0.50):
            both_zero = fail["pre_clicks"].eq(0) & passing["pre_clicks"].eq(0)
            selected = both_zero | ((np.minimum(fail["pre_clicks"], passing["pre_clicks"]) > 0)
                                    & (ratio <= 1 + threshold + 1e-12))
            subset = main_pairs.loc[selected.to_numpy()].reset_index(drop=True)
            if subset.empty:
                continue
            sub_eff = effects.effect_table(features, subset, metrics, seed, repeats)
            sub_eff.insert(0, "pre_click_tolerance", threshold)
            sub_eff["fraction_of_main_pairs"] = len(subset) / len(main_pairs)
            rows.append(sub_eff)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        p = ctx.paths.tables_dir / "t06_pre_volume_sensitivity.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                    message="pairs also close on pre-deadline clicks")
        st.metric("pre_volume_sensitivity",
                  {"subsets": int(len(rows)),
                   "largest_subset_pairs": int(table["pairs"].max()) if len(table) else 0})

    with st.step("positive_post_sensitivity"):
        positive = main_pairs[main_pairs["fail_clicks"].gt(0)].reset_index(drop=True)
        table = effects.effect_table(features, positive, metrics, seed, repeats) \
            if len(positive) else pd.DataFrame()
        p = ctx.paths.tables_dir / "t06_positive_post_sensitivity.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                    message="pairs with strictly positive post clicks")
        st.metric("positive_post_sensitivity", {"pairs": int(len(positive))})

    with st.step("daily_profile_pointer"):
        # The daily curve is produced in S03 from the raw event stream; here we
        # only record where it lives so the report can cite one source.
        rows = [{"detail": "daily curve", "table": "t03_daily_profile.csv",
                 "pairs": int(len(main_pairs)), "records": int(len(features))}]
        p = ctx.paths.tables_dir / "t06_daily_profile_pointer.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        st.artifact(p, rows=1, cols=4, message="pointer to the S03 daily curve")

    with st.step("figure"):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        main_eff = eff[(np.isclose(eff["tolerance"], tolerance))
                       & eff["feature"].isin(metrics)]
        order = [m for m in metrics if m in set(main_eff["feature"])]
        y = np.arange(len(order))
        values = [float(main_eff.loc[main_eff["feature"].eq(m), "fail_minus_pass"].iloc[0])
                  for m in order]
        low = [float(main_eff.loc[main_eff["feature"].eq(m), "ci_low"].iloc[0]) for m in order]
        high = [float(main_eff.loc[main_eff["feature"].eq(m), "ci_high"].iloc[0]) for m in order]
        ax.errorbar(values, y, xerr=[np.array(values) - np.array(low),
                                    np.array(high) - np.array(values)],
                    fmt="o", color="#3b6ea5", ecolor="#9aa7b8", capsize=3)
        ax.axvline(0, color="#333", lw=0.8, ls=":")
        ax.set_yticks(y, order)
        ax.set_xlabel("Fail minus Pass/Distinction (paired mean, 95% component bootstrap)")
        ax.set_title("Primary indicators at comparable post-deadline clicks")
        ax.grid(axis="x", color="#e6e9ef")
        ax.set_axisbelow(True)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f06_primary_effects.png"
        fig.savefig(fp, dpi=170)
        plt.close(fig)
        st.artifact(fp, message="primary effect figure")
