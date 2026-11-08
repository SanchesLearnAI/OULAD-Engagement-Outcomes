"""S08 — model ladder, calibration, label-permutation control and parity check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import read_splits, read_tma_features
from src.tma import modeling

REFERENCE = {
    # Published TMA-branch values (results/tma_two_week_20261103/models.csv and
    # results/tma_submission_timing_20261103/models.csv), five-seed means.
    "M0": 0.666262, "M1": 0.668890, "M2": 0.671415,
    "AP_M4_SCORE": 0.706335,
}


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    models_cfg = cfg["models"]
    frame = read_tma_features(ctx)
    splits = read_splits(ctx)
    target = float(models_cfg.get("precision_target", 0.80))
    include_scores = bool(cfg["leakage"]["include_scores"])
    models = [m for m in models_cfg["ladder"] if include_scores or m != "M4"]
    rf_models = list(models_cfg.get("rf_models", []))
    primary = str(models_cfg["primary_model"])

    st.check("rows_match_splits", len(frame) == splits["n"],
             f"features={len(frame)} splits={splits['n']}")
    st.check("binary_label_fail", set(np.unique(frame["y_fail"]).tolist()) == {0, 1},
             "y_fail in {0,1} with Fail = 1")
    st.check("scores_excluded_from_strict_models",
             not include_scores,
             "M0–M3 contain no score column; M4 is the upper bound")

    with st.step("ladder"):
        table = modeling.evaluate(frame, splits, seed=seed, models=models,
                                  rf_models=rf_models, precision_target=target)
        p = ctx.paths.tables_dir / "t08_model_metrics.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1],
                    message="metrics by model, split and seed")
        main = table[table["split"].eq("main_test")]
        for model, group in main.groupby("model"):
            st.metric(f"main_{model}", {
                "average_precision": float(group["average_precision"].mean()),
                "roc_auc": float(group["roc_auc"].mean()),
                "brier": float(group["brier"].mean()),
                "no_skill": float(group["no_skill_pr_auc"].mean())})
        st.check("ladder_complete", set(models).issubset(set(main["model"])),
                 f"models={sorted(main['model'].unique())}")

    with st.step("primary_model"):
        row = modeling._row  # keep the helper imported for clarity
        main = pd.read_csv(ctx.paths.tables_dir / "t08_model_metrics.csv")
        primary_rows = main[(main["split"].eq("main_test")) & main["model"].eq(primary)]
        if len(primary_rows):
            st.metric("primary_model", {
                "name": primary,
                "ap": float(primary_rows["average_precision"].mean()),
                "no_skill": float(primary_rows["no_skill_pr_auc"].mean()),
                "roc_auc": float(primary_rows["roc_auc"].mean()),
                "brier": float(primary_rows["brier"].mean()),
                "threshold_rules": sorted(primary_rows["threshold_rule"].unique().tolist())})
            if ctx.mode == "full":
                ap = float(primary_rows["average_precision"].mean())
                st.check("primary_ap_above_no_skill",
                         ap > float(primary_rows["no_skill_pr_auc"].mean()),
                         f"primary AP={ap:.4f} above prevalence")
                # The published value comes from a five-seed 80/20 holdout; the
                # 60/20/20 main split answers a slightly different question, so
                # the exact comparison is t08_parity.csv below.
                st.check("primary_ap_near_reference",
                         abs(ap - REFERENCE["M2"]) <= 0.02,
                         f"primary AP={ap:.4f}, TMA branch 80/20={REFERENCE['M2']:.4f} "
                         "(exact comparison in t08_parity.csv)")
        else:
            raise SystemExit("primary model missing from the ladder")

    with st.step("calibration"):
        rows = []
        y = frame["y_fail"].to_numpy(dtype=int)
        tr = np.array(splits["main"]["train"])
        va = np.array(splits["main"]["val"])
        te = np.array(splits["main"]["test"])
        for model in models:
            cols = modeling.columns_for(frame, model)
            prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", seed, va, target)
            if prob is None:
                continue
            curve = modeling.calibration_curve_table(
                y[te], prob, int(models_cfg.get("calibration_bins", 10)))
            curve["model"] = model
            curve["threshold_rule"] = rule
            rows.append(curve)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        p = ctx.paths.tables_dir / "t08_calibration.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                    message="reliability-curve data")
        if len(table):
            fig, ax = plt.subplots(figsize=(4.6, 4.4))
            for model, group in table.groupby("model"):
                ax.plot(group["mean_predicted"], group["observed_rate"], marker="o",
                        ms=3.5, lw=1.2, label=model)
            ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("observed Fail rate")
            ax.set_title("Calibration on the main test split")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fp = ctx.paths.figures_dir / "f08_calibration.png"
            fig.savefig(fp, dpi=170)
            plt.close(fig)
            st.artifact(fp, message="calibration figure")

    with st.step("threshold_rules"):
        main = pd.read_csv(ctx.paths.tables_dir / "t08_model_metrics.csv")
        rules = (main[main["split"].eq("main_test")].groupby("model")["threshold_rule"]
                 .agg(lambda s: ", ".join(sorted(set(s)))).reset_index())
        p = ctx.paths.tables_dir / "t08_threshold_rules.csv"
        rules.to_csv(p, index=False)
        st.artifact(p, rows=len(rules), cols=rules.shape[1],
                    message="threshold rule actually used per model")
        st.metric("threshold_rules", {r["model"]: r["threshold_rule"]
                                      for _, r in rules.iterrows()})

    with st.step("label_permutation"):
        perm = modeling.label_permutation(
            frame, splits, primary, seed=seed,
            repeats=int(models_cfg.get("label_permutations", 20)),
            precision_target=target)
        p = ctx.paths.tables_dir / "t08_label_permutation.csv"
        perm.to_csv(p, index=False)
        st.artifact(p, rows=len(perm), cols=perm.shape[1] if len(perm) else 0,
                    message="shuffled-label control")
        if len(perm):
            observed = float(perm["average_precision"].mean())
            prevalence = float(perm["fail_prevalence"].mean())
            st.metric("label_permutation", {"mean_ap": observed,
                                            "fail_prevalence": prevalence,
                                            "gap": observed - prevalence})
            # The 500-record rehearsal sample gives the shuffled AP a wide
            # standard error, so the protocol threshold is enforced on full runs.
            near = abs(observed - prevalence) <= 0.02
            st.check("permutation_near_chance", near,
                     f"shuffled AP={observed:.4f}, prevalence={prevalence:.4f}",
                     warn_only=ctx.mode != "full")

    with st.step("parity_80_20"):
        table = modeling.parity(frame, precision_target=target)
        p = ctx.paths.tables_dir / "t08_parity.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1] if len(table) else 0,
                    message="80/20 grouped holdout with the TMA-branch estimator")
        if len(table):
            means = table.groupby("model")["average_precision"].mean().to_dict()
            st.metric("parity", {k: float(v) for k, v in means.items()})
            for model in ("M0", "M1", "M2"):
                if model in means and ctx.mode == "full":
                    st.check(f"parity_{model}", abs(means[model] - REFERENCE[model]) <= 0.005,
                             f"{model}: parity={means[model]:.6f}, "
                             f"published={REFERENCE[model]:.6f}")

    with st.step("figure"):
        table = pd.read_csv(ctx.paths.tables_dir / "t08_model_metrics.csv")
        main = table[table["split"].eq("main_test")]
        folds = table[table["split"].str.startswith("group_kfold")]
        order = [m for m in models + [f"RF_{m}" for m in rf_models]
                 if m in set(main["model"])]
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        means = [float(main.loc[main["model"].eq(m), "average_precision"].mean()) for m in order]
        # Whiskers show the grouped-fold spread, which is wider than the single
        # main holdout and is the honest uncertainty statement for the ladder.
        lows = [float(min(means[i], folds.loc[folds["model"].eq(m), "average_precision"].min())
                      if len(folds.loc[folds["model"].eq(m)]) else means[i])
                for i, m in enumerate(order)]
        highs = [float(max(means[i], folds.loc[folds["model"].eq(m), "average_precision"].max())
                       if len(folds.loc[folds["model"].eq(m)]) else means[i])
                 for i, m in enumerate(order)]
        x = np.arange(len(order))
        ax.bar(x, means, 0.6, color="#3b6ea5")
        ax.errorbar(x, means, yerr=[np.array(means) - lows, np.array(highs) - means],
                    fmt="none", ecolor="#333", capsize=3)
        ax.axhline(float(main["no_skill_pr_auc"].mean()), color="#c05b5b", ls=":", lw=1,
                   label="no-skill (Fail prevalence)")
        ax.set_xticks(x, order, rotation=12)
        ax.set_ylabel("Fail average precision")
        ax.set_title("Model ladder on the grouped main test split")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f08_model_ladder.png"
        fig.savefig(fp, dpi=170)
        plt.close(fig)
        st.artifact(fp, message="model ladder figure")
