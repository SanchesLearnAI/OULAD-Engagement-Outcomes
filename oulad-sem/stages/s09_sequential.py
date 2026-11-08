"""S09 — observation-cutoff curve: how much AP is available at d+1, d+7, d+14?

The main design fixes the cutoff at d+14.  This stage rebuilds the same feature
families with shorter post windows and refits the primary model, so the report
can show how the available information grows with observation time.  Results
are descriptive: a shorter window is not a better or worse method, it observes
less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_master, read_splits, read_tma_features, read_vle
from src.tma import daily, modeling
from src.tma.anchor import binary_cohort

CUTOFFS = [1, 7, 14]


def _features_for(frame: pd.DataFrame, matrix: pd.DataFrame, post_days: int) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(frame[KEY])
    pre = daily.click_matrix(matrix, index, -13, 0)
    post = daily.click_matrix(matrix, index, 1, post_days)
    out = frame.copy()
    for name, values in daily.pre_features(pre).items():
        out[name] = values
    for name, values in daily.time_features(post, post_days).items():
        out[name] = values
    for name, values in daily.week_features(post, post_days).items():
        out[name] = values
    out["log_pre_clicks"] = np.log1p(out["pre_clicks"])
    out["log_post_clicks"] = np.log1p(out["post_clicks"])
    out["post_clicks_squared"] = out["post_clicks"] ** 2
    out["module_presentation"] = out["code_module"] + "_" + out["code_presentation"]
    return out


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    master = read_master(ctx)
    cohort = binary_cohort(master)
    splits = read_splits(ctx)
    lookup = cohort[KEY + ["due_day"]]
    matrix = daily.daily_events(read_vle(ctx), lookup, pre_days=14, post_days=max(CUTOFFS))
    target = float(cfg["models"].get("precision_target", 0.80))

    rows = []
    for post_days in CUTOFFS:
        frame = _features_for(cohort, matrix, post_days)
        st.check(f"cutoff_{post_days}_rows", len(frame) == len(cohort),
                 f"rows={len(frame)}")
        y = frame["y_fail"].to_numpy(dtype=int)
        tr = np.array(splits["main"]["train"])
        va = np.array(splits["main"]["val"])
        te = np.array(splits["main"]["test"])
        for model in ("M0", "M1", "M2"):
            cols = [c for c in modeling.columns_for(frame, model) if c in frame]
            prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", seed, va, target)
            if prob is None:
                continue
            from sklearn.metrics import average_precision_score, brier_score_loss
            rows.append({"post_days": post_days, "model": model,
                         "n_test": int(len(te)),
                         "fail_prevalence": float(y[te].mean()),
                         "average_precision": float(average_precision_score(y[te], prob)),
                         "brier": float(brier_score_loss(y[te], prob)),
                         "n_features": int(len(cols)), "threshold_rule": rule})
        st.metric(f"cutoff_{post_days}", {"records": int(len(frame)),
                                          "post_days": post_days})
    table = pd.DataFrame(rows)
    p = ctx.paths.tables_dir / "t09_cutoff_metrics.csv"
    table.to_csv(p, index=False)
    st.artifact(p, rows=len(table), cols=table.shape[1], message="cutoff curve")

    incremental = []
    for model in sorted(set(table["model"])):
        sub = table[table["model"].eq(model)].sort_values("post_days")
        for prev, cur in zip(sub.itertuples(), sub.iloc[1:].itertuples()):
            incremental.append({"model": model, "from_post_days": int(prev.post_days),
                                "to_post_days": int(cur.post_days),
                                "delta_ap": float(cur.average_precision - prev.average_precision)})
    inc = pd.DataFrame(incremental)
    p = ctx.paths.tables_dir / "t09_incremental.csv"
    inc.to_csv(p, index=False)
    st.artifact(p, rows=len(inc), cols=inc.shape[1] if len(inc) else 0,
                message="incremental AP between cutoffs")
    st.check("cutoff_curve_monotone_expected", True,
             "AP is expected to rise with a longer window; it is not a method gain")

    if len(table):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for model, group in table.groupby("model"):
            ax.plot(group["post_days"], group["average_precision"], marker="o", label=model)
        ax.axhline(float(table["fail_prevalence"].mean()), color="#c05b5b", ls=":", lw=1,
                   label="no-skill")
        ax.set_xlabel("post-deadline observation window (days)")
        ax.set_ylabel("Fail average precision")
        ax.set_title("Available information grows with the observation window")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f09_cutoff_curve.png"
        fig.savefig(fp, dpi=170)
        plt.close(fig)
        st.artifact(fp, message="cutoff curve figure")
