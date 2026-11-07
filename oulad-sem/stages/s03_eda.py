"""S03 — descriptive audit: is "similar early engagement" empirically plausible?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_master, read_vle


def _w1_frame(master, vle, cutoff):
    end = min(27, cutoff - 1)
    v = vle[(vle["date"] >= 0) & (vle["date"] <= end)]
    agg = (v.groupby(KEY, observed=True)
           .agg(w1_clicks=("sum_click", "sum"),
                w1_active_days=("date", "nunique")).reset_index())
    out = master[KEY + ["final_result", "y_t1", "y_t2", "is_pass", "is_withdrawn",
                        "date_unregistration",
                        f"in_at_risk_c{cutoff}"]].merge(agg, on=KEY, how="left")
    out["w1_clicks"] = out["w1_clicks"].fillna(0)
    out["w1_active_days"] = out["w1_active_days"].fillna(0)
    return out, end


def _overlap(x, y, bins=20):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    hi = float(np.quantile(np.concatenate([x, y]), 0.99))
    if hi <= 0:
        return 1.0
    hx, edges = np.histogram(x, bins=bins, range=(0, hi))
    hy, _ = np.histogram(y, bins=edges)
    px = hx / max(hx.sum(), 1)
    py = hy / max(hy.sum(), 1)
    return float(np.minimum(px, py).sum())


def run(ctx, st):
    master = read_master(ctx)
    vle = read_vle(ctx)
    cutoffs = ctx.cfg["windows"]["cutoffs"]

    with st.step("w1_aggregates"):
        frames = {}
        for c in cutoffs:
            f, end = _w1_frame(master, vle, c)
            frames[c] = f
            st.metric(f"w1_c{c}", {
                "window_end": int(end),
                "n_enrolments": int(len(f)),
                "zero_click_share": float((f["w1_clicks"] == 0).mean()),
                "median_clicks": float(f["w1_clicks"].median()),
            })

    with st.step("outcome_distribution"):
        rows = []
        for c, f in frames.items():
            for res, g in f.groupby("final_result", observed=True):
                q = g["w1_clicks"].quantile([0.25, 0.5, 0.75])
                rows.append({
                    "cutoff": c, "final_result": res, "n": int(len(g)),
                    "median_clicks": float(q.loc[0.5]),
                    "q1_clicks": float(q.loc[0.25]),
                    "q3_clicks": float(q.loc[0.75]),
                    "mean_clicks": float(g["w1_clicks"].mean()),
                    "zero_click_share": float((g["w1_clicks"] == 0).mean()),
                    "median_active_days": float(g["w1_active_days"].median()),
                })
        dist = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t03_clicks_by_outcome.csv"
        dist.to_csv(p, index=False)
        st.artifact(p, rows=len(dist), cols=dist.shape[1],
                    message="W1 clicks by outcome")

    with st.step("overlap"):
        rows = []
        for c, f in frames.items():
            t1 = f[f["y_t1"].notna()].copy()
            t1["log_clicks"] = np.log1p(t1["w1_clicks"])
            per = []
            for (mod, pres), g in t1.groupby(["code_module", "code_presentation"],
                                             observed=True):
                pos = g[g["is_pass"] == 1]["log_clicks"]
                neg = g[g["is_pass"] == 0]["log_clicks"]
                ov = _overlap(pos, neg)
                if not np.isnan(ov):
                    per.append((len(pos), len(neg), ov))
            if per:
                w = np.array([min(a, b) for a, b, _ in per], dtype=float)
                o = np.array([x for _, _, x in per], dtype=float)
                mean_ov = float(np.average(o, weights=w)) if w.sum() > 0 else float(np.mean(o))
                min_ov = float(np.min(o))
                n_pairs = len(per)
            else:
                mean_ov = min_ov = float("nan")
                n_pairs = 0
            rows.append({
                "cutoff": c,
                "n_t1": int(len(t1)),
                "base_rate_t1": float(t1["y_t1"].mean()) if len(t1) else float("nan"),
                "mean_overlap_coefficient": mean_ov,
                "min_overlap_coefficient": min_ov,
                "n_presentations_compared": n_pairs,
                "zero_click_share_t1": float((t1["w1_clicks"] == 0).mean()) if len(t1) else float("nan"),
            })
        ov = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t03_overlap.csv"
        ov.to_csv(p, index=False)
        st.artifact(p, rows=len(ov), cols=ov.shape[1], message="overlap audit")
        st.metric("overlap", {
            "min_mean_overlap": float(ov["mean_overlap_coefficient"].min()),
            "max_mean_overlap": float(ov["mean_overlap_coefficient"].max()),
        })
        st.check("overlap_computed", ov["mean_overlap_coefficient"].notna().all(),
                 "overlap coefficient available for every cutoff")
        st.check("zero_click_recorded", ov["zero_click_share_t1"].notna().all(),
                 "zero-click share recorded; zero-click rows retained")

    with st.step("cohort_flow"):
        rows = []
        for c, f in frames.items():
            ar = f[f[f"in_at_risk_c{c}"] == 1]
            t1_ar = ar[ar["y_t1"].notna()]
            rows.append({
                "cutoff": c,
                "n_full": int(len(f)),
                "n_t1_full": int(f["y_t1"].notna().sum()),
                "base_rate_t1_full": float(f["y_t1"].mean()),
                "n_at_risk": int(len(ar)),
                "n_t1_at_risk": int(len(t1_ar)),
                "base_rate_t1_at_risk": float(t1_ar["y_t1"].mean()) if len(t1_ar) else float("nan"),
                "n_t2_full": int(len(f)),
                "base_rate_t2_full": float(f["y_t2"].mean()),
                "withdrawn_by_w1": int(((f["is_withdrawn"] == 1) &
                                       (f["date_unregistration"] <= 27)).sum()),
                "withdrawn_by_w2": int(((f["is_withdrawn"] == 1) &
                                       (f["date_unregistration"] > 27) &
                                       (f["date_unregistration"] <= 55)).sum()),
            })
        flow = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t03_cohort_flow.csv"
        flow.to_csv(p, index=False)
        st.artifact(p, rows=len(flow), cols=flow.shape[1], message="cohort flow")
        st.metric("cohort", {"n_full": int(len(master)),
                             "n_t1": int(master["y_t1"].notna().sum())})

    with st.step("withdrawal_timing"):
        w = master[master["is_withdrawn"] == 1].copy()
        buckets = pd.cut(w["date_unregistration"],
                         bins=[-np.inf, -0.5, 27.5, 55.5, np.inf],
                         labels=["pre_start", "W1_0_27", "W2_28_55", "after_55"])
        tab = buckets.value_counts().rename("n").reset_index()
        tab.columns = ["bucket", "n"]
        p = ctx.paths.tables_dir / "t03_withdrawal_timing.csv"
        tab.to_csv(p, index=False)
        st.artifact(p, rows=len(tab), cols=2, message="withdrawal timing")
        st.metric("withdrawal_timing", {str(k): int(v) for k, v in
                                        tab.set_index("bucket")["n"].items()})
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.hist(w["date_unregistration"].dropna(), bins=50, color="#3b6ea5",
                edgecolor="white", linewidth=0.3)
        for x, lab in [(0, "W1 start"), (27.5, "W1 end"), (55.5, "W2 end")]:
            ax.axvline(x, color="#b23b3b", ls="--", lw=0.9)
            ax.text(x, ax.get_ylim()[1] * 0.92, lab, rotation=90,
                    ha="right", va="top", fontsize=8, color="#b23b3b")
        ax.set_xlabel("date_unregistration (days from course start)")
        ax.set_ylabel("withdrawn enrolments")
        ax.set_title("Withdrawal timing (negative values = before start)")
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f03_withdrawal_timing.png"
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        st.artifact(fp, message="withdrawal timing figure")

    with st.step("click_figure"):
        f = frames[28] if 28 in frames else frames[cutoffs[-1]]
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        order = ["Distinction", "Pass", "Fail", "Withdrawn"]
        data = [np.log1p(f.loc[f["final_result"] == r, "w1_clicks"].values)
                for r in order if (f["final_result"] == r).any()]
        labels = [r for r in order if (f["final_result"] == r).any()]
        ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.55)
        ax.set_ylabel("log1p(W1 clicks, days 0-27)")
        ax.set_title("Early engagement overlaps across outcomes (W1)")
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f03_click_distribution_by_outcome.png"
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        st.artifact(fp, message="W1 clicks by outcome figure")
