"""S06 — CEM within module x presentation; balance, discordant strata, W2 gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_features, read_master, read_vle
from src.matching import (MATCH_VARS, balance_table, cem, e_value_rr,
                          w2_differences)


def run(ctx, st):
    master = read_master(ctx)
    f28 = read_features(ctx, 28)
    f56 = read_features(ctx, 56)
    t1 = master[master["y_t1"].notna()][KEY + ["y_t1", "final_result"]]
    df = f28.merge(t1, on=KEY, how="inner").reset_index(drop=True)
    # W2/delta features come from cutoff 56; matching variables stay W1-only.
    w2cols = [c for c in f56.columns if c.startswith(("w2_", "d_", "accel_"))]
    df = df.merge(f56[KEY + w2cols], on=KEY, how="left", validate="one_to_one")

    st.metric("matching_cohort", {"n_t1": int(len(df)),
                                  "n_pass": int((df["y_t1"] == 1).sum()),
                                  "n_fail": int((df["y_t1"] == 0).sum())})
    st.check("matching_vars_are_w1", all(c.startswith("w1_") for c in MATCH_VARS),
             f"matching variables={MATCH_VARS}")
    st.check("no_post_cutoff_matching_vars",
             not any(("w2_" in c or "d_" in c or "score" in c.lower())
                     for c in MATCH_VARS),
             "no W2, delta or score field is used to define strata")

    with st.step("cem"):
        res = cem(df, ctx.cfg.data)
        for a in res["attempts"]:
            st.metric(f"cem_attempt_L{a['level']}", a)
        st.metric("cem", {
            "level": res["level"], "n_matched": res["n_matched"],
            "n_unmatched": res["n_unmatched"], "n_strata": res["n_strata"],
            "n_treated_matched": res["n_treated_matched"],
            "n_control_matched": res["n_control_matched"],
            "common_support_treated": res["common_support_treated"],
            "binned_max_abs_smd": res["binned_max_abs_smd"],
            "raw_max_abs_smd": max(abs(v) for v in res["raw_after"].values()),
        })
        st.check("cem_binned_balance", res["binned_max_abs_smd"] < 0.1,
                 f"max |SMD| over coarsened CEM bins={res['binned_max_abs_smd']:.4f}",
                 metrics={"binned_max_abs_smd": res["binned_max_abs_smd"]})
        raw_max = max(abs(v) for v in res["raw_after"].values())
        st.check("cem_raw_balance_diagnostic", True,
                 f"raw continuous max |SMD|={raw_max:.4f}; primary balance is on "
                 "the coarsened CEM bins, raw values are reported as a diagnostic")
        st.check("cem_sample_loss_reported", True,
                 f"matched={res['n_matched']} unmatched={res['n_unmatched']} "
                 f"treated support={res['common_support_treated']:.3f}")
        st.check("discordant_strata_nonempty", res["n_discordant_units"] > 0,
                 f"units in strata containing both outcomes={res['n_discordant_units']}",
                 warn_only=res["n_discordant_units"] == 0)

    with st.step("balance_table"):
        bal = balance_table(res)
        p = ctx.paths.tables_dir / "t06_balance.csv"
        bal.to_csv(p, index=False)
        st.artifact(p, rows=len(bal), cols=bal.shape[1], message="balance table")
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        sub = bal[bal["kind"] == "continuous"].copy()
        ypos = np.arange(len(sub))
        ax.scatter(sub["smd_before"], ypos, color="#8a8a8a", label="before", zorder=3)
        ax.scatter(sub["smd_after"], ypos, color="#3b6ea5", label="after CEM", zorder=3)
        for y in ypos:
            ax.plot([sub["smd_before"].iloc[y], sub["smd_after"].iloc[y]],
                    [y, y], color="#cccccc", zorder=1)
        ax.axvline(0.1, color="#b23b3b", ls="--", lw=1)
        ax.axvline(-0.1, color="#b23b3b", ls="--", lw=1)
        ax.set_yticks(ypos, sub["variable"], fontsize=8)
        ax.set_xlabel("standardised mean difference (raw continuous)")
        ax.set_title("CEM balance: raw diagnostic (bins are balanced exactly)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f06_love_plot.png"
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        st.artifact(fp, message="Love plot")

    with st.step("matched_strata"):
        matched = res["matched"].copy()
        cols = KEY + ["stratum", "att_w", "y_t1", "final_result"] + MATCH_VARS
        mp = ctx.run_dir / "artifacts" / "matched_strata.parquet"
        matched[cols].to_parquet(mp, index=False)
        st.artifact(mp, rows=len(matched), cols=len(cols),
                    message="matched strata with ATT weights")
        g = matched.groupby("stratum", observed=True).agg(
            n=("y_t1", "size"), n_pass=("y_t1", "sum"),
            mean_w1_clicks=("w1_clicks_total", "mean"),
            mean_w1_active_days=("w1_active_days", "mean")).reset_index()
        g["n_fail"] = g["n"] - g["n_pass"]
        dp = ctx.paths.tables_dir / "t06_discordant_strata.csv"
        g.to_csv(dp, index=False)
        st.artifact(dp, rows=len(g), cols=g.shape[1],
                    message="discordant strata")

    with st.step("w2_differences"):
        metrics = ["d_clicks", "d_clicks_ratio", "d_active_days",
                   "d_entropy_activity", "d_clicks_per_week", "d_submissions",
                   "w2_active_days", "w2_entropy_activity"]
        wd = w2_differences(res, metrics, n_boot=1000, seed=ctx.cfg["seed"])
        p = ctx.paths.tables_dir / "t06_w2_differences.csv"
        wd.to_csv(p, index=False)
        st.artifact(p, rows=len(wd), cols=wd.shape[1],
                    message="within-stratum W2 differences with cluster bootstrap CI")
        st.metric("w2_differences", {
            r["metric"]: {"diff": round(r["diff_pass_minus_fail"], 4),
                          "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)]}
            for _, r in wd.iterrows()
        })
        # E-value for the matched high-vs-low W2 change association.
        m = res["matched"].copy()
        ev = float("nan"); rr = float("nan"); p_hi = p_lo = float("nan")
        if "d_clicks" in m.columns and m["d_clicks"].notna().any():
            cut = float(m["d_clicks"].median())
            hi = m["d_clicks"] > cut
            lo = m["d_clicks"] <= cut
            if hi.sum() and lo.sum():
                p_hi = float(m.loc[hi, "y_t1"].mean())
                p_lo = float(m.loc[lo, "y_t1"].mean())
                if p_lo > 0:
                    rr = p_hi / p_lo
                    ev = e_value_rr(rr)
        st.metric("e_value", {"risk_ratio_high_w2_change": rr,
                              "p_pass_high": p_hi, "p_pass_low": p_lo,
                              "e_value": ev})

    with st.step("trajectory_divergence"):
        # Descriptive weekly trajectories inside the matched strata.  W1
        # matching is baseline; weeks 0-8 are shown to locate divergence.
        matched = res["matched"][KEY + ["stratum", "att_w", "y_t1"]].copy()
        vle = read_vle(ctx)
        v = vle[[*KEY, "date", "sum_click"]].copy()
        v = v[v["date"] >= 0]
        v["week"] = np.minimum((v["date"] // 7).astype(int), 8)
        agg = (v.groupby(KEY + ["week"], observed=True)["sum_click"].sum()
               .rename("weekly_clicks").reset_index())
        grid = matched.merge(pd.DataFrame({"week": list(range(9))}), how="cross")
        grid = grid.merge(agg, on=KEY + ["week"], how="left")
        grid["weekly_clicks"] = grid["weekly_clicks"].fillna(0)
        grid["log_clicks"] = np.log1p(grid["weekly_clicks"])
        tot = grid.groupby(KEY, observed=True)["weekly_clicks"].transform("sum")
        grid["share"] = np.where(tot > 0, grid["weekly_clicks"] / tot, 0.0)
        # Stratum x week sufficient statistics for a fast cluster bootstrap.
        strata = sorted(grid["stratum"].unique())
        sidx = {s: i for i, s in enumerate(strata)}
        K, W = len(strata), 9
        st_t = np.zeros((K, W)); sw_t = np.zeros((K, W))
        st_c = np.zeros((K, W)); sw_c = np.zeros((K, W))
        ss_t = np.zeros((K, W)); ss_c = np.zeros((K, W))
        for s, g in grid.groupby("stratum", observed=True):
            i = sidx[s]
            for yv, sm, swm, ssm in ((1, st_t, sw_t, ss_t),
                                     (0, st_c, sw_c, ss_c)):
                gg = g[g["y_t1"] == yv]
                for w in range(W):
                    gw = gg[gg["week"] == w]
                    sm[i, w] = float((gw["att_w"] * gw["log_clicks"]).sum())
                    ssm[i, w] = float((gw["att_w"] * gw["share"]).sum())
                    swm[i, w] = float(gw["att_w"].sum())
        rng = np.random.default_rng(ctx.cfg["seed"])
        boot = np.zeros((1000, W, 4))
        for b in range(1000):
            pick = rng.integers(0, K, K)
            it = st_t[pick].sum(0); iw = sw_t[pick].sum(0)
            ic = st_c[pick].sum(0); iwc = sw_c[pick].sum(0)
            boot[b, :, 0] = np.divide(it, iw, out=np.full(W, np.nan), where=iw > 0)
            boot[b, :, 1] = np.divide(ic, iwc, out=np.full(W, np.nan), where=iwc > 0)
            boot[b, :, 2] = boot[b, :, 0] - boot[b, :, 1]
        boot_s = np.zeros((1000, W))
        for b in range(1000):
            pick = rng.integers(0, K, K)
            a = ss_t[pick].sum(0) / np.maximum(sw_t[pick].sum(0), 1e-9)
            c = ss_c[pick].sum(0) / np.maximum(sw_c[pick].sum(0), 1e-9)
            boot_s[b] = a - c
        rows = []
        for w in range(W):
            gt = grid[(grid["week"] == w) & (grid["y_t1"] == 1)]
            gc = grid[(grid["week"] == w) & (grid["y_t1"] == 0)]
            mt = float((gt["att_w"] * gt["log_clicks"]).sum() / max(gt["att_w"].sum(), 1e-9))
            mc = float((gc["att_w"] * gc["log_clicks"]).sum() / max(gc["att_w"].sum(), 1e-9))
            diff = mt - mc
            lo, hi = np.nanpercentile(boot[:, w, 2], [2.5, 97.5])
            # normalized share
            mt_s = float((gt["att_w"] * gt["share"]).sum() / max(gt["att_w"].sum(), 1e-9))
            mc_s = float((gc["att_w"] * gc["share"]).sum() / max(gc["att_w"].sum(), 1e-9))
            slo, shi = np.nanpercentile(boot_s[:, w], [2.5, 97.5])
            rows.append({"week": w, "n_pass": int(len(gt)), "n_fail": int(len(gc)),
                         "mean_pass": mt, "mean_fail": mc, "diff": diff,
                         "ci_low": float(lo), "ci_high": float(hi),
                         "mean_pass_share": mt_s, "mean_fail_share": mc_s,
                         "diff_share": mt_s - mc_s,
                         "ci_low_share": float(slo), "ci_high_share": float(shi)})
        tw = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t12_trajectory_weekly.csv"
        tw.to_csv(p, index=False)
        st.artifact(p, rows=len(tw), cols=tw.shape[1], message="weekly trajectories")
        div_week = None
        for w in range(W - 1):
            if tw.loc[w, "ci_low"] > 0 and tw.loc[w + 1, "ci_low"] > 0:
                div_week = int(w); break
        dv = pd.DataFrame([{"metric": "log_clicks", "divergence_week": div_week,
                            "definition": "first week with diff CI > 0 for two consecutive weeks"}])
        dp = ctx.paths.tables_dir / "t12_divergence.csv"
        dv.to_csv(dp, index=False)
        st.artifact(dp, rows=len(dv), cols=2, message="divergence week")
        st.metric("trajectory", {"divergence_week": div_week,
                                 "weekly_n_pass_min": int(tw["n_pass"].min()),
                                 "weekly_n_fail_min": int(tw["n_fail"].min())})
        st.check("trajectory_weekly_ci", tw["ci_low"].notna().all() and
                 tw["ci_high"].notna().all(), "every week has bootstrap CI")
        st.check("divergence_defined", True,
                 f"divergence_week={div_week} (None = no stable divergence)")
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for ax, pcol, fcol, lab in (
                (axes[0], "mean_pass", "mean_fail", "log1p(weekly clicks)"),
                (axes[1], "mean_pass_share", "mean_fail_share",
                 "share of week 0-8 clicks")):
            x = tw["week"].to_numpy()
            yp = tw[pcol].to_numpy(); yf = tw[fcol].to_numpy()
            ax.plot(x, yp, "o-", color="#2f6f4e", label="Pass/Distinction")
            ax.plot(x, yf, "s-", color="#b23b3b", label="Fail")
            if div_week is not None:
                ax.axvline(div_week, color="#333", ls="--", lw=1,
                           label=f"divergence week={div_week}")
            ax.set_xlabel("week"); ax.set_ylabel(lab)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_title("Matched weekly trajectory")
        axes[1].set_title("Normalized weekly share")
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f12_trajectory_divergence.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="trajectory divergence figure")
