"""S07 — W1 behavioural profiles, k selection and bootstrap stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

from src.clustering import (consensus, pca_reduce, per_presentation_z,
                            stability)
from src.data import KEY, read_features, read_master
from src.features import feature_columns


def run(ctx, st):
    cfg = ctx.cfg
    master = read_master(ctx)
    f = read_features(ctx, 28)
    cols = feature_columns(f, 1, 28)
    df = f.merge(master[KEY + ["final_result", "y_t1"]], on=KEY, how="inner")
    n_all = len(df)
    min_days = int(cfg["features"].get("min_active_days_for_clustering", 1))
    df = df[df["w1_active_days"] >= min_days].reset_index(drop=True)
    st.metric("clustering_cohort", {
        "n_all": int(n_all), "n_clustered": int(len(df)),
        "excluded_low_activity": int(n_all - len(df)),
        "min_active_days": min_days, "n_features": len(cols),
    })
    st.check("enough_rows", len(df) >= 30, f"n={len(df)}")
    if len(df) < 30:
        return

    Xdf = per_presentation_z(df, cols, ["code_module", "code_presentation"])
    Xraw = Xdf[cols].to_numpy(dtype=float)
    X, evr = pca_reduce(Xraw, max_components=8, seed=cfg["seed"])
    st.metric("pca", {"n_components": int(X.shape[1]),
                      "explained_variance_ratio": [round(x, 4) for x in evr],
                      "cumulative": round(float(np.sum(evr)), 4)})

    k_lo, k_hi = cfg["clustering"]["k_range"]
    k_list = list(range(int(k_lo), int(k_hi) + 1))
    with st.step("stability"):
        tab, base, aris = stability(
            X, k_list, cfg["clustering"]["n_init"],
            cfg["clustering"]["bootstrap"], cfg["seed"])
        p = ctx.paths.tables_dir / "t07_stability.csv"
        tab.to_csv(p, index=False)
        st.artifact(p, rows=len(tab), cols=tab.shape[1], message="stability table")
        st.check("stability_all_k", len(tab) == len(k_list),
                 f"k rows={len(tab)} expected={len(k_list)}")
        st.check("stability_metrics_recorded", tab["mean_ari"].notna().all(),
                 "mean ARI recorded for every k")

    with st.step("select_k"):
        thr = float(cfg["clustering"]["ari_threshold"])
        ok = tab[tab["mean_ari"] >= thr]
        if len(ok):
            chosen = ok.sort_values(["mean_ari", "silhouette"],
                                    ascending=[False, False]).iloc[0]
            reason = (f"k={int(chosen.k)} selected from k with mean ARI >= {thr}; "
                      f"maximum stability first, silhouette as tie-break")
            stable = True
        else:
            chosen = tab.sort_values("mean_ari", ascending=False).iloc[0]
            reason = (f"no k reached mean ARI {thr}; k={int(chosen.k)} chosen by "
                      "maximum mean ARI and flagged unstable")
            stable = False
        k = int(chosen["k"])
        labels = base[k]
        st.notes = reason
        st.extra["k_reason"] = reason
        st.metric("k_selection", {
            "k_chosen": k, "mean_ari": float(chosen["mean_ari"]),
            "min_cluster_jaccard": float(chosen["min_cluster_jaccard"]),
            "silhouette": float(chosen["silhouette"]),
            "ari_threshold": thr, "stable": stable,
            "cluster_sizes": [int(x) for x in np.bincount(labels)],
        })
        st.check("k_selection_recorded", True, reason)

    with st.step("cluster_outcome"):
        df["cluster"] = labels
        ct = pd.crosstab(df["cluster"], df["final_result"])
        prop = ct.div(ct.sum(axis=1), axis=0)
        out = ct.copy()
        out.columns = [f"n_{c}" for c in out.columns]
        for c in prop.columns:
            out[f"share_{c}"] = prop[c]
        out["n"] = ct.sum(axis=1)
        out = out.reset_index()
        p = ctx.paths.tables_dir / "t07_cluster_outcome.csv"
        out.to_csv(p, index=False)
        st.artifact(p, rows=len(out), cols=out.shape[1],
                    message="cluster x outcome")
        st.metric("outcome_rates_by_cluster",
                  {str(r["cluster"]): {c: round(float(r[f"share_{c}"]), 4)
                                       for c in prop.columns}
                   for _, r in out.iterrows()})
        st.check("cluster_outcome_counts", int(ct.values.sum()) == len(df),
                 f"crosstab total={int(ct.values.sum())}")

    with st.step("figures"):
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        data = [aris[int(kk)] for kk in k_list]
        ax.boxplot(data, tick_labels=[str(kk) for kk in k_list],
                   showfliers=False)
        ax.axhline(cfg["clustering"]["ari_threshold"], color="#b23b3b",
                   ls="--", lw=1, label=f"ARI threshold={cfg['clustering']['ari_threshold']}")
        ax.set_xlabel("k"); ax.set_ylabel("bootstrap ARI")
        ax.set_title("Cluster stability across k")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f07_ari_distribution.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="ARI distribution")

        vis, cm = consensus(X, k, cfg["clustering"]["n_init"],
                            cfg["clustering"]["bootstrap"], cfg["seed"],
                            max_vis=2000)
        order = np.argsort(labels[vis], kind="stable")
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        im = ax.imshow(cm[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"Consensus matrix (k={k})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f07_consensus_matrix.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="consensus matrix")

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        order_cols = [c for c in ["Distinction", "Pass", "Fail", "Withdrawn"]
                      if c in prop.columns]
        bottom = np.zeros(len(prop))
        colors = {"Distinction": "#2f6f4e", "Pass": "#6ea36f",
                  "Fail": "#c05b5b", "Withdrawn": "#8a8a8a"}
        for c in order_cols:
            vals = prop[c].to_numpy()
            ax.bar([str(i) for i in prop.index], vals, bottom=bottom,
                   color=colors.get(c, "#999"), label=c)
            bottom += vals
        ax.set_xlabel("cluster"); ax.set_ylabel("share of cluster")
        ax.set_title("Outcome mix by W1 behavioural profile")
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f07_cluster_outcome.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="cluster x outcome figure")

        # PCA diagnostic and silhouette distribution (chosen k).
        rng = np.random.default_rng(cfg["seed"])
        idx = np.arange(len(X))
        if len(idx) > 5000:
            idx = rng.choice(idx, 5000, replace=False)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for cl in sorted(np.unique(labels)):
            m = labels[idx] == cl
            ax.scatter(X[idx][m, 0], X[idx][m, 1], s=4, alpha=0.18,
                       label=f"cluster {cl}")
        ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}%)")
        ax.set_title(f"PCA diagnostic (8 PCs, cumulative {100*sum(evr):.1f}%)")
        ax.legend(frameon=False, fontsize=7, markerscale=3)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f07_pca_scatter.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="PCA scatter")

        sil = silhouette_samples(X[idx], labels[idx])
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.hist(sil, bins=40, color="#3b6ea5", edgecolor="white", linewidth=0.3)
        ax.axvline(float(sil.mean()), color="#b23b3b", ls="--", lw=1,
                   label=f"mean={sil.mean():.3f}")
        ax.set_xlabel("silhouette (sample)")
        ax.set_ylabel("count")
        ax.set_title(f"Silhouette distribution, k={k}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f07_silhouette.png"
        fig.savefig(fp, dpi=160); plt.close(fig)
        st.artifact(fp, message="silhouette distribution")

    with st.step("unstable_clusters"):
        minj = float(chosen["min_cluster_jaccard"])
        if minj < 0.5:
            st.check("unstable_cluster_decision", True,
                     f"min cluster Jaccard={minj:.3f} < 0.5; cluster retained "
                     "but flagged as unstable and conclusions downgraded")
        else:
            st.check("cluster_jaccard", minj >= 0.5,
                     f"min cluster Jaccard={minj:.3f}")
