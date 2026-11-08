"""S07 — outcome-blind clustering of post-deadline timing in the paired sample.

Exploratory only: the clusters describe timing profiles among students with
comparable post-deadline clicks, and the final outcome is never used to choose
k or features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import read_tma_features

CLUSTER_FEATURES = ["post_active_days", "post_longest_inactive_run",
                    "post_week2_share", "post_peak7_share", "post_entropy_norm"]


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    frame = read_tma_features(ctx)
    tolerance = float(cfg["study"]["tolerance"])
    pairs_path = ctx.paths.outputs_dir / f"pairs_{int(round(tolerance * 100)):02d}.parquet"
    if not pairs_path.exists():
        st.check("pairs_available", False, f"missing {pairs_path.name}")
        return
    pairs = pd.read_parquet(pairs_path)
    positive = pairs[pairs["fail_clicks"].gt(0)].reset_index(drop=True)
    st.metric("clustering_input", {"pairs": int(len(pairs)),
                                   "positive_pairs": int(len(positive))})
    if len(positive) < 30:
        # The rehearsal sample can be too small to cluster; record that and let
        # S12 report the absence instead of failing the run.
        st.check("enough_pairs", ctx.mode == "full",
                 f"positive pairs={len(positive)}; clustering needs at least 30",
                 warn_only=ctx.mode != "full")
        if ctx.mode != "full":
            st.warn("clustering skipped: too few positive-click pairs in the sample")
            for name in ("t07_cluster_candidates.csv", "t07_cluster_profile.csv"):
                pd.DataFrame().to_csv(ctx.paths.tables_dir / name, index=False)
            return

    max_pairs = int(cfg["clustering"].get("max_pairs", 800))
    rng = np.random.default_rng(seed)
    chosen = positive.iloc[np.sort(rng.choice(len(positive),
                                             size=min(max_pairs, len(positive)),
                                             replace=False))].reset_index(drop=True)
    n = len(chosen)
    rows = np.concatenate([chosen["fail_index"].to_numpy(), chosen["pass_index"].to_numpy()])
    sample = frame.iloc[rows].reset_index(drop=True).copy()
    sample["pair_side"] = np.where(np.arange(len(sample)) < n, "fail", "pass_distinction")
    sample["pair_id"] = np.concatenate([np.arange(n), np.arange(n)])
    missing = [c for c in CLUSTER_FEATURES if c not in sample]
    if missing:
        st.check("cluster_features_present", False, f"missing={missing}")
        return

    x = StandardScaler().fit_transform(sample[CLUSTER_FEATURES].to_numpy(dtype=float))
    k_lo, k_hi = cfg["clustering"]["k_range"]
    candidates, labels_by_k = [], {}
    for k in range(int(k_lo), int(k_hi) + 1):
        model = KMeans(n_clusters=k, n_init=int(cfg["clustering"]["n_init"]),
                       random_state=seed).fit(x)
        labels = model.labels_
        counts = np.bincount(labels)
        if counts.min() < max(5, int(np.ceil(0.03 * len(sample)))):
            continue
        stability = []
        for repeat in range(int(cfg["clustering"]["bootstrap"])):
            sub = np.sort(rng.choice(len(sample), size=int(0.8 * len(sample)), replace=False))
            rerun = KMeans(n_clusters=k, n_init=10, random_state=seed + repeat + 1) \
                .fit_predict(x[sub])
            stability.append(float(adjusted_rand_score(labels[sub], rerun)))
        candidates.append({"k": k,
                           "silhouette": float(silhouette_score(
                               x, labels, sample_size=min(800, len(sample)), random_state=seed)),
                           "mean_ari": float(np.mean(stability)),
                           "min_cluster_share": float(counts.min() / len(sample))})
        labels_by_k[k] = labels
    if not candidates:
        st.check("cluster_candidates", False, "no admissible k")
        return
    table = pd.DataFrame(candidates).sort_values("k").reset_index(drop=True)
    p = ctx.paths.tables_dir / "t07_cluster_candidates.csv"
    table.to_csv(p, index=False)
    st.artifact(p, rows=len(table), cols=table.shape[1], message="k candidates")

    best = max(candidates, key=lambda row: (row["silhouette"], row["mean_ari"]))
    labels = labels_by_k[best["k"]]
    sample["cluster"] = labels
    sample[["code_module", "code_presentation", "id_student", "pair_id", "pair_side",
            "cluster", "post_clicks", "final_result"]].to_parquet(
        ctx.paths.outputs_dir / "cluster_sample.parquet", index=False)

    profile = (sample.groupby("cluster")
               .agg(n=("id_student", "size"),
                    mean_post_clicks=("post_clicks", "mean"),
                    median_post_clicks=("post_clicks", "median"),
                    submitted_by_cutoff=("submitted_by_cutoff", "mean"),
                    fail_share=("final_result", lambda s: float(s.eq("Fail").mean())))
               .join(sample.groupby("cluster")[CLUSTER_FEATURES].mean())
               .reset_index())
    p = ctx.paths.tables_dir / "t07_cluster_profile.csv"
    profile.to_csv(p, index=False)
    st.artifact(p, rows=len(profile), cols=profile.shape[1], message="cluster profiles")

    # Membership contrast: the same pair should mostly land in one cluster.
    membership = sample.pivot_table(index="pair_id", columns="pair_side",
                                    values="cluster", aggfunc="first")
    same = float((membership["fail"] == membership["pass_distinction"]).mean()) \
        if {"fail", "pass_distinction"} <= set(membership.columns) else float("nan")
    st.metric("cluster_outcome", {"k": int(best["k"]),
                                  "silhouette": best["silhouette"],
                                  "mean_ari": best["mean_ari"],
                                  "pairs_in_same_cluster": same})
    st.check("cluster_stability_reported",
             bool(np.isfinite(best["mean_ari"])),
             f"mean ARI={best['mean_ari']:.3f} at k={best['k']}")
    st.check("clusters_exploratory", True,
             "k and features were chosen without the final outcome")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(sample["post_active_days"], sample["post_longest_inactive_run"],
               c=labels, cmap="viridis", s=10, alpha=0.75)
    ax.set_xlabel("post-deadline active days")
    ax.set_ylabel("longest inactive run (days)")
    ax.set_title(f"Outcome-blind timing clusters (k={best['k']}) in the paired sample")
    fig.tight_layout()
    fp = ctx.paths.figures_dir / "f07_clusters.png"
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    st.artifact(fp, message="cluster scatter")
