"""Outcome-blind behavioral clustering on a bounded reproducible sample."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_behaviors(frame: pd.DataFrame, output_dir: Path, seed: int = 42, max_rows: int = 2500) -> None:
    population = frame[frame.total_clicks > 0].copy()
    if len(population) < 30:
        (output_dir / "cluster_note.txt").write_text("Fewer than 30 nonzero-click samples; clustering skipped.\n", encoding="utf-8")
        return
    sample = population.sample(n=min(max_rows, len(population)), random_state=seed).reset_index(drop=True)
    weekly = [f"week_{i}_share" for i in range(4)]
    resources = sorted(c for c in sample if c.startswith("resource_share__"))
    features = weekly + resources + ["active_days", "longest_inactive_run", "daily_concentration", "resource_entropy"]
    x = StandardScaler().fit_transform(sample[features].fillna(0))
    candidates = []
    fitted = {}
    rng = np.random.default_rng(seed)
    for algorithm in ("kmeans", "agglomerative"):
        for k in range(2, min(7, len(sample) // 10 + 1)):
            make = (lambda s: KMeans(n_clusters=k, n_init=10, random_state=s)) if algorithm == "kmeans" else (lambda s: AgglomerativeClustering(n_clusters=k))
            labels = make(seed).fit_predict(x)
            counts = np.bincount(labels)
            if counts.min() < max(5, len(sample) * 0.03):
                continue
            silhouette = float(silhouette_score(x, labels, sample_size=min(1000, len(x)), random_state=seed))
            stability = []
            for repeat in range(3):
                indices = np.sort(rng.choice(len(x), size=int(0.8 * len(x)), replace=False))
                rerun = make(seed + repeat + 1).fit_predict(x[indices])
                stability.append(float(adjusted_rand_score(labels[indices], rerun)))
            candidates.append({"algorithm": algorithm, "k": k, "silhouette": silhouette,
                               "stability_ari": float(np.mean(stability)), "min_cluster_share": float(counts.min() / len(x))})
            fitted[(algorithm, k)] = labels
    pd.DataFrame(candidates).to_csv(output_dir / "cluster_candidates.csv", index=False)
    if not candidates:
        (output_dir / "cluster_note.txt").write_text("No candidate passed the minimum cluster size rule.\n", encoding="utf-8")
        return
    best = max(candidates, key=lambda candidate: (candidate["silhouette"], candidate["stability_ari"]))
    sample["cluster"] = fitted[(best["algorithm"], best["k"])]
    sample[["code_module", "code_presentation", "id_student", "cluster", "final_result", "total_clicks"]].to_parquet(output_dir / "cluster_sample.parquet", index=False)
    profile = sample.groupby("cluster")[features + ["total_clicks"]].mean()
    profile.insert(0, "n", sample.groupby("cluster").size())
    profile.to_csv(output_dir / "cluster_profile.csv")
    outcomes = pd.crosstab(sample.cluster, sample.final_result, normalize="index")
    outcomes.to_csv(output_dir / "cluster_outcomes.csv")
    (output_dir / "cluster_choice.json").write_text(json.dumps({"selection": best, "sample_rows": len(sample), "population_rows": len(population),
                                                                  "selection_uses_outcome": False}, indent=2), encoding="utf-8")
    ax = profile[weekly].T.plot(marker="o", figsize=(7, 4))
    ax.set(xlabel="Week (0–3)", ylabel="Share of first-four-week clicks", title="Weekly activity by behavior cluster")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "cluster_weekly.png", dpi=160)
    plt.close(ax.figure)
    if resources:
        ax = profile[resources].plot(kind="bar", stacked=True, figsize=(9, 4))
        ax.set(xlabel="Cluster", ylabel="Resource click share", title="Resource composition by cluster")
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / "cluster_resources.png", dpi=160)
        plt.close(ax.figure)
    ax = outcomes.plot(kind="bar", stacked=True, figsize=(7, 4))
    ax.set(xlabel="Cluster", ylabel="Share", title="Final outcomes by cluster")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "cluster_outcomes.png", dpi=160)
    plt.close(ax.figure)
