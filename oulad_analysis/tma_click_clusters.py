"""Outcome-blind post-TMA click-volume clusters and within-cluster pair comparisons.

Requires student-level artifacts from tma_two_week.py. Student cluster labels
remain in ignored outputs; only aggregate tables belong in public results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .assessment_node import KEY
from .matched import _component_intervals
from .tma_two_week import PRIMARY

FEATURES = PRIMARY + ["post_peak7_share", "post_entropy_norm",
                      "post_resource_share__forumng", "post_resource_share__oucontent",
                      "submitted_by_cutoff", "submitted_after_due", "pre_clicks", "pre_active_days"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_volume_clusters(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[int, np.ndarray], int]:
    """Fit K-means to positive log post-clicks only; keep zero clicks separate."""
    clicks = frame.post_clicks.to_numpy(dtype=float)
    if np.any(~np.isfinite(clicks)) or np.any(clicks < 0):
        raise ValueError("Post-click counts must be finite and nonnegative")
    positive = clicks > 0
    x = np.log1p(clicks[positive]).reshape(-1, 1)
    if len(x) < 10:
        raise ValueError("Too few positive-click observations")
    candidates, assignments = [], {}
    rng = np.random.default_rng(seed + 8100)
    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
        order = np.argsort(model.cluster_centers_.ravel())
        remap = np.empty(k, dtype=int)
        remap[order] = np.arange(k)
        labels = np.full(len(frame), -1, dtype=int)
        labels[positive] = remap[model.labels_]
        assignments[k] = labels
        stability = []
        for repeat in range(3):
            selected = rng.choice(len(x), size=int(.8 * len(x)), replace=False)
            refit = KMeans(n_clusters=k, n_init=10, random_state=seed + repeat + 1).fit(x[selected])
            stability.append(adjusted_rand_score(model.labels_, refit.predict(x)))
        candidates.append({"k": k,
                           "silhouette": float(silhouette_score(x, model.labels_, sample_size=min(3000, len(x)), random_state=seed)),
                           "stability_ari": float(np.mean(stability)),
                           "min_cluster_share": float(np.bincount(model.labels_).min() / len(x)),
                           "centers_clicks_ascending": ";".join(f"{np.expm1(model.cluster_centers_[j, 0]):.2f}" for j in order)})
    choices = pd.DataFrame(candidates)
    best_k = int(choices.sort_values(["silhouette", "stability_ari"], ascending=False).iloc[0].k)
    return choices, assignments, best_k


def _cluster_name(label: int, k: int) -> str:
    if label == -1:
        return "zero"
    if k == 2:
        return ["low", "high"][label]
    if k == 3:
        return ["low", "medium", "high"][label]
    return f"volume_{label + 1}"


def _same_cluster_pairs(frame: pd.DataFrame, pairs: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, int]:
    if pairs.empty or pairs.fail_index.min() < 0 or pairs.pass_index.min() < 0:
        raise ValueError("Expected nonempty, valid 5% post-click pairs")
    if max(pairs.fail_index.max(), pairs.pass_index.max()) >= len(frame):
        raise ValueError("Pair indices do not fit the feature table")
    fail = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    if not fail.final_result.eq("Fail").all() or not passing.final_result.isin(["Pass", "Distinction"]).all():
        raise ValueError("Pair outcomes disagree with the feature table")
    if not fail[KEY[:2]].reset_index(drop=True).equals(passing[KEY[:2]].reset_index(drop=True)):
        raise ValueError("Pair members are in different presentations")
    if not np.allclose(fail.post_clicks, pairs.fail_clicks) or not np.allclose(passing.post_clicks, pairs.pass_clicks):
        raise ValueError("Pair clicks disagree with the feature table")
    if pairs.relative_gap.gt(.05 + 1e-12).any():
        raise ValueError("Expected 5% post-click matching")
    result = pairs.assign(cluster=labels[pairs.fail_index.to_numpy()],
                          pass_cluster=labels[pairs.pass_index.to_numpy()])
    cross = int(result.cluster.ne(result.pass_cluster).sum())
    return result[result.cluster.eq(result.pass_cluster)].drop(columns="pass_cluster").reset_index(drop=True), cross


def summarize_volume_clusters(frame: pd.DataFrame, pairs: pd.DataFrame, labels: np.ndarray,
                              k: int, seed: int, repeats: int,
                              detailed: bool = True) -> dict[str, pd.DataFrame | int]:
    """Keep within-presentation post-click pairs inside the same volume cluster."""
    if frame.duplicated(KEY).any() or not frame.final_result.isin(["Fail", "Pass", "Distinction"]).all():
        raise ValueError("Expected unique binary node records")
    if any(name not in frame for name in FEATURES):
        raise ValueError("Missing two-week behavior features")
    if len(labels) != len(frame):
        raise ValueError("Cluster assignments have wrong length")
    matched, cross = _same_cluster_pairs(frame, pairs, labels)
    names = {i: _cluster_name(i, k) for i in [-1] + list(range(k))}
    population = []
    for label, group in frame.assign(cluster=labels).groupby("cluster"):
        population.append({"cluster": names[label], "sample_records": len(group),
                           "fail_records": int(group.final_result.eq("Fail").sum()),
                           "fail_fraction_unmatched": float(group.final_result.eq("Fail").mean()),
                           "min_post_clicks": float(group.post_clicks.min()),
                           "max_post_clicks": float(group.post_clicks.max()),
                           "mean_post_clicks": float(group.post_clicks.mean())})
    balance, effects, pre_sensitivity, presentations = [], [], [], []
    for label, group in matched.groupby("cluster"):
        fail = frame.iloc[group.fail_index.to_numpy()].reset_index(drop=True)
        passing = frame.iloc[group.pass_index.to_numpy()].reset_index(drop=True)
        diff = fail[FEATURES].to_numpy(dtype=float) - passing[FEATURES].to_numpy(dtype=float)
        intervals = _component_intervals(group, diff, seed, repeats)
        pre_log_fail, pre_log_pass = np.log1p(fail.pre_clicks), np.log1p(passing.pre_clicks)
        sd = np.sqrt((pre_log_fail.var(ddof=0) + pre_log_pass.var(ddof=0)) / 2)
        balance.append({"cluster": names[label], "pairs": len(group),
                        "mean_post_clicks_fail": float(fail.post_clicks.mean()),
                        "mean_post_clicks_pass": float(passing.post_clicks.mean()),
                        "mean_pre_clicks_fail": float(fail.pre_clicks.mean()),
                        "mean_pre_clicks_pass": float(passing.pre_clicks.mean()),
                        "pre_log_click_smd": float((pre_log_fail.mean() - pre_log_pass.mean()) / sd) if sd else 0.0})
        for j, name in enumerate(FEATURES):
            effects.append({"cluster": names[label], "feature": name, "pairs": len(group),
                            "fail_mean": float(fail[name].mean()), "pass_mean": float(passing[name].mean()),
                            "fail_minus_pass": float(diff[:, j].mean()),
                            "ci_low": float(intervals[0, j]), "ci_high": float(intervals[1, j])})
        if not detailed:
            continue
        pre_fail, pre_pass = fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy()
        ratio = np.maximum(pre_fail, pre_pass) / np.maximum(np.minimum(pre_fail, pre_pass), 1)
        near = ((pre_fail == 0) & (pre_pass == 0)) | ((np.minimum(pre_fail, pre_pass) > 0) & (ratio <= 1.2 + 1e-12))
        if near.any():
            near_pairs = group.loc[near].reset_index(drop=True)
            near_diff = diff[near, :len(PRIMARY)]
            near_intervals = _component_intervals(near_pairs, near_diff, seed, repeats)
            for j, name in enumerate(PRIMARY):
                pre_sensitivity.append({"cluster": names[label], "pre_click_tolerance": .20,
                                        "pairs": len(near_pairs), "fraction_of_cluster_pairs": len(near_pairs) / len(group),
                                        "feature": name, "fail_minus_pass": float(near_diff[:, j].mean()),
                                        "ci_low": float(near_intervals[0, j]), "ci_high": float(near_intervals[1, j])})
        for (module, presentation), indices in group.groupby(KEY[:2]).indices.items():
            if len(indices) < 20:
                continue
            for j, name in enumerate(PRIMARY):
                presentations.append({"cluster": names[label], "code_module": module,
                                      "code_presentation": presentation, "pairs": len(indices),
                                      "feature": name, "fail_minus_pass": float(diff[indices, j].mean())})
    return {"population": pd.DataFrame(population), "balance": pd.DataFrame(balance),
            "effects": pd.DataFrame(effects), "pre_sensitivity": pd.DataFrame(pre_sensitivity),
            "presentations": pd.DataFrame(presentations), "cross_cluster_pairs": cross}


def plot_main_effects(effects: pd.DataFrame, output_path: Path) -> None:
    """Plot the four primary paired contrasts with their pointwise intervals."""
    import matplotlib.pyplot as plt

    names = {"post_active_days": ("Active days", "days", 1),
             "post_both_weeks": ("Active in both weeks", "percentage points", 100),
             "post_longest_inactive_run": ("Longest inactive run", "days", 1),
             "post_week2_share": ("Week 2 click share", "percentage points", 100)}
    colors = {"low": "#3856B3", "high": "#B87916"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (feature, (title, unit, scale)) in zip(axes.flat, names.items()):
        subset = effects.set_index(["cluster", "feature"])
        for y, cluster in enumerate(("low", "high")):
            row = subset.loc[(cluster, feature)]
            value = float(row.fail_minus_pass) * scale
            lower, upper = float(row.ci_low) * scale, float(row.ci_high) * scale
            ax.errorbar(value, y, xerr=[[value - lower], [upper - value]], fmt="o",
                        color=colors[cluster], markersize=7, capsize=3, linewidth=2)
        ax.axvline(0, color="#6E7480", linewidth=1)
        ax.set_yticks([0, 1], ["Low clicks", "High clicks"])
        ax.set_ylim(1.5, -.5)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel(f"Fail − Pass/Distinction ({unit})", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#E5E8EE", linewidth=.8)
        ax.set_axisbelow(True)
    fig.suptitle("Behavior differences within post-TMA click-volume clusters",
                 x=.06, ha="left", fontsize=15, weight="bold")
    indexed = effects.set_index(["cluster", "feature"])
    low_n = int(indexed.loc[("low", "post_active_days")].pairs)
    high_n = int(indexed.loc[("high", "post_active_days")].pairs)
    fig.text(.06, .035,
             f"Same course/presentation; ≤5% post-click pairs. Low n={low_n:,}; high n={high_n:,}.\n"
             "Pointwise 95% student-component bootstrap; check pre-TMA engagement balance separately.",
             fontsize=9, color="#4A5260")
    fig.tight_layout(rect=(0, .09, 1, .94), h_pad=2.3, w_pad=2)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def run(input_dir: Path, output_dir: Path, seed: int = 42, repeats: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    feature_path = input_dir / "features.parquet"
    pair_path = input_dir / "post_pairs_05.parquet"
    frame = pd.read_parquet(feature_path)
    pairs = pd.read_parquet(pair_path)
    candidates, assignments, best_k = fit_volume_clusters(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame[KEY].assign(volume_cluster=assignments[best_k]).to_parquet(
        output_dir / "sample_volume_clusters.parquet", index=False)
    candidates.to_csv(output_dir / "volume_cluster_candidates.csv", index=False)
    main = summarize_volume_clusters(frame, pairs, assignments[best_k], best_k, seed, repeats)
    for name in ("population", "balance", "effects", "pre_sensitivity", "presentations"):
        main[name].to_csv(output_dir / f"volume_cluster_{name}.csv", index=False)
    if best_k == 2:
        plot_main_effects(main["effects"], output_dir / "volume_cluster_effects.png")
    if best_k != 3:
        sensitivity = summarize_volume_clusters(frame, pairs, assignments[3], 3, seed, repeats, detailed=False)
        sensitivity["effects"].loc[lambda table: table.feature.isin(PRIMARY)].to_csv(
            output_dir / "three_cluster_sensitivity.csv", index=False)
    record = {"question": "post-TMA behavior within outcome-blind post-click volume clusters",
              "population": "binary two-week cohort; zero post clicks separate; K-means on log1p positive post clicks",
              "cluster_selection": "k=2..5, highest sampled silhouette; stability check; no outcome labels",
              "pair_comparison": "within same module/presentation, 5% post-click matched pairs, same volume cluster",
              "best_positive_k": best_k, "cross_cluster_pairs_excluded": main["cross_cluster_pairs"],
              "seed": seed, "pair_bootstrap": repeats,
              "input_sha256": {"features.parquet": _sha256(feature_path),
                               "post_pairs_05.parquet": _sha256(pair_path),
                               "two_week_protocol.json": _sha256(input_dir / "protocol.json")},
              "code_sha256": _sha256(Path(__file__)),
              "limitations": "exploratory volume bands; outcome-defined pairs; pre-TMA imbalance; no causal inference"}
    (output_dir / "volume_cluster_protocol.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.seed, args.repeats)


if __name__ == "__main__":
    main()
