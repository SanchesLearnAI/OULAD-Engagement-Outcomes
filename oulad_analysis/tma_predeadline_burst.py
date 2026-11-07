"""Explore short pre-deadline VLE click bursts around the first weighted TMA.

All burst windows end at d-1; d itself is reported separately because the
dataset has day, not time-of-day, granularity. Models observe through d+14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import KEY, _fit_predict
from .matched import _component_intervals
from .tma_score_probe import make_joint_pairs
from .tma_submission_timing import VERSIONS as TIMING_VERSIONS, add_submission_distance
from .tma_two_week import _sql_path

BURST = ["pre_last1_rate_lift", "pre_last3_rate_lift"]
INTERACTION = ["pre_last1_lift_x_late", "pre_last3_lift_x_late"]
PRE_ONLY = ["module_presentation", "log_pre_before_due_clicks", "pre_before_due_active_days"]
BASE = TIMING_VERSIONS["M1_add_submission_distance"]
SCORE_BASE = TIMING_VERSIONS["S0_score_add_submission_distance"]
VERSIONS = {
    "P0_pre_deadline_volume": PRE_ONLY,
    "P1_pre_plus_last1_lift": PRE_ONLY + BURST[:1],
    "P3_pre_plus_last3_lift": PRE_ONLY + BURST[1:],
    "P13_pre_plus_both_lifts": PRE_ONLY + BURST,
    "D0_timing_distance": BASE,
    "D1_plus_last1_lift": BASE + BURST[:1],
    "D3_plus_last3_lift": BASE + BURST[1:],
    "D13_plus_both_lifts": BASE + BURST,
    "D13_plus_late_interactions": BASE + BURST + INTERACTION,
    "S0_score_distance": SCORE_BASE,
    "S13_score_plus_both_lifts": SCORE_BASE + BURST,
    "S13_score_plus_late_interactions": SCORE_BASE + BURST + INTERACTION,
}
PAIR_FEATURES = BURST + ["pre_last1_share", "pre_last3_share", "pre_due_share",
                          "submitted_after_due", "relative_submission_day"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_burst_features(frame: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    """Pre columns are relative days -13..0, with day 0 excluded from bursts."""
    pre = np.asarray(pre, dtype=float)
    if pre.shape != (len(frame), 14) or not np.isfinite(pre).all() or (pre < 0).any():
        raise ValueError("Expected 14 finite nonnegative pre-deadline daily click columns")
    if not np.array_equal(pre.sum(axis=1), frame.pre_clicks.to_numpy(dtype=float)):
        raise ValueError("Daily clicks do not reconcile to pre_clicks")
    if not np.array_equal(pre[:, -3:].sum(axis=1), frame.last3_clicks.to_numpy(dtype=float)):
        raise ValueError("Daily clicks do not reconcile to existing last3_clicks")
    out = frame.copy()
    before_due = pre[:, :-1].sum(axis=1)  # d-13..d-1, 13 days
    out["pre_before_due_clicks"] = before_due
    out["log_pre_before_due_clicks"] = np.log1p(before_due)
    out["pre_before_due_active_days"] = (pre[:, :-1] > 0).sum(axis=1)
    last1 = pre[:, -2]  # d-1
    last3 = pre[:, -4:-1].sum(axis=1)  # d-3..d-1
    early12 = pre[:, :-2].sum(axis=1)  # d-13..d-2
    early10 = pre[:, :-4].sum(axis=1)  # d-13..d-4
    out["pre_last1_share"] = np.divide(last1, before_due, out=np.zeros(len(out)), where=before_due > 0)
    out["pre_last3_share"] = np.divide(last3, before_due, out=np.zeros(len(out)), where=before_due > 0)
    out["pre_due_share"] = np.divide(pre[:, -1], pre.sum(axis=1),
                                      out=np.zeros(len(out)), where=pre.sum(axis=1) > 0)
    out["pre_last1_rate_lift"] = np.log1p(last1) - np.log1p(early12 / 12)
    out["pre_last3_rate_lift"] = np.log1p(last3 / 3) - np.log1p(early10 / 10)
    out["pre_last1_lift_x_late"] = out.pre_last1_rate_lift * out.submitted_after_due
    out["pre_last3_lift_x_late"] = out.pre_last3_rate_lift * out.submitted_after_due
    return out


def load_pre_daily(frame: pd.DataFrame, data_dir: Path) -> np.ndarray:
    if frame.duplicated(KEY).any() or not frame.cutoff_day.sub(frame.due_day).eq(14).all():
        raise ValueError("Expected unique first-TMA two-week records")
    con = duckdb.connect()
    try:
        con.register("cohort", frame[KEY + ["due_day"]])
        daily = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student,
                   e.date - c.due_day AS relative_day,
                   sum(e.sum_click) AS clicks,
                   count(*) FILTER (WHERE e.sum_click IS NULL OR e.sum_click < 0) AS bad_rows
            FROM read_csv_auto('{_sql_path(data_dir / 'studentVle.csv')}', header=true) e
            JOIN cohort c USING (code_module, code_presentation, id_student)
            WHERE e.date - c.due_day BETWEEN -13 AND 0
            GROUP BY 1, 2, 3, 4
        """).df()
    finally:
        con.close()
    if daily.bad_rows.sum() or daily.clicks.isna().any():
        raise ValueError("Invalid VLE clicks")
    index = pd.MultiIndex.from_frame(frame[KEY])
    wide = (daily.pivot(index=KEY, columns="relative_day", values="clicks")
            .reindex(index=index, columns=list(range(-13, 1))).fillna(0))
    return wide.to_numpy(dtype=float)


def evaluate_models(frame: pd.DataFrame, seed: int, bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, intervals = [], []
    comparisons = [("P1_pre_plus_last1_lift", "P0_pre_deadline_volume"),
                   ("P3_pre_plus_last3_lift", "P0_pre_deadline_volume"),
                   ("P13_pre_plus_both_lifts", "P0_pre_deadline_volume"),
                   ("D1_plus_last1_lift", "D0_timing_distance"),
                   ("D3_plus_last3_lift", "D0_timing_distance"),
                   ("D13_plus_both_lifts", "D0_timing_distance"),
                   ("D13_plus_late_interactions", "D13_plus_both_lifts"),
                   ("S13_score_plus_both_lifts", "S0_score_distance"),
                   ("S13_score_plus_late_interactions", "S13_score_plus_both_lifts")]
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across split")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, columns) for name, columns in VERSIONS.items()}
        for name, score in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train),
                         "test_rows": len(test), "test_students": test.id_student.nunique(),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, score)),
                         "brier": float(brier_score_loss(y, score))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 28100)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for name, comparator in comparisons:
            p, q = predictions[name], predictions[comparator]
            ap_deltas, brier_deltas = [], []
            for indices in draws:
                if len(np.unique(y[indices])) < 2:
                    continue
                ap_deltas.append(average_precision_score(y[indices], p[indices]) -
                                 average_precision_score(y[indices], q[indices]))
                brier_deltas.append(brier_score_loss(y[indices], p[indices]) -
                                    brier_score_loss(y[indices], q[indices]))
            intervals.append({"model": name, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ap_ci_low": float(np.quantile(ap_deltas, .025)),
                              "ap_ci_high": float(np.quantile(ap_deltas, .975)),
                              "delta_brier": float(brier_score_loss(y, p) - brier_score_loss(y, q)),
                              "brier_ci_low": float(np.quantile(brier_deltas, .025)),
                              "brier_ci_high": float(np.quantile(brier_deltas, .975))})
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def profile_daily(frame: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    rows = []
    denominator = pre.sum(axis=1)
    positive = denominator > 0
    for outcome, selected in [("Fail", frame.final_result.eq("Fail").to_numpy()),
                              ("Pass/Distinction", frame.final_result.ne("Fail").to_numpy())]:
        selected = selected & positive
        shares = pre[selected] / denominator[selected, None]
        for j, day in enumerate(range(-13, 1)):
            rows.append({"outcome": outcome, "relative_day": day, "records": int(selected.sum()),
                         "mean_clicks": float(pre[selected, j].mean()),
                         "mean_student_share": float(shares[:, j].mean()),
                         "active_fraction": float((pre[selected, j] > 0).mean())})
    return pd.DataFrame(rows)


def compare_pairs(frame: pd.DataFrame, output_dir: Path, seed: int,
                  repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame[frame.first_tma_score.notna()].reset_index(drop=True).copy()
    scored["total_clicks"] = scored.post_clicks
    pairs = make_joint_pairs(scored, score_gap=5, click_tolerance=.05, seed=seed)
    if pairs.empty:
        raise ValueError("No score-and-click matched pairs")
    pairs.to_parquet(output_dir / "joint_score_post_pairs.parquet", index=False)
    fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    smaller = np.minimum(fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy())
    larger = np.maximum(fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy())
    pre_close = ((smaller == 0) & (larger == 0)) | ((smaller > 0) &
                                                   (larger / np.maximum(smaller, 1) <= 1.2 + 1e-12))
    positive_pre_close = fail.post_clicks.gt(0).to_numpy() & pre_close
    both_on_time = fail.submitted_after_due.eq(0).to_numpy() & passing.submitted_after_due.eq(0).to_numpy()
    subsets = [("all_scored", np.ones(len(pairs), dtype=bool)),
               ("positive_post_pre20", positive_pre_close),
               ("positive_post_pre20_both_not_late", positive_pre_close & both_on_time)]
    balances, effects = [], []
    for name, selected in subsets:
        group = pairs.loc[selected].reset_index(drop=True)
        a, b = fail.loc[selected].reset_index(drop=True), passing.loc[selected].reset_index(drop=True)
        difference = a[PAIR_FEATURES].to_numpy(dtype=float) - b[PAIR_FEATURES].to_numpy(dtype=float)
        ci = _component_intervals(group, difference, seed, repeats)
        balances.append({"subset": name, "pairs": len(group),
                         "mean_abs_score_gap": float(group.score_gap.abs().mean()),
                         "mean_relative_post_click_gap": float(group.relative_click_gap.mean()),
                         "fail_pre_click_mean": float(a.pre_clicks.mean()),
                         "pass_pre_click_mean": float(b.pre_clicks.mean()),
                         "fail_post_click_mean": float(a.post_clicks.mean()),
                         "pass_post_click_mean": float(b.post_clicks.mean())})
        for j, feature in enumerate(PAIR_FEATURES):
            effects.append({"subset": name, "feature": feature, "pairs": len(group),
                            "fail_mean": float(a[feature].mean()), "pass_mean": float(b[feature].mean()),
                            "fail_minus_pass": float(difference[:, j].mean()),
                            "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j])})
    return pd.DataFrame(balances), pd.DataFrame(effects)


def compare_volume_clusters(frame: pd.DataFrame, pair_path: Path, cluster_path: Path,
                            seed: int, repeats: int) -> pd.DataFrame:
    labels = pd.read_parquet(cluster_path)
    if labels.duplicated(KEY).any():
        raise ValueError("Duplicate volume-cluster key")
    scored = frame[frame.first_tma_score.notna()].reset_index(drop=True)
    scored = scored.merge(labels[KEY + ["volume_cluster"]], on=KEY,
                          how="left", validate="one_to_one", sort=False)
    if scored.volume_cluster.isna().any():
        raise ValueError("Missing volume-cluster label")
    pairs = pd.read_parquet(pair_path)
    fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    same = fail.volume_cluster.eq(passing.volume_cluster).to_numpy()
    smaller = np.minimum(fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy())
    larger = np.maximum(fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy())
    pre_close = ((smaller == 0) & (larger == 0)) | ((smaller > 0) &
                                                   (larger / np.maximum(smaller, 1) <= 1.2 + 1e-12))
    both_on_time = fail.submitted_after_due.eq(0).to_numpy() & passing.submitted_after_due.eq(0).to_numpy()
    rows = []
    for label, name in [(-1, "zero"), (0, "low"), (1, "high")]:
        for subset, selected in [("all_scored", np.ones(len(pairs), dtype=bool)),
                                 ("pre20_both_on_time", pre_close & both_on_time)]:
            selected = selected & same & fail.volume_cluster.eq(label).to_numpy()
            group = pairs.loc[selected].reset_index(drop=True)
            if group.empty:
                continue
            a = fail.loc[selected].reset_index(drop=True)
            b = passing.loc[selected].reset_index(drop=True)
            difference = (a.pre_last3_rate_lift.to_numpy() -
                          b.pre_last3_rate_lift.to_numpy()).reshape(-1, 1)
            ci = _component_intervals(group, difference, seed, repeats)
            rows.append({"volume_cluster": name, "subset": subset, "pairs": len(group),
                         "fail_mean_last3_lift": float(a.pre_last3_rate_lift.mean()),
                         "pass_mean_last3_lift": float(b.pre_last3_rate_lift.mean()),
                         "fail_minus_pass": float(difference.mean()),
                         "ci_low": float(ci[0, 0]), "ci_high": float(ci[1, 0]),
                         "fail_pre_click_mean": float(a.pre_clicks.mean()),
                         "pass_pre_click_mean": float(b.pre_clicks.mean()),
                         "mean_relative_post_click_gap": float(group.relative_click_gap.mean()),
                         "mean_abs_score_gap": float(group.score_gap.abs().mean())})
    return pd.DataFrame(rows)


def overdue_burst_groups(frame: pd.DataFrame) -> pd.DataFrame:
    status = np.select([frame.submitted_by_cutoff.eq(0), frame.submitted_after_due.eq(1)],
                       ["not_submitted_by_14", "late_by_14"], default="on_time_by_14")
    return (frame.assign(submission_status=status)
            .groupby("submission_status")
            .agg(records=("id_student", "size"), fail_records=("final_result", lambda x: int(x.eq("Fail").sum())),
                 mean_pre_clicks=("pre_clicks", "mean"),
                 mean_last1_lift=("pre_last1_rate_lift", "mean"),
                 mean_last3_lift=("pre_last3_rate_lift", "mean"),
                 mean_last3_share=("pre_last3_share", "mean"))
            .assign(fail_fraction=lambda x: x.fail_records / x.records).reset_index())


def plot_daily(profile: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for outcome, color in [("Fail", "#B87916"), ("Pass/Distinction", "#3856B3")]:
        part = profile[profile.outcome.eq(outcome)]
        ax.plot(part.relative_day, part.mean_student_share, marker="o", markersize=3.5,
                color=color, label=outcome)
    ax.axvline(-1, color="#78818E", linewidth=1, linestyle="--")
    ax.set_xlabel("Day relative to first weighted TMA scheduled deadline")
    ax.set_ylabel("Mean share of student's d−13..d clicks")
    ax.set_title("Pre-deadline VLE clicks by eventual outcome", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E8EE")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def run(data_dir: Path, node_dir: Path, two_week_dir: Path, output_dir: Path,
        reference_models: Path, cluster_dir: Path | None = None, seed: int = 42, bootstrap: int = 500,
        repeats: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    node_path = node_dir / "node_features.parquet"
    feature_path = two_week_dir / "features.parquet"
    two_week_protocol = json.loads((two_week_dir / "protocol.json").read_text(encoding="utf-8"))
    node_protocol = json.loads((node_dir / "node_protocol.json").read_text(encoding="utf-8"))
    if two_week_protocol["node_features_sha256"] != _sha256(node_path):
        raise ValueError("Two-week features and node records do not share a source")
    for name in ("assessments", "studentAssessment", "studentVle"):
        if node_protocol["sources_sha256"][name] != _sha256(data_dir / f"{name}.csv"):
            raise ValueError(f"Node source differs: {name}")
    frame = add_submission_distance(pd.read_parquet(feature_path), pd.read_parquet(node_path), data_dir)
    pre = load_pre_daily(frame, data_dir)
    frame = add_burst_features(frame, pre)
    models, intervals = evaluate_models(frame, seed, bootstrap)
    old = pd.read_csv(reference_models)
    reference = old[old.model.isin(["M1_add_submission_distance", "S0_score_add_submission_distance"])]
    current = models[models.model.isin(["D0_timing_distance", "S0_score_distance"])].copy()
    current["model"] = current.model.replace({"D0_timing_distance": "M1_add_submission_distance",
                                               "S0_score_distance": "S0_score_add_submission_distance"})
    check = current.merge(reference, on=["seed", "model"], suffixes=("_new", "_old"), validate="one_to_one")
    if len(check) != 10 or not np.allclose(check.average_precision_new, check.average_precision_old,
                                          atol=1e-12, rtol=0) or not np.allclose(
                                              check.brier_new, check.brier_old, atol=1e-12, rtol=0):
        raise ValueError("Burst study baselines disagree with submission-timing study")
    output_dir.mkdir(parents=True, exist_ok=True)
    balance, effects = compare_pairs(frame, output_dir, seed, repeats)
    profile = profile_daily(frame, pre)
    tables = {"models": models, "ap_intervals": intervals, "daily_profile": profile,
              "joint_balance": balance, "joint_effects": effects,
              "overdue_burst_groups": overdue_burst_groups(frame)}
    if cluster_dir is not None:
        cluster_protocol = json.loads((cluster_dir / "volume_cluster_protocol.json").read_text(encoding="utf-8"))
        if cluster_protocol["input_sha256"]["features.parquet"] != _sha256(feature_path):
            raise ValueError("Volume clusters and two-week features do not share a source")
        tables["volume_cluster_effects"] = compare_volume_clusters(
            frame, output_dir / "joint_score_post_pairs.parquet",
            cluster_dir / "sample_volume_clusters.parquet", seed, repeats)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    plot_daily(profile, output_dir / "predeadline_daily_profile.png")
    frame[KEY + ["final_result", "pre_clicks", "post_clicks", "first_tma_score",
                 "submitted_by_cutoff", "submitted_after_due"] + BURST + INTERACTION +
          ["pre_before_due_clicks", "log_pre_before_due_clicks", "pre_before_due_active_days",
           "pre_last1_share", "pre_last3_share", "pre_due_share"]].to_parquet(
              output_dir / "burst_features.parquet", index=False)
    protocol = {"question": "Do short VLE click rate lifts immediately before first weighted TMA deadline add information?",
                "window": {"all_pre": [-13, 0], "burst_last1": [-1, -1], "burst_last3": [-3, -1],
                           "comparison_last1": [-13, -2], "comparison_last3": [-13, -4]},
                "lift": "log1p(mean clicks/day in short window)-log1p(mean clicks/day in earlier window)",
                "deadline_day": "d=0 excluded from burst; daily timestamps lack hour",
                "pre_only_models": "P versions use only module/presentation and VLE clicks dated d-13..d-1; cohort still selected retrospectively at d+14",
                "late": "studentAssessment.date_submitted > assessments.date and <= d+14; already in baseline",
                "models": VERSIONS, "seed": seed, "student_bootstrap": bootstrap, "pair_bootstrap": repeats,
                "input_sha256": {"node_features.parquet": _sha256(node_path),
                                 "features.parquet": _sha256(feature_path),
                                 "studentVle.csv": _sha256(data_dir / "studentVle.csv"),
                                 "assessments.csv": _sha256(data_dir / "assessments.csv"),
                                 "studentAssessment.csv": _sha256(data_dir / "studentAssessment.csv")},
                "reference_models_sha256": _sha256(reference_models),
                "volume_clusters_sha256": _sha256(cluster_dir / "sample_volume_clusters.parquet") if cluster_dir else None,
                "code_sha256": _sha256(Path(__file__)),
                "limitations": "VLE clicks are course-wide and not all learning activity; assignment release dates and personal extensions unknown; P models use a retrospectively selected d+14 non-withdrawn cohort; grades and late submission observed retrospectively; repeated exploratory testing; no causal effect"}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True)
    parser.add_argument("--two-week-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-models", type=Path, required=True)
    parser.add_argument("--cluster-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.data_dir, args.node_dir, args.two_week_dir, args.output_dir,
        args.reference_models, args.cluster_dir, args.seed, args.bootstrap, args.repeats)


if __name__ == "__main__":
    main()
