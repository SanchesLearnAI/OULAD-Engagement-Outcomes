"""TMA submission distance from the scheduled deadline in the two-week cohort.

Submission day is observed by +14; final recorded score release day is unknown.
The score-adjusted comparison is therefore retrospective. Scheduled deadlines
may differ from individual extensions, which OULAD does not record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import BATCH, KEY, _first_weighted_tma, _fit_predict
from .matched import _component_intervals
from .tma_score_gap import BASE, SCORE, build_score_frame
from .tma_score_probe import make_joint_pairs

DISTANCE = ["days_early", "days_late"]
VERSIONS = {"M1_timing": BASE,
            "M1_add_submission_distance": BASE + DISTANCE,
            "S0_score": BASE + SCORE,
            "S0_score_add_submission_distance": BASE + SCORE + DISTANCE}
PAIR_FEATURES = ["relative_submission_day", "days_early", "days_late",
                 "first_tma_score", "pre_clicks", "post_clicks", "post_active_days"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_submission_distance(features: pd.DataFrame, node: pd.DataFrame,
                            data_dir: Path) -> pd.DataFrame:
    frame = build_score_frame(features, node, data_dir)
    selected = _first_weighted_tma(pd.read_csv(data_dir / "assessments.csv", na_values=["?"]))
    grades = pd.read_csv(data_dir / "studentAssessment.csv", na_values=["?"])
    if grades.duplicated(["id_assessment", "id_student"]).any():
        raise ValueError("Duplicate assessment submission key")
    frame = frame.merge(selected[BATCH + ["id_assessment", "due_day"]], on=BATCH,
                        validate="many_to_one", suffixes=("", "_source"), sort=False)
    if not frame.due_day.eq(frame.due_day_source).all():
        raise ValueError("TMA deadline disagrees with two-week node")
    frame = frame.merge(grades[["id_assessment", "id_student", "date_submitted", "is_banked"]],
                        on=["id_assessment", "id_student"], how="left", validate="one_to_one", sort=False)
    if frame.is_banked.eq(1).any():
        raise ValueError("Banked assessment in eligible cohort")
    observed_day = frame.date_submitted.where(frame.date_submitted.le(frame.cutoff_day))
    relative_day = observed_day - frame.due_day
    if not frame.submitted_by_cutoff.eq(observed_day.notna().astype(int)).all():
        raise ValueError("Submitted-by-cutoff indicator disagrees with submission day")
    if not frame.submitted_after_due.eq(relative_day.gt(0).astype(int)).all():
        raise ValueError("After-due indicator disagrees with submission day")
    if frame.first_tma_score.notna().any() and not frame.loc[frame.first_tma_score.notna(), "date_submitted"].le(
            frame.loc[frame.first_tma_score.notna(), "cutoff_day"]).all():
        raise ValueError("Future submission has been assigned a score")
    frame["relative_submission_day"] = relative_day
    frame["days_early"] = (-relative_day).clip(lower=0)
    frame["days_late"] = relative_day.clip(lower=0)
    return frame.drop(columns=["due_day_source", "date_submitted", "is_banked"])


def evaluate_models(frame: pd.DataFrame, seed: int,
                    bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, intervals = [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across train and test")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, columns) for name, columns in VERSIONS.items()}
        for name, scores in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train),
                         "test_rows": len(test), "test_students": test.id_student.nunique(),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, scores)),
                         "brier": float(brier_score_loss(y, scores))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 19100)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for model, comparator in (("M1_add_submission_distance", "M1_timing"),
                                  ("S0_score_add_submission_distance", "S0_score")):
            p, q = predictions[model], predictions[comparator]
            ap_deltas, brier_deltas = [], []
            for indices in draws:
                if len(np.unique(y[indices])) < 2:
                    continue
                ap_deltas.append(average_precision_score(y[indices], p[indices]) -
                                 average_precision_score(y[indices], q[indices]))
                brier_deltas.append(brier_score_loss(y[indices], p[indices]) -
                                    brier_score_loss(y[indices], q[indices]))
            intervals.append({"model": model, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ap_ci_low": float(np.quantile(ap_deltas, .025)),
                              "ap_ci_high": float(np.quantile(ap_deltas, .975)),
                              "delta_brier": float(brier_score_loss(y, p) - brier_score_loss(y, q)),
                              "brier_ci_low": float(np.quantile(brier_deltas, .025)),
                              "brier_ci_high": float(np.quantile(brier_deltas, .975))})
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def describe_submission(frame: pd.DataFrame) -> pd.DataFrame:
    day = frame.relative_submission_day
    bands = np.select([day.isna(), day.le(-7), day.lt(0), day.eq(0), day.le(7)],
                      ["not_submitted_by_14", "early_7plus", "early_1to6",
                       "due_day", "late_1to7"], default="late_8to14")
    grouped = (frame.assign(submission_band=bands).groupby("submission_band").agg(
        records=("id_student", "size"),
        fail_records=("final_result", lambda s: int(s.eq("Fail").sum())),
        mean_recorded_score=("first_tma_score", "mean"),
        mean_pre_clicks=("pre_clicks", "mean"),
        mean_post_clicks=("post_clicks", "mean")).reset_index())
    grouped["fail_fraction"] = grouped.fail_records / grouped.records
    return grouped


def compare_score_post_pairs(frame: pd.DataFrame, output_dir: Path,
                             seed: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame[frame.first_tma_score.notna()].reset_index(drop=True).copy()
    scored["total_clicks"] = scored.post_clicks
    pairs = make_joint_pairs(scored, score_gap=5, click_tolerance=.05, seed=seed)
    if pairs.empty:
        raise ValueError("No score-and-post-click matched pairs")
    pairs.to_parquet(output_dir / "joint_score_post_pairs.parquet", index=False)
    fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    pre_fail, pre_pass = fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy()
    smaller, larger = np.minimum(pre_fail, pre_pass), np.maximum(pre_fail, pre_pass)
    pre_close = ((smaller == 0) & (larger == 0)) | ((smaller > 0) &
                                                   (larger / np.maximum(smaller, 1) <= 1.2 + 1e-12))
    positive = fail.post_clicks.gt(0).to_numpy()
    subsets = [("all_scored", np.ones(len(pairs), dtype=bool)),
               ("positive_post", positive),
               ("positive_post_pre20", positive & pre_close)]
    balance, effects = [], []
    for name, selected in subsets:
        group = pairs.loc[selected].reset_index(drop=True)
        a, b = fail.loc[selected].reset_index(drop=True), passing.loc[selected].reset_index(drop=True)
        differences = a[PAIR_FEATURES].to_numpy(dtype=float) - b[PAIR_FEATURES].to_numpy(dtype=float)
        ci = _component_intervals(group, differences, seed, repeats)
        balance.append({"subset": name, "pairs": len(group),
                        "mean_abs_score_gap": float(group.score_gap.abs().mean()),
                        "mean_relative_post_click_gap": float(group.relative_click_gap.mean()),
                        "fail_score_mean": float(a.first_tma_score.mean()),
                        "pass_score_mean": float(b.first_tma_score.mean()),
                        "fail_pre_click_mean": float(a.pre_clicks.mean()),
                        "pass_pre_click_mean": float(b.pre_clicks.mean()),
                        "fail_post_click_mean": float(a.post_clicks.mean()),
                        "pass_post_click_mean": float(b.post_clicks.mean())})
        for j, feature in enumerate(PAIR_FEATURES):
            effects.append({"subset": name, "feature": feature, "pairs": len(group),
                            "fail_mean": float(a[feature].mean()), "pass_mean": float(b[feature].mean()),
                            "fail_minus_pass": float(differences[:, j].mean()),
                            "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j])})
    return pd.DataFrame(balance), pd.DataFrame(effects)


def compare_within_volume_clusters(frame: pd.DataFrame, pair_path: Path,
                                   cluster_path: Path, seed: int, repeats: int) -> pd.DataFrame:
    labels = pd.read_parquet(cluster_path)
    if labels.duplicated(KEY).any():
        raise ValueError("Duplicate cluster label key")
    scored = frame[frame.first_tma_score.notna()].reset_index(drop=True)
    scored = scored.merge(labels[KEY + ["volume_cluster"]], on=KEY,
                          how="left", validate="one_to_one", sort=False)
    if scored.volume_cluster.isna().any():
        raise ValueError("Missing volume cluster label")
    pairs = pd.read_parquet(pair_path)
    fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    same = fail.volume_cluster.eq(passing.volume_cluster).to_numpy()
    pre_fail, pre_pass = fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy()
    smaller, larger = np.minimum(pre_fail, pre_pass), np.maximum(pre_fail, pre_pass)
    pre_close = ((smaller == 0) & (larger == 0)) | ((smaller > 0) &
                                                   (larger / np.maximum(smaller, 1) <= 1.2 + 1e-12))
    rows = []
    for label, name in [(-1, "zero"), (0, "low"), (1, "high")]:
        for subset, additional in [("all_scored", np.ones(len(pairs), dtype=bool)),
                                   ("pre_clicks_within_20pct", pre_close)]:
            chosen = same & fail.volume_cluster.eq(label).to_numpy() & additional
            group = pairs.loc[chosen].reset_index(drop=True)
            a = fail.loc[chosen].reset_index(drop=True)
            b = passing.loc[chosen].reset_index(drop=True)
            if group.empty:
                continue
            difference = (a.relative_submission_day.to_numpy() -
                          b.relative_submission_day.to_numpy()).reshape(-1, 1)
            ci = _component_intervals(group, difference, seed, repeats)
            rows.append({"volume_cluster": name, "subset": subset, "pairs": len(group),
                         "fail_mean_relative_day": float(a.relative_submission_day.mean()),
                         "pass_mean_relative_day": float(b.relative_submission_day.mean()),
                         "fail_minus_pass_days": float(difference.mean()),
                         "ci_low": float(ci[0, 0]), "ci_high": float(ci[1, 0]),
                         "fail_pre_click_mean": float(a.pre_clicks.mean()),
                         "pass_pre_click_mean": float(b.pre_clicks.mean()),
                         "mean_relative_post_click_gap": float(group.relative_click_gap.mean()),
                         "mean_abs_score_gap": float(group.score_gap.abs().mean())})
    return pd.DataFrame(rows)


def plot_model_ap(models: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    comparisons = [("M1: timing", "M1_timing", "#3856B3"),
                   ("M1 + submission distance", "M1_add_submission_distance", "#B87916"),
                   ("S0: timing + mark", "S0_score", "#3856B3"),
                   ("S0 + submission distance", "S0_score_add_submission_distance", "#B87916")]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for y, (_, name, color) in enumerate(comparisons):
        values = models.loc[models.model.eq(name)].sort_values("seed").average_precision.to_numpy()
        ax.scatter(values, y + np.linspace(-.10, .10, len(values)), s=42, color=color, alpha=.65)
        ax.scatter([values.mean()], [y], marker="D", s=78, facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=3)
    ax.set_yticks(range(len(comparisons)), [label for label, _, _ in comparisons])
    ax.set_ylim(len(comparisons) - .5, -.5)
    ax.set_xlabel("Fail average precision (focused scale)")
    ax.set_title("Submission distance from scheduled TMA deadline", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E8EE")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.23, .025, "Dots: overlapping student-grouped tests; open diamond: mean. Mark-release dates are unknown.",
             fontsize=9, color="#4A5260")
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def run(data_dir: Path, node_dir: Path, two_week_dir: Path, output_dir: Path,
        reference_models: Path | None = None, cluster_dir: Path | None = None, seed: int = 42,
        bootstrap: int = 500, repeats: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    node_path = node_dir / "node_features.parquet"
    feature_path = two_week_dir / "features.parquet"
    source_protocol = json.loads((two_week_dir / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol["node_features_sha256"] != _sha256(node_path):
        raise ValueError("Two-week features and node records do not share a source")
    node_protocol = json.loads((node_dir / "node_protocol.json").read_text(encoding="utf-8"))
    for name in ("assessments", "studentAssessment"):
        if node_protocol["sources_sha256"][name] != _sha256(data_dir / f"{name}.csv"):
            raise ValueError(f"Assessment source differs: {name}")
    frame = add_submission_distance(pd.read_parquet(feature_path), pd.read_parquet(node_path), data_dir)
    models, intervals = evaluate_models(frame, seed, bootstrap)
    if reference_models is not None:
        previous = pd.read_csv(reference_models)
        previous = previous[previous.model.isin(["M1_timing", "S0_score"])]
        current = models[models.model.isin(["M1_timing", "S0_score"])]
        merged = current.merge(previous, on=["seed", "model"], suffixes=("_new", "_old"), validate="one_to_one")
        if len(merged) != 10 or not np.allclose(merged.average_precision_new, merged.average_precision_old,
                                               rtol=0, atol=1e-12) or not np.allclose(
                                                   merged.brier_new, merged.brier_old, rtol=0, atol=1e-12):
            raise ValueError("Submission study baselines disagree with prior score study")
    output_dir.mkdir(parents=True, exist_ok=True)
    balance, effects = compare_score_post_pairs(frame, output_dir, seed, repeats)
    tables = {"models": models, "ap_intervals": intervals,
              "submission_bands": describe_submission(frame),
              "joint_balance": balance, "joint_effects": effects}
    if cluster_dir is not None:
        cluster_protocol = json.loads((cluster_dir / "volume_cluster_protocol.json").read_text(encoding="utf-8"))
        if cluster_protocol["input_sha256"]["features.parquet"] != _sha256(feature_path):
            raise ValueError("Volume clusters and two-week features do not share a source")
        tables["volume_cluster_effects"] = compare_within_volume_clusters(
            frame, output_dir / "joint_score_post_pairs.parquet",
            cluster_dir / "sample_volume_clusters.parquet", seed, repeats)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    plot_model_ap(models, output_dir / "submission_model_ap.png")
    protocol = {"question": "Does submission distance from first weighted TMA deadline add information after click, timing, and mark?",
                "scheduled_deadline": "assessments.date; individual extensions unrecorded",
                "submission": "studentAssessment.date_submitted, only if <= due+14; future dates masked",
                "distance": "days_early=max(due-submission,0), days_late=max(submission-due,0); missing if no submission by +14",
                "model_columns": VERSIONS,
                "seed": seed, "student_bootstrap": bootstrap, "pair_bootstrap": repeats,
                "joint_pairs": "same course/presentation, first TMA score within 5 points, post clicks within 5% after exact; no reuse",
                "input_sha256": {"node_features.parquet": _sha256(node_path),
                                 "features.parquet": _sha256(feature_path),
                                 "assessments.csv": _sha256(data_dir / "assessments.csv"),
                                 "studentAssessment.csv": _sha256(data_dir / "studentAssessment.csv")},
                "reference_models_sha256": _sha256(reference_models) if reference_models else None,
                "volume_clusters_sha256": _sha256(cluster_dir / "sample_volume_clusters.parquet") if cluster_dir else None,
                "code_sha256": _sha256(Path(__file__)),
                "limitations": "scheduled due date may differ from personal extensions; score release unknown; score contributes to final outcome; retrospective selected cohort; repeated test use; no causal effect"}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True)
    parser.add_argument("--two-week-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-models", type=Path)
    parser.add_argument("--cluster-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.data_dir, args.node_dir, args.two_week_dir, args.output_dir,
        args.reference_models, args.cluster_dir, args.seed, args.bootstrap, args.repeats)


if __name__ == "__main__":
    main()
