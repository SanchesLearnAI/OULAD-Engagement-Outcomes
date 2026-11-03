"""Retrospective score/rank ablation at first weighted TMA +21 and +28 days.

Final recorded marks have no release timestamp in OULAD. This is an explanatory
benchmark, not a deployable as-of-cutoff early-warning model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import KEY, SUBMIT, TRACK, _fit_predict
from .tma_matched import RESOURCE_SUMMARY

SEEDS = (42, 143, 144, 145, 146)
VOLUME = ["module_presentation", "log_clicks", "log_clicks_squared",
          "pre_clicks", "post_clicks"] + SUBMIT
BASE = VOLUME + TRACK
SCORE = ["first_tma_score", "score_recorded_for_submission"]
RANK = ["first_tma_rank_train"]


def _percentiles(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    observed = values.notna().to_numpy()
    if observed.any() and len(reference):
        ordered = np.sort(reference)
        x = values.to_numpy(dtype=float)[observed]
        left = np.searchsorted(ordered, x, side="left")
        right = np.searchsorted(ordered, x, side="right")
        result[observed] = (left + right) / (2 * len(ordered))
    return result


def add_train_only_rank(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map scores to within-presentation percentiles using training marks only."""
    train = train.copy()
    test = test.copy()
    all_reference = train.first_tma_score.dropna().to_numpy(dtype=float)
    if not len(all_reference):
        raise ValueError("No recorded first-TMA marks in training data")
    for frame in (train, test):
        frame["first_tma_rank_train"] = np.nan
        for presentation, positions in frame.groupby("module_presentation").indices.items():
            reference = train.loc[train.module_presentation.eq(presentation),
                                  "first_tma_score"].dropna().to_numpy(dtype=float)
            if not len(reference):
                reference = all_reference
            frame.iloc[positions, frame.columns.get_loc("first_tma_rank_train")] = _percentiles(
                frame.iloc[positions].first_tma_score, reference)
    return train, test


def evaluate(frame: pd.DataFrame, post_days: int, seed: int = 42,
             bootstrap: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.duplicated(KEY).any() or not frame.final_result.isin(["Fail", "Pass", "Distinction"]).all():
        raise ValueError("Expected unique binary TMA records")
    resource = RESOURCE_SUMMARY + sorted(c for c in frame if c.startswith("resource_share__") or
                                          c.startswith("post_resource_share__"))
    versions = {"B3_timing": BASE,
                "B4_resources": BASE + resource,
                "S0_score_volume": VOLUME + SCORE,
                "S1_add_score": BASE + SCORE,
                "S2_add_rank": BASE + SCORE + RANK,
                "S3_full": BASE + SCORE + RANK + resource}
    metrics, intervals = [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                              random_state=split_seed).split(frame, groups=frame.id_student))
        train, test = add_train_only_rank(frame.iloc[left], frame.iloc[right])
        if set(train.id_student) & set(test.id_student):
            raise AssertionError("Student ID in both train and test")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, columns)
                       for name, columns in versions.items()}
        for name, p in predictions.items():
            metrics.append({"post_days": post_days, "seed": split_seed, "model": name,
                            "train_rows": len(train), "test_rows": len(test),
                            "train_students": train.id_student.nunique(),
                            "test_students": test.id_student.nunique(),
                            "test_fail_prevalence": float(y.mean()),
                            "train_score_coverage": float(train.first_tma_score.notna().mean()),
                            "test_score_coverage": float(test.first_tma_score.notna().mean()),
                            "ap": float(average_precision_score(y, p)),
                            "roc_auc": float(roc_auc_score(y, p)),
                            "brier": float(brier_score_loss(y, p))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 31000 + post_days)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for name, comparator in [("S1_add_score", "B3_timing"),
                                  ("S1_add_score", "S0_score_volume"),
                                  ("S2_add_rank", "S1_add_score"),
                                  ("S3_full", "S2_add_rank"),
                                  ("S3_full", "B3_timing")]:
            p, q = predictions[name], predictions[comparator]
            deltas = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                      for ix in draws if len(np.unique(y[ix])) == 2]
            intervals.append({"post_days": post_days, "model": name,
                              "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ci_low": float(np.quantile(deltas, .025)),
                              "ci_high": float(np.quantile(deltas, .975)),
                              "bootstrap": bootstrap})
    return pd.DataFrame(metrics), pd.DataFrame(intervals)


def run(three: Path, four: Path, output_dir: Path,
        seed: int = 42, bootstrap: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    inputs = {21: three, 28: four}
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, intervals = [], []
    for days, path in inputs.items():
        frame = pd.read_parquet(path / f"scored_features_{days}d.parquet")
        if not frame.cutoff_day.sub(frame.due_day).eq(days).all():
            raise ValueError(f"Wrong observation cutoff for {days}-day cohort")
        if not frame.first_tma_score.dropna().between(0, 100).all():
            raise ValueError("Score outside 0–100")
        a, b = evaluate(frame, days, seed, bootstrap)
        metrics.append(a); intervals.append(b)
    pd.concat(metrics, ignore_index=True).to_csv(output_dir / "models.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(output_dir / "ap_intervals.csv", index=False)
    (output_dir / "protocol.json").write_text(json.dumps({
        "task": "Retrospective Fail vs Pass/Distinction at first weighted TMA +21/+28",
        "not_operational": "OULAD lacks score release dates; submitted-before-cutoff final recorded scores may not have been known at cutoff",
        "rank": "Within module presentation, mapped to training-score empirical midrank percentile; missing rank is imputed from training with an explicit score-availability indicator",
        "split": "student-ID GroupShuffleSplit 80/20, seeds 42,143-146; identical test rows within each horizon",
        "seed": seed, "student_bootstrap": bootstrap,
        "features_sha256": {str(days): hashlib.sha256((path / f"scored_features_{days}d.parquet").read_bytes()).hexdigest()
                            for days, path in inputs.items()},
        "caution": "First weighted TMA contributes to final result. Absolute AP across horizons uses different observation times; overlapping splits are exploratory."
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three", type=Path, required=True,
                        help="Directory containing scored_features_21d.parquet")
    parser.add_argument("--four", type=Path, required=True,
                        help="Directory containing scored_features_28d.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    run(args.three, args.four, args.output_dir, args.seed, args.bootstrap)


if __name__ == "__main__":
    main()
