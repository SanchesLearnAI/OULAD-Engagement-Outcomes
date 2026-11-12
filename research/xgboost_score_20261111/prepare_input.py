"""Rebuild the five score-inclusive OULAD splits and their logistic baseline.

The resulting NPZ contains student-level rows and must stay in an ignored local
output directory. Only aggregate evaluation results belong in Git.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from oulad_analysis.assessment_node import _fit_predict  # noqa: E402
from oulad_analysis.tma_submission_persistence import MODELS  # noqa: E402


NUMERIC = [
    "log_pre_clicks", "pre_active_days", "last3_share", "log_post_clicks",
    "log_post_clicks_squared", "submitted_by_cutoff", "submitted_after_due",
    "post_active_days", "post_both_weeks", "post_longest_inactive_run",
    "post_week2_share", "post_peak7_share", "post_entropy_norm",
    "post_first_active_day", "days_early", "days_late", "first_tma_score",
    "score_recorded_for_submission", "unsubmitted_x_inactive_run",
]
SEEDS = (42, 143, 144, 145, 146)
BASELINE_NAME = "S_single_unsubmitted_x_inactive_run"
REFERENCE_AP = (0.7186571521619861, 0.7188570756997005,
                0.7004578184803578, 0.7053049239504816,
                0.7249313395187422)


def prepare(features_path: Path, scores_path: Path, output: Path) -> None:
    frame = pd.read_parquet(features_path)
    scores = pd.read_parquet(scores_path)
    key = ["code_module", "code_presentation", "id_student"]
    if frame.duplicated(key).any() or scores.duplicated(key).any():
        raise ValueError("Enrolment keys must be unique in both input tables")
    if not frame[key].equals(scores[key]):
        raise ValueError("Feature and score rows must have identical enrolment keys")
    if len(frame) != 19806 or int(frame.y_fail.sum()) != 6092:
        raise ValueError("Unexpected cohort; this benchmark requires the frozen cohort")
    baseline_features = MODELS[BASELINE_NAME]
    if (len(NUMERIC) != len(set(NUMERIC)) or
            set(NUMERIC) != set(baseline_features) - {"module_presentation"}):
        raise ValueError("XGBoost numeric features no longer match the logistic baseline")

    # features_scores.parquet includes eventual scores. Hide any score not
    # recorded for a submission by the d+14 observation cutoff.
    available = scores.has_score_by_cutoff.eq(1)
    if scores.loc[available, "score"].isna().any():
        raise ValueError("A score marked available is missing")
    frame["first_tma_score"] = scores.score.where(available)
    frame["score_recorded_for_submission"] = available.astype(int)
    frame["unsubmitted_x_inactive_run"] = (
        (1 - frame.submitted_by_cutoff) * frame.post_longest_inactive_run / 14)
    category = pd.get_dummies(frame.module_presentation, dtype=float)
    if category.shape[1] != 19:
        raise ValueError("Expected 19 module-presentation categories")
    names = NUMERIC + [f"module_presentation={column}" for column in category.columns]
    numeric = frame[NUMERIC].to_numpy(dtype=float)
    labels = frame.y_fail.to_numpy(dtype=np.int64)
    students = frame.id_student.to_numpy(dtype=np.int64)
    payload: dict[str, np.ndarray] = {
        "y": labels, "student": students, "feature_names": np.asarray(names),
    }
    for seed, expected in zip(SEEDS, REFERENCE_AP):
        train, test = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=seed)
            .split(frame, groups=students))
        if set(students[train]) & set(students[test]):
            raise AssertionError("Student appears in both train and test")
        baseline = _fit_predict(
            frame.iloc[train], frame.iloc[test],
            baseline_features)
        ap = average_precision_score(labels[test], baseline)
        if not np.isclose(ap, expected, atol=1e-10, rtol=0):
            raise AssertionError(f"Baseline mismatch for seed {seed}: {ap} != {expected}")
        transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        transform.fit(numeric[train])
        # Fitting the unsupervised transform on the complete outer train set
        # preserves the original experiment's exact inner-validation setup.
        x = np.concatenate([transform.transform(numeric), category.to_numpy()],
                           axis=1).astype("float32")
        if x.shape[1] != 38:
            raise AssertionError("Expected 19 numeric and 19 category columns")
        payload.update({f"x_{seed}": x, f"train_{seed}": train,
                        f"test_{seed}": test, f"baseline_{seed}": baseline})
        print(f"seed={seed} test={len(test)} baseline_ap={ap:.9f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=REPO / "oulad-sem/data/processed/features.parquet")
    parser.add_argument("--scores", type=Path,
                        default=REPO / "oulad-sem/data/processed/features_scores.parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.features, args.scores, args.output)
