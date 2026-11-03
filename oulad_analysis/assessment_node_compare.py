"""Compare 2/3/4-week TMA cutoffs on a fixed retrospective common cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import KEY, SEEDS, SUBMIT, TRACK, _fit_predict


def compare_common_cohort(run_dirs: dict[int, Path]) -> pd.DataFrame:
    frames = {weeks: pd.read_parquet(path / "node_features.parquet") for weeks, path in run_dirs.items()}
    if set(frames) != {2, 3, 4}:
        raise ValueError("Supply 2, 3, and 4 week runs")
    binary = {weeks: frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].copy()
              for weeks, frame in frames.items()}
    key_sets = {}
    for weeks, frame in binary.items():
        if frame.duplicated(KEY).any():
            raise ValueError(f"Duplicate enrollment key in week {weeks}")
        key_sets[weeks] = set(map(tuple, frame[KEY].itertuples(index=False, name=None)))
    common = set.intersection(*key_sets.values())
    if not common:
        raise ValueError("No common binary cohort")
    ordered = sorted(common)
    cohorts = {}
    for weeks, frame in binary.items():
        indexed = frame.set_index(KEY).loc[ordered].reset_index()
        cohorts[weeks] = indexed
    for weeks in (3, 4):
        if not cohorts[2].final_result.equals(cohorts[weeks].final_result):
            raise ValueError("Outcome changed across windows")
        if not cohorts[2].id_student.equals(cohorts[weeks].id_student):
            raise AssertionError("Common-cohort order differs")

    labels = cohorts[2].final_result.eq("Fail")
    students = cohorts[2].id_student
    baseline = ["module_presentation", "log_clicks", "log_clicks_squared", "pre_clicks", "post_clicks"] + SUBMIT
    rows = []
    for seed in SEEDS:
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(cohorts[2], labels, students))
        assert set(students.iloc[left]).isdisjoint(set(students.iloc[right]))
        for weeks, frame in cohorts.items():
            train, test = frame.iloc[left], frame.iloc[right]
            y = test.final_result.eq("Fail").astype(int)
            for model, columns in [("B2_clicks_submission", baseline),
                                   ("B3_clicks_submission_timing", baseline + TRACK)]:
                scores = _fit_predict(train, test, columns)
                rows.append({"seed": seed, "post_weeks": weeks, "model": model,
                             "common_cohort_rows": len(frame), "train_rows": len(train), "test_rows": len(test),
                             "test_students": test.id_student.nunique(), "fail_prevalence": float(y.mean()),
                             "average_precision": float(average_precision_score(y, scores)),
                             "brier": float(brier_score_loss(y, scores))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two", type=Path, required=True, help="Two-week run directory")
    parser.add_argument("--three", type=Path, required=True, help="Three-week run directory")
    parser.add_argument("--four", type=Path, required=True, help="Four-week run directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    result = compare_common_cohort({2: args.two, 3: args.three, 4: args.four})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
