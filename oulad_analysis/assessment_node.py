"""Experiment 2, 3, or 4 weeks after the first weighted TMA deadline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]
SEEDS = (42, 143, 144, 145, 146)
TRACK = ["pre_active_days", "post_active_days", "last3_share", "post_both_weeks",
         "post_active_week_fraction", "post_minus_pre_log"]
SUBMIT = ["submitted_by_cutoff", "submitted_after_due"]


def _read(path: Path, *, na_values: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, na_values=na_values)


def _first_weighted_tma(assessments: pd.DataFrame) -> pd.DataFrame:
    a = assessments.copy()
    a = a[a.assessment_type.eq("TMA")].copy()
    a["date"] = pd.to_numeric(a.date, errors="raise")
    a["weight"] = pd.to_numeric(a.weight, errors="raise")
    a = a[a.weight.gt(0)].copy()
    a = a.sort_values(BATCH + ["date", "id_assessment"])
    first = a.drop_duplicates(BATCH)
    if first.duplicated(BATCH).any() or first.date.isna().any():
        raise ValueError("Invalid first weighted TMA definition")
    return first[BATCH + ["id_assessment", "date"]].rename(columns={"date": "due_day"})


def build_node_features(data_dir: Path, *, post_weeks: int = 2, chunksize: int = 500_000) -> tuple[pd.DataFrame, dict]:
    if post_weeks not in (2, 3, 4):
        raise ValueError("post_weeks must be 2, 3, or 4")
    post_days = 7 * post_weeks
    info = _read(data_dir / "studentInfo.csv")
    registration = _read(data_dir / "studentRegistration.csv", na_values=["?"])
    assessment = _first_weighted_tma(_read(data_dir / "assessments.csv", na_values=["?"]))
    submission = _read(data_dir / "studentAssessment.csv", na_values=["?"])
    if info.duplicated(KEY).any() or registration.duplicated(KEY).any():
        raise ValueError("Student or registration keys are not unique")
    if submission.duplicated(["id_assessment", "id_student"]).any():
        raise ValueError("Assessment submission keys are not unique")
    if not info.final_result.isin(["Pass", "Distinction", "Fail", "Withdrawn"]).all():
        raise ValueError("Unknown final result")
    frame = info.merge(registration, on=KEY, validate="one_to_one", how="left")
    frame = frame.merge(assessment, on=BATCH, validate="many_to_one", how="inner")
    frame = frame.merge(submission[["id_assessment", "id_student", "date_submitted", "is_banked"]],
                        on=["id_assessment", "id_student"], validate="one_to_one", how="left")
    eligible = frame.date_registration.notna() & frame.date_registration.le(frame.due_day)
    eligible &= frame.date_unregistration.isna() | frame.date_unregistration.gt(frame.due_day + post_days)
    eligible &= ~frame.is_banked.eq(1)
    stats = {"source_rows": int(len(info)), "weighted_tma_presentations": int(len(assessment)),
             "with_first_weighted_tma": int(len(frame)), "eligible_at_cutoff": int(eligible.sum()),
             "excluded_banked": int(frame.is_banked.eq(1).sum())}
    frame = frame.loc[eligible, KEY + ["final_result", "due_day", "date_submitted"]].copy()
    if frame.empty:
        raise ValueError("No eligible records")
    frame["cutoff_day"] = frame.due_day + post_days
    frame["submitted_by_cutoff"] = frame.date_submitted.le(frame.cutoff_day).astype(int)
    frame["submitted_after_due"] = (frame.date_submitted.gt(frame.due_day) &
                                     frame.date_submitted.le(frame.cutoff_day)).astype(int)

    daily_parts = []
    lookup = frame[KEY + ["due_day"]]
    for chunk in pd.read_csv(data_dir / "studentVle.csv", usecols=KEY + ["date", "sum_click"], chunksize=chunksize):
        if chunk[KEY + ["date", "sum_click"]].isna().any().any() or chunk.sum_click.lt(0).any():
            raise ValueError("Invalid VLE event")
        part = chunk.merge(lookup, on=KEY, validate="many_to_one", how="inner")
        part["relative_day"] = part.date - part.due_day
        part = part[part.relative_day.between(-13, post_days)]
        if not part.empty:
            daily_parts.append(part.groupby(KEY + ["relative_day"], as_index=False).sum_click.sum())
    if daily_parts:
        daily = pd.concat(daily_parts).groupby(KEY + ["relative_day"], as_index=False).sum_click.sum()
    else:
        daily = pd.DataFrame(columns=KEY + ["relative_day", "sum_click"])
    if not daily.relative_day.between(-13, post_days).all():
        raise AssertionError("Event outside observation window")

    for name, lo, hi in [("pre", -13, 0), ("post", 1, post_days), ("last3", -2, 0),
                          ("post_week1", 1, 7), ("post_week2", 8, 14)]:
        part = daily[daily.relative_day.between(lo, hi)]
        clicks = part.groupby(KEY).sum_click.sum().rename(name + "_clicks")
        active = part[part.sum_click.gt(0)].groupby(KEY).relative_day.nunique().rename(name + "_active_days")
        frame = frame.join(clicks, on=KEY).join(active, on=KEY)
        frame[[name + "_clicks", name + "_active_days"]] = frame[[name + "_clicks", name + "_active_days"]].fillna(0)
    frame["total_clicks"] = frame.pre_clicks + frame.post_clicks
    frame["log_clicks"] = np.log1p(frame.total_clicks)
    frame["log_clicks_squared"] = frame.log_clicks ** 2
    frame["last3_share"] = (frame.last3_clicks / frame.pre_clicks.replace(0, np.nan)).fillna(0)
    frame["post_both_weeks"] = (frame.post_week1_active_days.gt(0) & frame.post_week2_active_days.gt(0)).astype(int)
    post_positive = daily[daily.relative_day.between(1, post_days) & daily.sum_click.gt(0)].copy()
    post_positive["relative_week"] = ((post_positive.relative_day - 1) // 7).astype(int)
    active_weeks = post_positive.groupby(KEY).relative_week.nunique().rename("post_active_weeks")
    frame = frame.join(active_weeks, on=KEY)
    frame["post_active_weeks"] = frame.post_active_weeks.fillna(0).astype(int)
    frame["post_active_week_fraction"] = frame.post_active_weeks / post_weeks
    frame["post_minus_pre_log"] = np.log1p(frame.post_clicks) - np.log1p(frame.pre_clicks)
    frame["module_presentation"] = frame.code_module + "_" + frame.code_presentation
    stats["eligible_binary"] = int(frame.final_result.isin(["Pass", "Distinction", "Fail"]).sum())
    stats["eligible_withdrawn"] = int(frame.final_result.eq("Withdrawn").sum())
    stats["event_daily_rows"] = int(len(daily))
    stats["post_weeks"] = post_weeks
    stats["post_days"] = post_days
    # Future submission dates are needed only for the as-of-cutoff indicators.
    # Do not expose them in the saved feature table.
    return frame.drop(columns="date_submitted").reset_index(drop=True), stats


def _model(columns: list[str]) -> object:
    categorical = [c for c in columns if c in ("module_presentation", "code_module")]
    numeric = [c for c in columns if c not in categorical]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("context", OneHotEncoder(handle_unknown="ignore"), categorical)])
    return make_pipeline(transform, LogisticRegression(max_iter=1000))


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    model = _model(columns)
    model.fit(train[columns], train.final_result.eq("Fail").astype(int))
    return model.predict_proba(test[columns])[:, 1]


def evaluate_node(frame: pd.DataFrame, seed: int = 42, bootstrap: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].reset_index(drop=True)
    baseline = ["module_presentation", "log_clicks", "log_clicks_squared", "pre_clicks", "post_clicks"]
    versions = {"B0_clicks": baseline, "B1_clicks_timing": baseline + TRACK,
                "B2_clicks_submission": baseline + SUBMIT,
                "B3_clicks_submission_timing": baseline + SUBMIT + TRACK}
    rows, intervals, course_rows = [], [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(binary, groups=binary.id_student))
        train, test = binary.iloc[left], binary.iloc[right]
        assert set(train.id_student).isdisjoint(set(test.id_student))
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, cols) for name, cols in versions.items()}
        for name, probabilities in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train), "test_rows": len(test),
                         "test_students": test.id_student.nunique(), "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, probabilities)),
                         "brier": float(brier_score_loss(y, probabilities))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 42300)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))]) for _ in range(bootstrap)]
        for name, comparator in [("B1_clicks_timing", "B0_clicks"),
                                  ("B2_clicks_submission", "B0_clicks"),
                                  ("B3_clicks_submission_timing", "B2_clicks_submission")]:
            p, q = predictions[name], predictions[comparator]
            deltas = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                      for ix in draws if len(np.unique(y[ix])) == 2]
            intervals.append({"model": name, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975))})
            for module, ix in test.groupby("code_module").indices.items():
                if len(np.unique(y[ix])) == 2:
                    course_rows.append({"module": module, "model": name, "comparator": comparator, "n": len(ix),
                                        "delta_ap": float(average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix]))})
    return pd.DataFrame(rows), pd.DataFrame(intervals), pd.DataFrame(course_rows)


def evaluate_year_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    """2013 to 2014 check, with repeated students removed from later tests."""
    binary = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])]
    train = binary[binary.code_presentation.str.startswith("2013")]
    later = binary[binary.code_presentation.str.startswith("2014") & ~binary.id_student.isin(train.id_student)]
    baseline = ["code_module", "log_clicks", "log_clicks_squared", "pre_clicks", "post_clicks"] + SUBMIT
    rows = []
    for cohort, test in [("seen_modules", later[later.code_module.isin(train.code_module)]),
                         ("new_CCC", later[later.code_module.eq("CCC")])]:
        if train.empty or test.empty or test.final_result.eq("Fail").nunique() != 2:
            continue
        for name, columns in [("B2_clicks_submission", baseline),
                              ("B3_clicks_submission_timing", baseline + TRACK)]:
            scores = _fit_predict(train, test, columns)
            y = test.final_result.eq("Fail").astype(int)
            rows.append({"cohort": cohort, "model": name, "train_rows": len(train), "test_rows": len(test),
                         "fail_prevalence": float(y.mean()), "average_precision": float(average_precision_score(y, scores)),
                         "brier": float(brier_score_loss(y, scores))})
    return pd.DataFrame(rows)


def run(data_dir: Path, output_dir: Path, seed: int = 42, bootstrap: int = 1000, post_weeks: int = 2) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    required = ["studentInfo", "studentRegistration", "assessments", "studentAssessment", "studentVle"]
    for name in required:
        if not (data_dir / f"{name}.csv").is_file():
            raise FileNotFoundError(data_dir / f"{name}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, stats = build_node_features(data_dir, post_weeks=post_weeks)
    frame.to_parquet(output_dir / "node_features.parquet", index=False)
    metrics, intervals, course = evaluate_node(frame, seed, bootstrap)
    metrics.to_csv(output_dir / "node_models.csv", index=False)
    intervals.to_csv(output_dir / "node_ap_intervals.csv", index=False)
    course.to_csv(output_dir / "node_course_deltas.csv", index=False)
    evaluate_year_transfer(frame).to_csv(output_dir / "node_year_transfer.csv", index=False)
    stats.update({"seed": seed, "bootstrap": bootstrap,
                  "observation": f"first weighted TMA due_day-13 through due_day+{7*post_weeks} inclusive",
                  "label": f"Fail vs Pass/Distinction among records still registered at due_day+{7*post_weeks}; Withdrawn counted separately",
                  "sources_sha256": {name: hashlib.sha256((data_dir / f"{name}.csv").read_bytes()).hexdigest() for name in required}})
    (output_dir / "node_protocol.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--post-weeks", type=int, choices=(2, 3, 4), default=2)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.seed, args.bootstrap, args.post_weeks)


if __name__ == "__main__":
    main()
