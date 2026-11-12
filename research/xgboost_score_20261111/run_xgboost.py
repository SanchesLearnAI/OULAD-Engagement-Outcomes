"""Compare GPU XGBoost with the frozen score-inclusive logistic baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


SEEDS = (42, 143, 144, 145, 146)
CANDIDATES = ((3, .03), (5, .03), (3, .05))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def classifier(seed: int, depth: int, rate: float, iterations: int,
               early_stopping: bool) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=iterations, max_depth=depth, learning_rate=rate,
        tree_method="hist", device="cuda", subsample=.85,
        colsample_bytree=.85, reg_lambda=1.0, eval_metric="aucpr",
        early_stopping_rounds=35 if early_stopping else None,
        importance_type="gain", random_state=seed, n_jobs=4)


def assert_gpu(model: XGBClassifier) -> str:
    config = json.loads(model.get_booster().save_config())
    device = config["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost did not train on CUDA: {device}")
    return device


def bootstrap_interval(y, prediction, baseline, students) -> dict:
    members = [np.flatnonzero(students == student) for student in np.unique(students)]
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(400):
        chosen = np.concatenate([members[i] for i in
                                 rng.integers(len(members), size=len(members))])
        sampled_y = y[chosen]
        if 0 < sampled_y.sum() < len(sampled_y):
            deltas.append(average_precision_score(sampled_y, prediction[chosen]) -
                          average_precision_score(sampled_y, baseline[chosen]))
    return {"seed": 42, "delta_ap": average_precision_score(y, prediction) -
            average_precision_score(y, baseline),
            "ci_low": float(np.quantile(deltas, .025)),
            "ci_high": float(np.quantile(deltas, .975)),
            "bootstrap_repeats": 400, "resampling_unit": "id_student"}


def run(input_path: Path, output: Path) -> None:
    data = np.load(input_path)
    y, students = data["y"], data["student"]
    names = data["feature_names"].tolist()
    if len(y) != 19806 or int(y.sum()) != 6092 or len(names) != 38:
        raise ValueError("Input does not match the frozen score-inclusive cohort")
    output.mkdir(parents=True, exist_ok=True)
    metrics, importance = [], []
    interval = None
    observed_device = None
    for seed in SEEDS:
        train, test, baseline = [data[f"{part}_{seed}"] for part in
                                 ("train", "test", "baseline")]
        if set(students[train]) & set(students[test]):
            raise AssertionError("Student appears in both train and test")
        x = data[f"x_{seed}"]
        fit_position, val_position = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=seed + 1000)
            .split(x[train], groups=students[train]))
        fit, val = train[fit_position], train[val_position]
        options = []
        for depth, rate in CANDIDATES:
            candidate = classifier(seed, depth, rate, 400, early_stopping=True)
            candidate.fit(x[fit], y[fit], eval_set=[(x[val], y[val])], verbose=False)
            observed_device = assert_gpu(candidate)
            val_prediction = candidate.predict_proba(x[val])[:, 1]
            options.append((average_precision_score(y[val], val_prediction),
                            depth, rate, int(candidate.best_iteration) + 1))
        selected = max(options, key=lambda option: option[0])
        model = classifier(seed, selected[1], selected[2], selected[3],
                           early_stopping=False)
        model.fit(x[train], y[train])
        observed_device = assert_gpu(model)
        prediction = model.predict_proba(x[test])[:, 1]
        baseline_ap = average_precision_score(y[test], baseline)
        test_ap = average_precision_score(y[test], prediction)
        row = {
            "seed": seed, "n_test": len(test),
            "baseline_ap": baseline_ap, "xgboost_ap": test_ap,
            "delta_ap": test_ap - baseline_ap,
            "baseline_roc_auc": roc_auc_score(y[test], baseline),
            "xgboost_roc_auc": roc_auc_score(y[test], prediction),
            "baseline_brier": brier_score_loss(y[test], baseline),
            "xgboost_brier": brier_score_loss(y[test], prediction),
            "validation_ap": selected[0], "max_depth": selected[1],
            "learning_rate": selected[2], "n_estimators": selected[3],
        }
        metrics.append(row)
        importance.extend({"seed": seed, "feature": name,
                           "gain_importance": float(value)}
                          for name, value in zip(names, model.feature_importances_))
        if seed == 42:
            interval = bootstrap_interval(y[test], prediction, baseline,
                                          students[test])
        print(json.dumps(row), flush=True)

    save_csv(output / "metrics.csv", metrics)
    save_csv(output / "importance_by_seed.csv", importance)
    by_feature = {name: [r["gain_importance"] for r in importance
                         if r["feature"] == name] for name in names}
    summary = [
        {"feature": name, "mean_gain_importance": float(np.mean(values)),
         "min_gain_importance": float(np.min(values)),
         "max_gain_importance": float(np.max(values))}
        for name, values in by_feature.items()]
    summary.sort(key=lambda row: row["mean_gain_importance"], reverse=True)
    save_csv(output / "importance_summary.csv", summary)
    save_csv(output / "interval_seed42.csv", [interval])
    (output / "provenance.json").write_text(json.dumps({
        "baseline": "S_single_unsubmitted_x_inactive_run",
        "seeds": SEEDS, "positive_class": "Fail", "n_records": len(y),
        "score_masked_to_cutoff_submission": True,
        "selection": "inner student-group validation sklearn AP; final outer-train refit",
        "candidate_depth_rate": CANDIDATES,
        "inner_preprocessing": "outer-train median and standardization (original experiment)",
        "device": observed_device, "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "test_previously_inspected": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)
