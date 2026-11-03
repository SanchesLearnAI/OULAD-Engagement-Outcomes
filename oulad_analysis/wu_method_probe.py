"""Adapt Wu et al. HGB and calibration protocol to the TMA Fail task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .assessment_node import SUBMIT, TRACK

NUMERIC = ["log_clicks", "log_clicks_squared", "pre_clicks", "post_clicks"] + SUBMIT + TRACK
SEED = 42


def _pipeline(model, scale: bool) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))] +
                       ([("scale", StandardScaler())] if scale else []))
    pre = ColumnTransformer([("num", numeric, NUMERIC),
                             ("batch", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["module_presentation"])],
                            sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(scores)[::-1]
    top = order[:max(1, int(np.ceil(.1 * len(y))))]
    return {"ap": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "brier": float(brier_score_loss(y, scores)),
            "top10_ppv": float(y[top].mean()),
            "top10_recall": float(y[top].sum() / y.sum()),
            "top10_n": len(top)}


def run_probe(frame: pd.DataFrame, seed: int = SEED, bootstrap: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].reset_index(drop=True)
    if binary.empty or binary.duplicated(["code_module", "code_presentation", "id_student"]).any():
        raise ValueError("Expected unique binary enrollment records")
    y = binary.final_result.eq("Fail").astype(int).to_numpy()
    X = binary[NUMERIC + ["module_presentation"]]
    results, intervals = [], []
    for split_name, groups in [("student_holdout", binary.id_student.to_numpy()),
                               ("presentation_holdout", binary.module_presentation.to_numpy()),
                               ("presentation_no_repeated_students", binary.module_presentation.to_numpy())]:
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, y, groups))
        if split_name == "presentation_no_repeated_students":
            right = right[~binary.id_student.iloc[right].isin(binary.id_student.iloc[left]).to_numpy()]
        train, test = X.iloc[left], X.iloc[right]
        labels, truth = y[left], y[right]
        if len(np.unique(truth)) < 2:
            raise ValueError("Held-out split has only one outcome")
        common = {"split": split_name, "train_rows": len(left), "test_rows": len(right),
                  "train_groups": len(np.unique(groups[left])), "test_groups": len(np.unique(groups[right])),
                  "test_prevalence": float(truth.mean()),
                  "overlap_student_ids": int(len(set(binary.id_student.iloc[left]) & set(binary.id_student.iloc[right])))}
        models = {
            "project_lr": _pipeline(LogisticRegression(max_iter=1000, random_state=seed), True),
            "paper_weighted_lr": _pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed), True),
            "paper_hgb": _pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=6,
                                                                     random_state=seed), False)}
        predictions = {}
        for name, model in models.items():
            model.fit(train, labels)
            predictions[name] = model.predict_proba(test)[:, 1]
            results.append({**common, "model": name, **_metrics(truth, predictions[name])})
        # Follow the paper's 80/20 stratified calibration partition within training.
        fit_idx, cal_idx = train_test_split(np.arange(len(left)), test_size=.2, stratify=labels, random_state=seed)
        base = _pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=6,
                                                         random_state=seed), False)
        base.fit(train.iloc[fit_idx], labels[fit_idx])
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid")
        calibrated.fit(train.iloc[cal_idx], labels[cal_idx])
        predictions["paper_hgb_sigmoid"] = calibrated.predict_proba(test)[:, 1]
        results.append({**common, "model": "paper_hgb_sigmoid", **_metrics(truth, predictions["paper_hgb_sigmoid"])})
        resample_groups = binary.id_student.iloc[right].to_numpy() if split_name == "student_holdout" else groups[right]
        positions = list(pd.Series(np.arange(len(right))).groupby(resample_groups).apply(list))
        rng = np.random.default_rng(seed + 2026)
        draws = [np.concatenate([positions[i] for i in rng.integers(len(positions), size=len(positions))]) for _ in range(bootstrap)]
        for name, comparator in [("paper_hgb", "project_lr"),
                                  ("paper_hgb_sigmoid", "project_lr"),
                                  ("paper_hgb_sigmoid", "paper_hgb")]:
            p, q = predictions[name], predictions[comparator]
            differences = [average_precision_score(truth[ix], p[ix]) - average_precision_score(truth[ix], q[ix])
                           for ix in draws if len(np.unique(truth[ix])) == 2]
            intervals.append({"split": split_name, "model": name, "comparator": comparator,
                              "delta_ap": float(average_precision_score(truth, p) - average_precision_score(truth, q)),
                              "ci_low": float(np.quantile(differences, .025)),
                              "ci_high": float(np.quantile(differences, .975)),
                              "resample_unit": "student" if split_name == "student_holdout" else "presentation",
                              "bootstrap": bootstrap})
    return pd.DataFrame(results), pd.DataFrame(intervals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.features)
    post_days = frame.cutoff_day.sub(frame.due_day).unique()
    if len(post_days) != 1 or int(post_days[0]) not in (21, 28):
        parser.error("Expected one TMA cutoff at due day +21 or +28")
    result, intervals = run_probe(frame)
    result.to_csv(args.output_dir / "model_comparison.csv", index=False)
    intervals.to_csv(args.output_dir / "paired_ap_intervals.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({"source_features": str(args.features),
       "source_paper": "doi:10.1371/journal.pone.0352867", "author_code": "doi:10.5281/zenodo.20388848",
       "task": f"Fail vs Pass/Distinction at first weighted TMA +{int(post_days[0])} days; not a withdrawal replication",
       "source_features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
       "seed": SEED, "features": NUMERIC + ["module_presentation"],
       "caution": "Paper-style presentation split can repeat student IDs between partitions; overlap is reported."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
