"""Grouped holdout, train-only threshold selection and paired student bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CONCENTRATION_FEATURES

CONTEXT = ["code_module", "code_presentation", "module_presentation"]


def _pipeline(columns: list[str], algorithm: str, seed: int) -> Pipeline:
    numeric = [c for c in columns if c not in CONTEXT]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                     ("scale", StandardScaler())]), numeric))
    categorical = [c for c in columns if c in CONTEXT]
    if categorical:
        transformers.append(("context", OneHotEncoder(handle_unknown="ignore"), categorical))
    model = (LogisticRegression(max_iter=1000, random_state=seed) if algorithm == "lr" else
             RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                    n_jobs=4, random_state=seed))
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    return {"precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")}


def _threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.01, 0.99, np.quantile(scores, np.linspace(0.02, 0.98, 97))])
    values = [f1_score(y, scores >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def _paired_ci(y: np.ndarray, groups: np.ndarray, scores: np.ndarray, threshold: float,
               baseline_scores: np.ndarray, baseline_threshold: float, seed: int, repeats: int = 500) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    students = np.unique(groups)
    by_student = {student: np.flatnonzero(groups == student) for student in students}
    differences = {metric: [] for metric in ("f1", "average_precision")}
    for _ in range(repeats):
        chosen = rng.choice(students, size=len(students), replace=True)
        indices = np.concatenate([by_student[student] for student in chosen])
        if len(np.unique(y[indices])) < 2:
            continue
        current = _metrics(y[indices], scores[indices], threshold)
        baseline = _metrics(y[indices], baseline_scores[indices], baseline_threshold)
        for metric in differences:
            differences[metric].append(current[metric] - baseline[metric])
    return {metric: [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [float("nan")] * 2
            for metric, values in differences.items()}


def evaluate_models(frame: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    population = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].copy().reset_index(drop=True)
    population["module_presentation"] = population.code_module + "_" + population.code_presentation
    population["log_clicks_squared"] = population.log_clicks ** 2
    y = (population.final_result == "Fail").astype(int).to_numpy()
    groups = population.id_student.to_numpy()
    if len(np.unique(groups)) < 12 or min(np.bincount(y)) < 6:
        raise ValueError("Need at least 12 students and six samples per class for grouped evaluation")
    split = None
    for train, test in GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=seed).split(population, y, groups):
        if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2:
            split = train, test
            break
    if split is None:
        raise ValueError("Could not create a grouped holdout containing both classes")
    train, test = split
    assert not set(groups[train]).intersection(groups[test])
    resources = sorted(c for c in population if c.startswith("resource_share__"))
    time = ["active_days", "longest_inactive_run", "daily_concentration", "daily_entropy"] + [f"week_{i}_share" for i in range(3)]
    resource = ["resource_type_count", "resource_entropy"] + resources[:-1]
    sets = {"click_only_lr": ["log_clicks"],
            "baseline_lr": ["log_clicks"] + CONTEXT,
            "nonlinear_baseline_lr": ["log_clicks", "log_clicks_squared", "module_presentation"],
            "full_nonlinear_lr": ["log_clicks", "log_clicks_squared"] + time + resource + ["module_presentation"],
            "concentration_full_lr": ["log_clicks", "log_clicks_squared"] + time + resource + CONCENTRATION_FEATURES + ["module_presentation"],
            "time_lr": ["log_clicks"] + time + CONTEXT,
            "resource_lr": ["log_clicks"] + resource + CONTEXT,
            "full_lr": ["log_clicks"] + time + resource + CONTEXT,
            "baseline_rf": ["log_clicks"] + CONTEXT,
            "full_rf": ["log_clicks"] + time + resource + CONTEXT}
    results = []
    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    heldout = population.iloc[test][["code_module", "code_presentation", "id_student", "final_result"]].copy()
    for name, columns in sets.items():
        algorithm = "rf" if name.endswith("rf") else "lr"
        model = _pipeline(columns, algorithm, seed)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        oof = cross_val_predict(model, population.iloc[train][columns], y[train], groups=groups[train],
                                cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold = _threshold(y[train], oof)
        model.fit(population.iloc[train][columns], y[train])
        probabilities = model.predict_proba(population.iloc[test][columns])[:, 1]
        scores[name] = probabilities
        thresholds[name] = threshold
        result = {"model": name, "threshold_from_training": threshold, "train_rows": len(train),
                  "test_rows": len(test), "test_students": len(np.unique(groups[test])),
                  "fail_prevalence": float(y[test].mean()), "features": columns,
                  **_metrics(y[test], probabilities, threshold)}
        results.append(result)
        heldout[f"score_{name}"] = probabilities
        heldout[f"prediction_{name}"] = (probabilities >= threshold).astype(int)

    for result in results:
        name = result["model"]
        comparator = ("full_nonlinear_lr" if name == "concentration_full_lr" else
                      "baseline_rf" if name.endswith("rf") else ("nonlinear_baseline_lr" if name in ("nonlinear_baseline_lr", "full_nonlinear_lr") else "baseline_lr"))
        result["comparator"] = comparator
        result["delta_vs_controlled_baseline_ci"] = _paired_ci(
            y[test], groups[test], scores[name], thresholds[name], scores[comparator], thresholds[comparator], seed)
        result["delta_average_precision"] = result["average_precision"] - next(r["average_precision"] for r in results if r["model"] == comparator)
    (output_dir / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    heldout.to_parquet(output_dir / "heldout_predictions.parquet", index=False)
    presentation_rows = []
    for (module, presentation), indices in heldout.groupby(["code_module", "code_presentation"]).indices.items():
        local_y = y[test][indices]
        if len(np.unique(local_y)) != 2:
            continue
        for name in sets:
            presentation_rows.append({"module": module, "presentation": presentation, "model": name,
                                      "n": len(indices), "fail_rate": float(local_y.mean()),
                                      **_metrics(local_y, scores[name][indices], thresholds[name])})
    pd.DataFrame(presentation_rows).to_csv(output_dir / "presentation_metrics.csv", index=False)

    # Separate robustness checks use training/holdout pairs, never the published holdout.
    robustness = []
    for repeat in range(5):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + 101 + repeat).split(population, y, groups))
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            continue
        row = {"split": f"group_{repeat}", "train_rows": len(a), "test_rows": len(b)}
        for name in ("nonlinear_baseline_lr", "full_nonlinear_lr", "concentration_full_lr"):
            m = _pipeline(sets[name], "lr", seed)
            m.fit(population.iloc[a][sets[name]], y[a])
            row[name] = float(average_precision_score(y[b], m.predict_proba(population.iloc[b][sets[name]])[:,1]))
        row["delta_ap"] = row["full_nonlinear_lr"] - row["nonlinear_baseline_lr"]
        row["delta_concentration_ap"] = row["concentration_full_lr"] - row["full_nonlinear_lr"]
        robustness.append(row)
    train_year = np.flatnonzero(population.code_presentation.str.startswith("2013").to_numpy())
    test_year = np.flatnonzero(population.code_presentation.str.startswith("2014").to_numpy() & ~np.isin(groups, groups[train_year]))
    if len(train_year) and len(test_year) and len(np.unique(y[train_year])) == 2 and len(np.unique(y[test_year])) == 2:
        row = {"split": "2013_to_2014", "train_rows": len(train_year), "test_rows": len(test_year)}
        for name in ("nonlinear_baseline_lr", "full_nonlinear_lr"):
            m = _pipeline(sets[name], "lr", seed)
            m.fit(population.iloc[train_year][sets[name]], y[train_year])
            row[name] = float(average_precision_score(y[test_year], m.predict_proba(population.iloc[test_year][sets[name]])[:,1]))
        row["delta_ap"] = row["full_nonlinear_lr"] - row["nonlinear_baseline_lr"]
        robustness.append(row)
    pd.DataFrame(robustness).to_csv(output_dir / "robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ("baseline_lr", "nonlinear_baseline_lr", "full_nonlinear_lr", "concentration_full_lr", "time_lr", "resource_lr", "full_lr", "baseline_rf", "full_rf"):
        precision, recall, _ = precision_recall_curve(y[test], scores[name])
        ax.plot(recall, precision, label=name)
    ax.axhline(y[test].mean(), color="gray", linestyle="--", label="Fail prevalence")
    ax.set(xlabel="Fail recall", ylabel="Fail precision", title="Unmatched grouped holdout PR curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["model"] for r in results], [r["average_precision"] for r in results])
    ax.axhline(y[test].mean(), color="gray", linestyle="--")
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="Average precision", title="Feature ablation on one held-out set")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation.png", dpi=160)
    plt.close(fig)
