"""Model ladder, held-out metrics, calibration and bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logistic":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                       random_state=seed)),
        ])
    if kind == "histgb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, class_weight="balanced",
                random_state=seed)),
        ])
    raise ValueError(kind)


def select_threshold(y, p, target_precision: float = 0.80):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5, float("nan"), float("nan"), "single_class"
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, float("nan"), float("nan"), "no_thresholds"
    pv, rv = prec[:-1], rec[:-1]
    mask = pv >= target_precision
    if mask.any():
        idx = np.argmax(np.where(mask, rv, -1))
        return float(thr[idx]), float(pv[idx]), float(rv[idx]), "target_precision"
    f1 = 2 * pv * rv / np.maximum(pv + rv, 1e-12)
    idx = int(np.argmax(f1))
    return float(thr[idx]), float(pv[idx]), float(rv[idx]), "max_f1_fallback"


def calibration(y, p, bins: int = 10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan,
                "ece": np.nan, "brier": np.nan}
    brier = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan")
    # slope/intercept of y ~ logit(p)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q)).reshape(-1, 1)
    if len(np.unique(y)) > 1 and np.std(logit) > 1e-12:
        lr = LogisticRegression(C=1e6, max_iter=2000)
        lr.fit(logit, y)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    else:
        slope = intercept = float("nan")
    # equal-width ECE
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            ece += (m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return {"calibration_slope": slope, "calibration_intercept": intercept,
            "ece": float(ece), "brier": brier}


def metrics_at(y, p, threshold: float) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    out = {"n": int(len(y)), "positive_rate": float(np.mean(y)) if len(y) else np.nan,
           "threshold": float(threshold)}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"pr_auc": np.nan, "roc_auc": np.nan, "precision": np.nan,
                    "recall": np.nan, "f1": np.nan, "brier": np.nan,
                    "calibration_slope": np.nan, "calibration_intercept": np.nan,
                    "ece": np.nan})
        return out
    pred = (p >= threshold).astype(int)
    out["pr_auc"] = float(average_precision_score(y, p))
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out.update(calibration(y, p))
    return out


def _fast_metrics(y, p, threshold, bins=10):
    """Bootstrap metrics without the per-resample calibration-slope logistic fit."""
    pred = (p >= threshold).astype(int)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    out["ece"] = float(ece)
    return out


def bootstrap_metrics(y, p, threshold: float, n_boot: int = 1000, seed: int = 42):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    rng = np.random.default_rng(seed)
    # The calibration slope is reported as a point estimate; bootstrapping it
    # would require a logistic fit on every resample and dominates runtime.
    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "ece"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = _fast_metrics(yy, pp, threshold)
        for k in keys:
            if np.isfinite(m.get(k, np.nan)):
                vals[k].append(m[k])
    ci = {}
    for k in keys:
        if vals[k]:
            ci[k] = (float(np.percentile(vals[k], 2.5)),
                     float(np.percentile(vals[k], 97.5)))
        else:
            ci[k] = (np.nan, np.nan)
    return ci
