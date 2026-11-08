"""Model ladder, calibration, permutation control and parity checks.

``core_fit`` reproduces the TMA branch estimator exactly (median imputation,
standardised numeric block, one-hot ``module_presentation``, unweighted logistic
regression) so parity numbers can be compared with the published tables.  The
report ladder keeps the same estimator family as the frozen pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.modeling import make_model, metrics_at, select_threshold

from .core import CONTEXT, core_fit, ladder_fit

# The published TMA two-week ladder uses log click volume and its square, plus
# the raw pre-window volume controls; the same columns keep the parity
# comparison like-for-like with results/tma_two_week_20261103/models.csv.
PRE = ["log_pre_clicks", "pre_active_days", "last3_share"]
POST_VOLUME = ["log_post_clicks", "log_post_clicks_squared"]
SUBMIT = ["submitted_by_cutoff", "submitted_after_due"]
DISTANCE = ["days_early", "days_late"]
TIMING = ["post_active_days", "post_both_weeks", "post_longest_inactive_run",
          "post_week2_share", "post_peak7_share", "post_entropy_norm",
          "post_first_active_day"]
SCORE = ["score", "has_score_by_cutoff"]

LADDER = {
    "M0": PRE + POST_VOLUME + SUBMIT,
    "M1": PRE + POST_VOLUME + SUBMIT + TIMING,
    "M2": PRE + POST_VOLUME + SUBMIT + TIMING + DISTANCE,
    "M3": PRE + POST_VOLUME + SUBMIT + TIMING + DISTANCE,
    "M4": PRE + POST_VOLUME + SUBMIT + TIMING + DISTANCE + SCORE,
}
PARITY_SEEDS = [42, 143, 144, 145, 146]


def resource_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame if c.startswith("post_resource_share__"))


def columns_for(frame: pd.DataFrame, model: str) -> list[str]:
    cols = list(LADDER[model])
    if model == "M3":
        cols += ["post_resource_type_count", "post_resource_entropy"]
        cols += resource_columns(frame)
    return [c for c in cols if c in frame]


def _fit(frame: pd.DataFrame, columns: list[str], train: np.ndarray,
         test: np.ndarray, kind: str, seed: int, val: np.ndarray | None = None,
         precision_target: float = 0.80):
    prob = ladder_fit(frame, columns, train, test, seed=seed, kind=kind)
    if prob is None:
        return None, np.nan, "no_fit"
    y = frame["y_fail"].to_numpy(dtype=int)
    threshold, rule = np.nan, "no_validation"
    if val is not None and len(val) and len(np.unique(y[val])) == 2:
        fitted = _ladder_model(frame, columns, train, kind, seed)
        pv = fitted.predict_proba(frame.iloc[val][columns + CONTEXT])[:, 1]
        threshold, _, _, rule = select_threshold(y[val], pv, precision_target)
        refit = _ladder_model(frame, columns, np.concatenate([train, val]), kind, seed)
        prob = refit.predict_proba(frame.iloc[test][columns + CONTEXT])[:, 1]
    return prob, threshold, rule


def _ladder_model(frame: pd.DataFrame, columns: list[str], rows: np.ndarray,
                  kind: str, seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if kind == "logistic":
        transform = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             columns),
            ("context", OneHotEncoder(handle_unknown="ignore"), CONTEXT)])
        model = make_pipeline(transform, LogisticRegression(max_iter=2000,
                                                           class_weight="balanced",
                                                           random_state=seed))
    else:
        transform = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), columns),
            ("context", OneHotEncoder(handle_unknown="ignore"), CONTEXT)])
        model = make_pipeline(transform, HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=seed))
    model.fit(frame.iloc[rows][columns + CONTEXT],
              frame.iloc[rows]["y_fail"].to_numpy(dtype=int))
    return model


def _row(model, split, seed, y, p, thr, rule, n_train, n_features, permutation=0):
    prev = float(np.mean(y)) if len(y) else float("nan")
    out = {"model": model, "split": split, "seed": int(seed), "n_test": int(len(y)),
           "n_train": int(n_train), "n_features": int(n_features),
           "no_skill_pr_auc": prev,
           "all_positive_f1": (2 * prev / (1 + prev)) if np.isfinite(prev) else float("nan"),
           "threshold_rule": rule,
           "threshold": float(thr) if np.isfinite(thr) else np.nan,
           "permutation": int(permutation)}
    if p is None or len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"average_precision": np.nan, "roc_auc": np.nan, "brier": np.nan,
                    "precision": np.nan, "recall": np.nan, "f1": np.nan,
                    "calibration_slope": np.nan, "calibration_intercept": np.nan,
                    "ece": np.nan})
        return out
    m = metrics_at(y, p, thr if np.isfinite(thr) else 0.5)
    out.update({"average_precision": float(average_precision_score(y, p)),
                "roc_auc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "calibration_slope": m["calibration_slope"],
                "calibration_intercept": m["calibration_intercept"],
                "ece": m["ece"]})
    return out


def _holdout(train_idx: np.ndarray, frame: pd.DataFrame, seed: int):
    if len(train_idx) < 30:
        return train_idx, np.array([], dtype=int)
    y = frame["y_fail"].to_numpy(dtype=int)[train_idx]
    groups = frame["id_student"].to_numpy()[train_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    a, b = next(gss.split(frame.iloc[train_idx], y, groups))
    return train_idx[a], train_idx[b]


def evaluate(frame: pd.DataFrame, splits: dict, *, seed: int, models: list[str],
             rf_models: list[str], precision_target: float) -> pd.DataFrame:
    rows = []
    y = frame["y_fail"].to_numpy(dtype=int)
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    for model in models + [f"RF_{m}" for m in rf_models]:
        base = model[3:] if model.startswith("RF_") else model
        cols = columns_for(frame, base)
        kind = "histgb" if model.startswith("RF_") else "logistic"
        p, thr, rule = _fit(frame, cols, tr, te, kind, seed, va, precision_target)
        rows.append(_row(model, "main_test", seed, y[te], p, thr, rule,
                         len(tr) + len(va), len(cols)))

    for fold in splits["group_kfold"]:
        tr_f = np.array(fold["train"])
        te_f = np.array(fold["test"])
        tr80, va20 = _holdout(tr_f, frame, seed + 2)
        for model in models:
            cols = columns_for(frame, model)
            p, thr, rule = _fit(frame, cols, tr80, te_f, "logistic", seed, va20,
                                precision_target)
            rows.append(_row(model, f"group_kfold_{fold['fold']}", seed, y[te_f], p,
                             thr, rule, len(tr80) + len(va20), len(cols)))

    for name, key, tr_seed, te_seed in (
            ("temporal", "temporal", seed + 3, seed + 3),
            ("lomo", "lomo", seed + 4, seed + 4)):
        entries = [splits["temporal"]] if key == "temporal" else splits["lomo"]
        for entry in entries:
            tr_e = np.array(entry["train"])
            te_e = np.array(entry["test"])
            tr80, va20 = _holdout(tr_e, frame, tr_seed)
            label = name if key == "temporal" else f"lomo_{entry['module']}"
            for model in models:
                cols = columns_for(frame, model)
                p, thr, rule = _fit(frame, cols, tr80, te_e, "logistic", seed, va20,
                                    precision_target)
                rows.append(_row(model, label, seed, y[te_e], p, thr, rule,
                                 len(tr80) + len(va20), len(cols)))
    return pd.DataFrame(rows)


def calibration_curve_table(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not m.any():
            continue
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def label_permutation(frame: pd.DataFrame, splits: dict, model: str, *, seed: int,
                      repeats: int, precision_target: float) -> pd.DataFrame:
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    cols = columns_for(frame, model)
    rng = np.random.default_rng(seed + 9091)
    rows = []
    base = frame.copy()
    for i in range(repeats):
        base["y_fail"] = rng.permutation(frame["y_fail"].to_numpy())
        p, thr, rule = _fit(base, cols, tr, te, "logistic", seed + i, va,
                            precision_target)
        if p is None:
            continue
        y = base["y_fail"].to_numpy()[te]
        rows.append({"permutation": i,
                     "average_precision": float(average_precision_score(y, p)),
                     "fail_prevalence": float(y.mean())})
    return pd.DataFrame(rows)


def parity(frame: pd.DataFrame, *, precision_target: float) -> pd.DataFrame:
    """80/20 grouped holdout with the TMA-branch seeds and estimator."""
    rows = []
    for split_seed in PARITY_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)
        tr, te = next(gss.split(frame, frame["y_fail"],
                                frame["id_student"].to_numpy()))
        for model in ("M0", "M1", "M2"):
            cols = columns_for(frame, model)
            p = core_fit(frame, cols, tr, te)
            y = frame["y_fail"].to_numpy()[te]
            rows.append({"seed": split_seed, "model": model,
                         "n_test": int(len(te)),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, p)),
                         "brier": float(brier_score_loss(y, p))})
    return pd.DataFrame(rows)
