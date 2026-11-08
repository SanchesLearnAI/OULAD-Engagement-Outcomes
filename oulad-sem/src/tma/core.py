"""Model fitting shared by S08, with a faithful parity port of the TMA branch.

``core_fit`` reproduces ``oulad_analysis.assessment_node._fit_predict`` and
``_model`` exactly - median imputation, standardisation of the numeric block,
one-hot encoding of ``module_presentation``, and an unweighted
``LogisticRegression(max_iter=1000)`` - so a parity run can be compared with the
published TMA tables.  ``ladder_fit`` is the same estimator used for the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONTEXT = ["module_presentation"]


def core_model(numeric: list[str], context: list[str] = CONTEXT):
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         numeric),
        ("context", OneHotEncoder(handle_unknown="ignore"), context)])
    return make_pipeline(transform, LogisticRegression(max_iter=1000))


def ladder_model(numeric: list[str], context: list[str] = CONTEXT, seed: int = 42):
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         numeric),
        ("context", OneHotEncoder(handle_unknown="ignore"), context)])
    return make_pipeline(transform, LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed))


def core_fit(frame: pd.DataFrame, numeric: list[str], train: np.ndarray,
             test: np.ndarray, *, label: str = "y_fail") -> np.ndarray:
    """Predict with the TMA-branch specification (no class weighting)."""
    cols = [c for c in numeric if c in frame]
    context = [c for c in CONTEXT if c in frame]
    model = core_model(cols, context)
    model.fit(frame.iloc[train][cols + context],
              frame.iloc[train]["final_result"].eq("Fail").astype(int))
    return model.predict_proba(frame.iloc[test][cols + context])[:, 1]


def ladder_fit(frame: pd.DataFrame, numeric: list[str], train: np.ndarray,
               test: np.ndarray, *, seed: int, kind: str = "logistic",
               label: str = "y_fail") -> np.ndarray | None:
    cols = [c for c in numeric if c in frame]
    context = [c for c in CONTEXT if c in frame]
    y = frame[label].to_numpy(dtype=int)
    if len(train) < 20 or len(np.unique(y[train])) < 2:
        return None
    if kind == "logistic":
        model = ladder_model(cols, context, seed)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        transform = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), cols),
            ("context", OneHotEncoder(handle_unknown="ignore"), context)])
        model = make_pipeline(transform, HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=seed))
    model.fit(frame.iloc[train][cols + context], y[train])
    return model.predict_proba(frame.iloc[test][cols + context])[:, 1]
