"""Helpers for the specification curve, matching alternatives and negative controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import make_model, metrics_at, select_threshold


def group_split(df: pd.DataFrame, seed: int):
    groups = df["id_student"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr, temp = next(gss.split(df, np.zeros(len(df)), groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    va_rel, te_rel = next(gss2.split(temp.reshape(-1, 1), np.zeros(len(temp)),
                                     groups[temp]))
    return tr, temp[va_rel], temp[te_rel]


def evaluate(df: pd.DataFrame, cols: list[str], kind: str, seed: int,
             target_precision: float = 0.80):
    tr, va, te = group_split(df, seed)
    y = df["y"].to_numpy(dtype=int)
    X = df[cols].to_numpy(dtype=float)
    if len(np.unique(y[tr])) < 2 or len(te) == 0:
        return None
    if kind == "rf":
        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                random_state=seed, n_jobs=1)),
        ])
    else:
        model = make_model("logistic", seed)
    model.fit(X[tr], y[tr])
    pv = model.predict_proba(X[va])[:, 1]
    thr, _, _, _ = select_threshold(y[va], pv, target_precision)
    p = model.predict_proba(X[te])[:, 1]
    m = metrics_at(y[te], p, thr)
    m["n_train"] = int(len(tr))
    return m


def _coarsen(df, cols, n_bins=5):
    out = df.copy()
    pres = ["code_module", "code_presentation"]
    for c in cols:
        out[c + "_bin"] = out.groupby(pres, observed=True)[c].transform(
            lambda s: pd.qcut(s.rank(method="first"),
                              max(2, min(n_bins, len(s) // 2)),
                              labels=False, duplicates="drop"))
    return out


def stratified_effect(df: pd.DataFrame, cols: list[str], metric: str):
    d = _coarsen(df, cols)
    bins = [c + "_bin" for c in cols]
    d["stratum"] = d.groupby(["code_module", "code_presentation"] + bins,
                             observed=True).ngroup()
    g = d.groupby("stratum", observed=True)["y"].agg(["size", "sum"])
    g["n_t"] = g["sum"].astype(int); g["n_c"] = g["size"].astype(int) - g["n_t"]
    keep = g.index[(g.n_t > 0) & (g.n_c > 0)]
    m = d[d["stratum"].isin(keep)].copy()
    if m.empty:
        return float("nan"), 0
    cnt = m.groupby("stratum", observed=True)["y"].agg(["size", "sum"])
    cnt["n_t"] = cnt["sum"].astype(int); cnt["n_c"] = cnt["size"].astype(int) - cnt["n_t"]
    m = m.merge((cnt.n_t / cnt.n_c).rename("cw"), left_on="stratum",
                right_index=True, how="left")
    m["w"] = np.where(m["y"] == 1, 1.0, m["cw"])
    t = m["y"].to_numpy() == 1
    w = m["w"].to_numpy()
    x = m[metric].to_numpy(dtype=float)
    mt = np.sum(w[t] * x[t]) / max(w[t].sum(), 1e-9)
    mc = np.sum(w[~t] * x[~t]) / max(w[~t].sum(), 1e-9)
    return float(mt - mc), int(len(m))


def psm_effect(df: pd.DataFrame, cols: list[str], metric: str, caliper=0.05):
    d = df.copy()
    pres = ["code_module", "code_presentation"]
    dummies = pd.get_dummies(d["code_presentation"], prefix="pres", drop_first=True)
    X = np.column_stack([d[cols].to_numpy(dtype=float), dummies.to_numpy(dtype=float)])
    y = d["y"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), 0
    ps = make_model("logistic", 42)
    ps.fit(X, y)
    d["ps"] = ps.predict_proba(X)[:, 1]
    pairs = []
    for _, g in d.groupby(pres, observed=True):
        t = g[g["y"] == 1]; c = g[g["y"] == 0]
        if t.empty or c.empty:
            continue
        used = set()
        for _, row in t.sort_values("ps", ascending=False).iterrows():
            cand = c[~c.index.isin(used)]
            if cand.empty:
                break
            dist = (cand["ps"] - row["ps"]).abs()
            j = dist.idxmin()
            if dist.loc[j] <= caliper:
                used.add(j)
                pairs.append((row[metric], cand.loc[j, metric]))
    if not pairs:
        return float("nan"), 0
    arr = np.array(pairs, dtype=float)
    return float(np.mean(arr[:, 0] - arr[:, 1])), len(pairs)


def cramers_v(a, b) -> float:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
