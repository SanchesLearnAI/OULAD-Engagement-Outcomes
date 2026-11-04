"""K-means profile clustering with bootstrap stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, jaccard_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


def align_labels(base: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """K-means labels are permutation-invariant; align pred to base for Jaccard."""
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(base, pred):
        cm[int(a), int(b)] += 1
    row, col = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([mapping[int(x)] for x in pred], dtype=int)


def per_presentation_z(df: pd.DataFrame, cols: list[str],
                       pres: list[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(pres, observed=True)
    for c in cols:
        mean = g[c].transform("mean")
        std = g[c].transform("std").replace(0, 1).fillna(1)
        out[c] = (out[c] - mean) / std
    return out


def pca_reduce(X: np.ndarray, max_components: int = 8, seed: int = 42):
    p = PCA(n_components=min(max_components, X.shape[1]), random_state=seed)
    Z = p.fit_transform(X)
    return Z, p.explained_variance_ratio_.tolist()


def stability(X: np.ndarray, k_range, n_init: int, n_boot: int, seed: int,
              max_boot_n: int = 5000):
    # Cap the resample size on full data; all points are still predicted.
    rng = np.random.default_rng(seed)
    n = len(X)
    boot_n = min(n, max_boot_n)
    rows = []
    base = {}
    aris = {}
    for k in k_range:
        km = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=seed).fit(X)
        labels = km.labels_
        base[int(k)] = labels
        a = []
        j = []
        for b in range(n_boot):
            idx = rng.choice(n, size=boot_n, replace=False)
            kmb = KMeans(n_clusters=int(k), n_init=10,
                         random_state=seed + 1 + b).fit(X[idx])
            pred = kmb.predict(X)
            a.append(adjusted_rand_score(labels, pred))
            pred_aligned = align_labels(labels, pred, int(k))
            j.append(min(jaccard_score(labels == c, pred_aligned == c)
                         for c in range(int(k))))
        sil = float(silhouette_score(X, labels, sample_size=min(n, 5000),
                                     random_state=seed)) if len(set(labels)) > 1 else float("nan")
        rows.append({
            "k": int(k),
            "silhouette": sil,
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "mean_ari": float(np.mean(a)),
            "median_ari": float(np.median(a)),
            "min_ari": float(np.min(a)),
            "p05_ari": float(np.percentile(a, 5)),
            "min_cluster_jaccard": float(np.min(j)),
            "mean_cluster_jaccard": float(np.mean(j)),
            "cluster_sizes": [int(x) for x in np.bincount(labels)],
        })
        aris[int(k)] = a
    return pd.DataFrame(rows), base, aris


def consensus(X: np.ndarray, k: int, n_init: int, n_boot: int, seed: int,
              max_vis: int = 2000):
    rng = np.random.default_rng(seed)
    n = len(X)
    vis = np.arange(n) if n <= max_vis else np.sort(
        rng.choice(n, size=max_vis, replace=False))
    acc = np.zeros((len(vis), len(vis)), dtype=np.float32)
    boot_n = min(n, 5000)
    for b in range(n_boot):
        idx = rng.choice(n, size=boot_n, replace=False)
        km = KMeans(n_clusters=int(k), n_init=10,
                    random_state=seed + 1 + b).fit(X[idx])
        pred = km.predict(X[vis])
        acc += (pred[:, None] == pred[None, :]).astype(np.float32)
    return vis, acc / max(n_boot, 1)
