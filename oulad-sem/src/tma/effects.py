"""Paired contrasts, bootstrap intervals and Holm-adjusted primary tests.

The six primary indicators are fixed in
``research/TMA_PRIMARY_PROTOCOL.md``; Holm's step-down procedure controls the
family-wise error rate across them.  ``relative_submission_day`` only exists
for pairs where both records submitted by the cutoff, as the protocol states,
so each metric carries its own valid-pair count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pairing import component_intervals, paired_differences

PRIMARY_METRICS = [
    "submitted_by_cutoff",
    "submitted_after_due",
    "relative_submission_day",
    "post_active_days",
    "post_longest_inactive_run",
    "post_week2_share",
]
# Only definable when both sides submitted before the cutoff.
SUBMITTED_ONLY = {"relative_submission_day", "days_early", "days_late"}
EXPLORATORY_METRICS = [
    "post_both_weeks", "post_peak7_share", "post_entropy_norm",
    "post_first_active_day", "post_resource_type_count", "post_resource_entropy",
    "pre_clicks", "pre_active_days", "last3_share", "pre_log_clicks",
    "post_log_clicks", "post_log_clicks_squared", "post_minus_pre_log",
]


def valid_mask(frame: pd.DataFrame, pairs: pd.DataFrame, metric: str) -> np.ndarray:
    """Rows of ``pairs`` where the metric exists on both sides of the pair."""
    fail, passing = paired_differences(frame, pairs, [metric])
    mask = fail[metric].notna().to_numpy() & passing[metric].notna().to_numpy()
    return mask


def feature_differences(frame: pd.DataFrame, pairs: pd.DataFrame,
                        features: list[str]) -> np.ndarray:
    fail, passing = paired_differences(frame, pairs, features)
    return (fail[features].to_numpy(dtype=float)
            - passing[features].to_numpy(dtype=float))


def effect_table(frame: pd.DataFrame, pairs: pd.DataFrame, features: list[str],
                 seed: int, repeats: int) -> pd.DataFrame:
    """Mean Fail-minus-Pass differences with component-bootstrap intervals."""
    rows = []
    for name in features:
        if name not in frame:
            continue
        mask = valid_mask(frame, pairs, name)
        subset = pairs.loc[mask].reset_index(drop=True)
        if subset.empty:
            continue
        differences = feature_differences(frame, subset, [name])
        intervals = component_intervals(subset, differences, seed, repeats)
        fail, passing = paired_differences(frame, subset, [name])
        rows.append({"feature": name, "pairs": int(len(subset)),
                     "fail_mean": float(fail[name].mean()),
                     "pass_mean": float(passing[name].mean()),
                     "fail_minus_pass": float(differences[:, 0].mean()),
                     "ci_low": float(intervals[0, 0]),
                     "ci_high": float(intervals[1, 0])})
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at 1."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * float(p_values[i])
        running = max(running, value)
        adjusted[i] = min(1.0, running)
    return adjusted


def _component_labels(pairs: pd.DataFrame) -> np.ndarray:
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(pairs["fail_student"], pairs["pass_student"]):
        parent[find(int(a))] = find(int(b))
    return pd.factorize(np.asarray([find(int(a)) for a in pairs["fail_student"]]))[0]


def paired_bootstrap_p(frame: pd.DataFrame, pairs: pd.DataFrame, metric: str,
                       seed: int, repeats: int) -> dict | None:
    """Sign-flip bootstrap p-value and interval for one paired mean difference."""
    mask = valid_mask(frame, pairs, metric)
    subset = pairs.loc[mask].reset_index(drop=True)
    if len(subset) < 10:
        return None
    differences = feature_differences(frame, subset, [metric])[:, 0]
    labels = _component_labels(subset)
    n = int(labels.max()) + 1
    sums = np.bincount(labels, weights=differences, minlength=n)
    counts = np.bincount(labels, minlength=n)
    rng = np.random.default_rng(seed + 7331)
    observed = float(differences.mean())
    boots = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0, n, n)
        boots[i] = sums[selected].sum() / counts[selected].sum()
    centered = boots - observed
    p = float((np.abs(centered) >= abs(observed)).mean()) if observed != 0 else 1.0
    return {"pairs": int(len(subset)), "fail_minus_pass": observed,
            "ci_low": float(np.quantile(boots, .025)),
            "ci_high": float(np.quantile(boots, .975)),
            "p_value": max(p, 1.0 / repeats)}


def primary_tests(frame: pd.DataFrame, pairs: pd.DataFrame, *, seed: int,
                  repeats: int, metrics: list[str] | None = None) -> pd.DataFrame:
    """Bootstrap p-values for the primary indicators with Holm adjustment."""
    metrics = list(metrics or PRIMARY_METRICS)
    rows = []
    for name in metrics:
        if name not in frame:
            continue
        result = paired_bootstrap_p(frame, pairs, name, seed, repeats)
        if result is None:
            continue
        rows.append({"metric": name, **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = holm_adjust(table["p_value"].tolist())
    table["reject_holm_0_05"] = table["p_holm"] < 0.05
    table["family"] = f"primary_{len(table)}"
    return table


def presentation_slices(frame: pd.DataFrame, pairs: pd.DataFrame,
                        metrics: list[str]) -> pd.DataFrame:
    """Per-presentation paired difference, the heterogeneity check for S06."""
    rows = []
    for (module, presentation), idx in pairs.groupby(["code_module", "code_presentation"]).indices.items():
        sub = pairs.loc[list(idx)].reset_index(drop=True)
        for name in metrics:
            if name not in frame:
                continue
            keep = valid_mask(frame, sub, name)
            if keep.sum() == 0:
                continue
            chosen = sub.loc[keep]
            f = frame.iloc[chosen["fail_index"].to_numpy()][name].to_numpy(dtype=float)
            p = frame.iloc[chosen["pass_index"].to_numpy()][name].to_numpy(dtype=float)
            rows.append({"code_module": module, "code_presentation": presentation,
                         "feature": name, "pairs": int(len(chosen)),
                         "fail_minus_pass": float((f - p).mean())})
    return pd.DataFrame(rows)
