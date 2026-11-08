"""Within-presentation 1:1 pairing on post-deadline clicks.

Ported from ``oulad_analysis/matched.py`` (``make_pairs``,
``_component_intervals``) with the matching variable renamed to the
post-deadline click total.  Pairs are descriptive: they are built from the
final outcome, so they cannot estimate a population failure rate, and the
greedy order is not claimed to be optimal.
"""
from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]


def make_pairs(frame: pd.DataFrame, tolerance: float, seed: int = 42,
               match_column: str = "post_clicks") -> pd.DataFrame:
    """Exact matches first, then nearest available; no record reused.

    For positive counts, ``max(clicks)/min(clicks) <= 1 + tolerance``.
    Zero clicks only match zero clicks.  Matching runs inside each
    (module, presentation) and uses the row position in ``frame``.
    """
    population = frame[frame["final_result"].isin(["Fail", "Pass", "Distinction"])]
    population = population.reset_index(drop=True)
    if match_column not in population:
        raise KeyError(match_column)
    rng = np.random.default_rng(seed)
    pairs = []
    for (module, presentation), group in population.groupby(BATCH):
        failures = rng.permutation(group.index[group["final_result"].eq("Fail")]).tolist()
        passing = rng.permutation(group.index[~group["final_result"].eq("Fail")]).tolist()
        passing.sort(key=lambda i: population.at[i, match_column])
        clicks = [float(population.at[i, match_column]) for i in passing]
        remaining = []
        for exact in (True, False):
            if not exact:
                failures = remaining
            for i in failures:
                value = float(population.at[i, match_column])
                pos = bisect_left(clicks, value)
                options = [j for j in (pos - 1, pos) if 0 <= j < len(clicks)]
                if exact:
                    options = [j for j in options if clicks[j] == value]
                else:
                    options = [j for j in options
                               if (value == clicks[j] == 0) or
                               (min(value, clicks[j]) > 0 and
                                max(value, clicks[j]) / min(value, clicks[j]) <= 1 + tolerance + 1e-12)]
                if not options:
                    if exact:
                        remaining.append(i)
                    continue
                j = min(options, key=lambda j: abs(np.log1p(value) - np.log1p(clicks[j])))
                pi = passing.pop(j)
                other = clicks.pop(j)
                pairs.append({"code_module": module, "code_presentation": presentation,
                              "fail_index": i, "pass_index": pi,
                              "fail_student": int(population.at[i, "id_student"]),
                              "pass_student": int(population.at[pi, "id_student"]),
                              "fail_clicks": value, "pass_clicks": other,
                              "relative_gap": (max(value, other) / min(value, other) - 1
                                               if min(value, other) > 0 else 0.0)})
    return pd.DataFrame(pairs, columns=["code_module", "code_presentation", "fail_index",
                                        "pass_index", "fail_student", "pass_student",
                                        "fail_clicks", "pass_clicks", "relative_gap"])


def component_intervals(pairs: pd.DataFrame, differences: np.ndarray, seed: int,
                        repeats: int) -> np.ndarray:
    """Paired 95% intervals from a student-component bootstrap.

    Pairs sharing any student are resampled together; this keeps student
    dependence across presentations and keeps both sides of each pair.  The
    intervals are conditional on the fixed pairing, so re-matching uncertainty
    is not included.
    """
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(pairs["fail_student"], pairs["pass_student"]):
        parent[find(int(a))] = find(int(b))
    labels = pd.factorize(np.asarray([find(int(a)) for a in pairs["fail_student"]]))[0]
    n = int(labels.max()) + 1
    sums = np.zeros((n, differences.shape[1]))
    np.add.at(sums, labels, differences)
    counts = np.bincount(labels)
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, differences.shape[1]))
    for i in range(repeats):
        selected = rng.integers(0, n, n)
        values[i] = sums[selected].sum(axis=0) / counts[selected].sum()
    return np.quantile(values, [.025, .975], axis=0)


def paired_differences(frame: pd.DataFrame, pairs: pd.DataFrame,
                       features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail minus Pass/Distinction feature means for both sides of each pair."""
    fail = frame.iloc[pairs["fail_index"].to_numpy()].reset_index(drop=True)
    passing = frame.iloc[pairs["pass_index"].to_numpy()].reset_index(drop=True)
    return fail, passing


def balance_row(frame: pd.DataFrame, pairs: pd.DataFrame, tolerance: float,
                match_column: str = "post_clicks") -> dict:
    """One balance summary: counts, coverage and residual click imbalance."""
    fail, passing = paired_differences(frame, pairs, [])
    post_sd = np.sqrt((fail["log_post_clicks"].var(ddof=0)
                       + passing["log_post_clicks"].var(ddof=0)) / 2)
    pre_sd = np.sqrt((fail["log_pre_clicks"].var(ddof=0)
                      + passing["log_pre_clicks"].var(ddof=0)) / 2)
    return {
        "tolerance": tolerance,
        "binary_records": int(len(frame)),
        "fail_records": int(frame["final_result"].eq("Fail").sum()),
        "pairs": int(len(pairs)),
        "fail_coverage": float(len(pairs) / max(frame["final_result"].eq("Fail").sum(), 1)),
        "zero_post_pairs": int(pairs["fail_clicks"].eq(0).sum()),
        "mean_post_click_gap": float(pairs["relative_gap"].mean()) if len(pairs) else float("nan"),
        "post_log_click_smd": float((fail["log_post_clicks"].mean()
                                     - passing["log_post_clicks"].mean()) / post_sd) if post_sd else 0.0,
        "pre_log_click_smd": float((fail["log_pre_clicks"].mean()
                                    - passing["log_pre_clicks"].mean()) / pre_sd) if pre_sd else 0.0,
        "fail_pre_click_mean": float(fail["pre_clicks"].mean()),
        "pass_pre_click_mean": float(passing["pre_clicks"].mean()),
        "fail_post_click_mean": float(fail["post_clicks"].mean()),
        "pass_post_click_mean": float(passing["post_clicks"].mean()),
    }
