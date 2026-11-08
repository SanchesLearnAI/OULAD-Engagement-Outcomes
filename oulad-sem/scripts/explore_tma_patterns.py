"""Second exploration pass: significance-tested sequence patterns and capacity.

E6  Discriminative day-symbol patterns with gap tolerance, tested against a
    label-permutation null and controlled with Benjamini-Hochberg.
E7  Alert-capacity operating points (top-k precision/recall), calibration drift
    by submission stratum, and a matched-versus-unmatched comparison of what the
    pairing design can and cannot answer.

Run from ``oulad-sem/``::

    python scripts/explore_tma_patterns.py --out ../results/tma_deep_exploration_20261108
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import daily, pairing  # noqa: E402
from src.tma.stats import bh  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]


def symbols_for(post: np.ndarray) -> np.ndarray:
    """0 = silent, 1 = below own positive median, 2 = at or above it."""
    positive = np.where(post > 0, post, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(positive, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    symbols = np.zeros_like(post, dtype=int)
    symbols[(post > 0) & (post < medians[:, None])] = 1
    symbols[(post > 0) & (post >= medians[:, None])] = 2
    return symbols


def _pattern_counts(symbols: np.ndarray, pattern: tuple[int, ...],
                    max_gap: int) -> np.ndarray:
    """Does the pattern occur in order allowing at most ``max_gap`` skipped days?"""
    n, days = symbols.shape
    hit = np.zeros(n, dtype=bool)
    rows = np.arange(n)
    for start in range(0, days):
        if not (symbols[:, start] == pattern[0]).any():
            continue
        position = np.full(n, start)
        ok = symbols[:, start] == pattern[0]
        for token in pattern[1:]:
            found = np.zeros(n, dtype=bool)
            best = np.full(n, -1)
            for offset in range(1, max_gap + 2):
                cand = position + offset
                valid = cand < days
                idx = np.where(valid, cand, 0)
                match = valid & (symbols[rows, idx] == token) & ~found
                found |= match
                best = np.where(match, cand, best)
            ok &= found
            position = np.where(found, best, position)
            if not ok.any():
                break
        hit |= ok
    return hit


def discriminative_patterns(symbols: np.ndarray, y: np.ndarray, *,
                            lengths=(2, 3, 4), max_gap: int = 1,
                            min_support: int = 40, permutations: int = 200,
                            seed: int = 42) -> pd.DataFrame:
    """Enumerate short patterns, then test each lift against shuffled labels."""
    rng = np.random.default_rng(seed)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    candidates = []
    for length in lengths:
        for pattern in np.ndindex(*([3] * length)):
            if len(set(pattern)) == 1 and pattern[0] == 0 and length > 2:
                continue  # long all-silent runs are captured by the run features
            hit = _pattern_counts(symbols, tuple(pattern), max_gap)
            a = int((hit & (y == 1)).sum())
            b = int((hit & (y == 0)).sum())
            if a + b < min_support:
                continue
            candidates.append({"pattern": "".join(map(str, pattern)),
                               "length": length,
                               "n_fail_with": a, "n_pass_with": b,
                               "support_fail": a / max(n_pos, 1),
                               "support_pass": b / max(n_neg, 1),
                               "hit": hit})
    if not candidates:
        return pd.DataFrame()
    null = np.empty((permutations, len(candidates)))
    for i in range(permutations):
        shuffled = rng.permutation(y)
        pos = shuffled == 1
        neg = ~pos
        for j, cand in enumerate(candidates):
            hit = cand["hit"]
            null[i, j] = ((hit & pos).sum() / max(pos.sum(), 1)
                          - (hit & neg).sum() / max(neg.sum(), 1))
    rows = []
    for j, cand in enumerate(candidates):
        observed = cand["support_fail"] - cand["support_pass"]
        p = float((np.abs(null[:, j]) >= abs(observed)).mean())
        rows.append({"pattern": cand["pattern"], "length": cand["length"],
                     "max_gap": max_gap,
                     "n_fail_with": cand["n_fail_with"],
                     "n_pass_with": cand["n_pass_with"],
                     "support_fail": cand["support_fail"],
                     "support_pass": cand["support_pass"],
                     "support_diff": observed,
                     "log_lift": float(np.log((cand["support_fail"] + 1e-9)
                                              / (cand["support_pass"] + 1e-9))),
                     "p_perm": max(p, 1.0 / permutations)})
    table = pd.DataFrame(rows)
    table["p_bh"] = bh(table["p_perm"].to_numpy())
    table["reject_bh_0_05"] = table["p_bh"] < 0.05
    return table.sort_values("support_diff", ascending=False).reset_index(drop=True)


def pair_discordance(symbols: np.ndarray, pairs: pd.DataFrame, *,
                    max_gap: int = 1, min_discordant: int = 20) -> pd.DataFrame:
    """Within-pair test: does the pattern favour Fail or the matched partner?

    Pairing fixes post-deadline click volume, so a plain support difference can
    be driven by volume alone.  Conditioning on pairs where exactly one side has
    the pattern removes that channel and turns the comparison into a sign test.
    """
    from scipy.stats import binomtest

    fi = pairs["fail_index"].to_numpy()
    pi = pairs["pass_index"].to_numpy()
    rows = []
    for length in (2, 3):
        for pattern in np.ndindex(*([3] * length)):
            hit = _pattern_counts(symbols, tuple(pattern), max_gap)
            fail_only = int((hit[fi] & ~hit[pi]).sum())
            pass_only = int((~hit[fi] & hit[pi]).sum())
            discordant = fail_only + pass_only
            if discordant < min_discordant:
                continue
            p = float(binomtest(fail_only, discordant, 0.5).pvalue)
            rows.append({"pattern": "".join(map(str, pattern)),
                         "length": length, "max_gap": max_gap,
                         "pairs_fail_only": fail_only,
                         "pairs_pass_only": pass_only,
                         "discordant_pairs": discordant,
                         "odds_fail_over_pass": fail_only / max(pass_only, 1),
                         "p_value": p})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_bh"] = bh(table["p_value"].to_numpy())
    table["reject_bh_0_05"] = table["p_bh"] < 0.05
    return table.sort_values("odds_fail_over_pass", ascending=False).reset_index(drop=True)


def unmatched_contrasts(frame: pd.DataFrame, master: pd.DataFrame,
                        pairs: pd.DataFrame) -> pd.DataFrame:
    """The same indicators with and without pairing.

    Pairing fixes post-deadline clicks, so any indicator that is a direct
    function of those clicks (for example "never returns") is mechanically
    balanced and cannot be studied in the matched sample.
    """
    cohort = frame.copy()
    indicators = ["post_clicks", "post_active_days", "post_longest_inactive_run",
                  "post_week2_share", "submitted_by_cutoff", "submitted_after_due"]
    rows = []
    for name in indicators:
        values = cohort[name].to_numpy(dtype=float)
        fail = values[cohort["final_result"].eq("Fail").to_numpy()]
        passing = values[~cohort["final_result"].eq("Fail").to_numpy()]
        rows.append({"indicator": name, "design": "unmatched_cohort",
                     "n_fail": int(len(fail)), "n_pass": int(len(passing)),
                     "fail_mean": float(np.nanmean(fail)),
                     "pass_mean": float(np.nanmean(passing)),
                     "fail_minus_pass": float(np.nanmean(fail) - np.nanmean(passing))})
        f = values[pairs["fail_index"].to_numpy()]
        p = values[pairs["pass_index"].to_numpy()]
        rows.append({"indicator": name, "design": "matched_pairs",
                     "n_fail": int(len(f)), "n_pass": int(len(p)),
                     "fail_mean": float(np.nanmean(f)),
                     "pass_mean": float(np.nanmean(p)),
                     "fail_minus_pass": float(np.nanmean(f) - np.nanmean(p))})
    # "Never returns" is a thresholded version of post_clicks, so the pairing
    # makes it uninformative by construction; record that explicitly.
    never = (cohort["post_clicks"].to_numpy() == 0).astype(float)
    rows.append({"indicator": "never_returns", "design": "unmatched_cohort",
                 "n_fail": int(cohort["final_result"].eq("Fail").sum()),
                 "n_pass": int((~cohort["final_result"].eq("Fail")).sum()),
                 "fail_mean": float(never[cohort["final_result"].eq("Fail").to_numpy()].mean()),
                 "pass_mean": float(never[(~cohort["final_result"].eq("Fail")).to_numpy()].mean()),
                 "fail_minus_pass": float(
                     never[cohort["final_result"].eq("Fail").to_numpy()].mean()
                     - never[(~cohort["final_result"].eq("Fail")).to_numpy()].mean())})
    return pd.DataFrame(rows)


def capacity_curve(y: np.ndarray, p: np.ndarray, steps=(0.01, 0.02, 0.05, 0.10,
                                                       0.15, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Precision and recall at fixed alert-list sizes (intervention capacity)."""
    rows = []
    base = float(y.mean())
    for frac in steps:
        k = max(int(round(len(y) * frac)), 1)
        order = np.argsort(-p)
        boundary = p[order[k - 1]]
        above = p > boundary
        ties = p == boundary
        slots = k - int(above.sum())
        hits = y[above].sum() + (y[ties].mean() if ties.any() and slots > 0 else 0.0) * slots
        precision = hits / k
        rows.append({"alert_fraction": frac, "alerted": k,
                     "precision": float(precision),
                     "recall": float(hits / max(y.sum(), 1)),
                     "lift_over_base": float(precision / base) if base else float("nan"),
                     "base_rate": base})
    return pd.DataFrame(rows)


def calibration_by_stratum(frame: pd.DataFrame, prob: np.ndarray,
                           test_idx: np.ndarray) -> pd.DataFrame:
    """Wu et al. (2026) Table 9 style drift check on the submission strata."""
    y = frame["y_fail"].to_numpy(dtype=int)[test_idx]
    p = prob
    submitted = frame["submitted_by_cutoff"].to_numpy()[test_idx]
    band = np.where(submitted == 0, "no_submission",
                    np.where(frame["relative_submission_day"].to_numpy()[test_idx] >= 0,
                             "late_submission", "early_submission"))
    rows = []
    for name in ("no_submission", "early_submission", "late_submission"):
        mask = band == name
        if mask.sum() == 0:
            continue
        rows.append({"stratum": name, "n": int(mask.sum()),
                     "observed": float(y[mask].mean()),
                     "predicted": float(p[mask].mean()),
                     "calibration_error": float(p[mask].mean() - y[mask].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    post_days = int(cfg["study"]["post_days"])

    master = pd.read_parquet(root / "data" / "interim" / "master.parquet")
    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    cohort = binary_cohort(master)
    lookup = cohort[KEY + ["due_day"]]
    events = daily.daily_events(root / "data" / "raw" / "oulad" / "studentVle.csv",
                               lookup, pre_days=14, post_days=post_days)
    post = daily.click_matrix(events, pd.MultiIndex.from_frame(frame[KEY]), 1, post_days)
    pairs = pairing.make_pairs(frame, 0.05, cfg["seed"])
    y = frame["y_fail"].to_numpy(dtype=int)

    symbols = symbols_for(post)
    patterns = discriminative_patterns(symbols, y, max_gap=1,
                                       permutations=args.permutations, seed=cfg["seed"])
    patterns.to_csv(out / "e6_patterns_permutation.csv", index=False)

    contrasts = unmatched_contrasts(frame, master, pairs)
    contrasts.to_csv(out / "e7_matched_vs_unmatched.csv", index=False)

    # Within-pair discordance: the version of the pattern analysis that the
    # pairing actually licenses.
    discord = pair_discordance(symbols, pairs)
    discord.to_csv(out / "e7_pair_discordance.csv", index=False)

    # A single fixed split, refitted exactly as the primary model, for the
    # capacity and calibration views.
    from sklearn.model_selection import GroupShuffleSplit
    from src.tma import modeling
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    cols = modeling.columns_for(frame, "M2")
    prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", cfg["seed"], va, 0.80)
    cap = capacity_curve(y[te], prob)
    cap.to_csv(out / "e7_capacity_curve.csv", index=False)
    calib = calibration_by_stratum(frame, prob, te)
    calib.to_csv(out / "e7_calibration_strata.csv", index=False)

    summary = {
        "patterns_tested": int(len(patterns)),
        "patterns_significant_bh": int(patterns["reject_bh_0_05"].sum()),
        "top_pattern": patterns.iloc[0]["pattern"] if len(patterns) else None,
        "top_pattern_support_diff": float(patterns.iloc[0]["support_diff"]) if len(patterns) else None,
        "never_returns_unmatched_diff": float(
            contrasts[(contrasts["indicator"] == "never_returns")
                      & (contrasts["design"] == "unmatched_cohort")]["fail_minus_pass"].iloc[0]),
        "never_returns_matched_diff": float(
            contrasts[(contrasts["indicator"] == "post_clicks")
                      & (contrasts["design"] == "matched_pairs")]["fail_minus_pass"].iloc[0]),
        "discordant_patterns_tested": int(len(discord)),
        "discordant_significant_bh": int(discord["reject_bh_0_05"].sum()),
        "top5_precision": float(cap.loc[cap["alert_fraction"].eq(0.05), "precision"].iloc[0]),
        "top20_recall": float(cap.loc[cap["alert_fraction"].eq(0.20), "recall"].iloc[0]),
        "primary_threshold_rule": rule,
    }
    (out / "summary_patterns.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
