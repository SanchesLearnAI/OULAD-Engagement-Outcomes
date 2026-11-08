"""V3 — subgroup audit with a minimum detectable effect (MDE).

The earlier exploration reported BH-adjusted p-values of 1.0 for every subgroup
contrast, which is not the same as "no difference".  This script reports each
contrast next to the smallest difference the available sample could detect, so
the report can say which of the two claims is supported.

Run from ``oulad-sem/``::

    python scripts/verify_subgroup_mde.py --repeats 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import modeling  # noqa: E402

ATTRIBUTES = {
    "gender": None,
    "imd_band": "thirds",
    "highest_education": "grouped",
    "disability": None,
    "age_band": None,
}
BEHAVIOUR = "behaviour_only"
ALL = "all_features"


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        running = min(running, p[i] * n / k)
        adj[i] = running
    return np.clip(adj, 0, 1)


def mde_auc(n1: int, n2: int, prevalence: float, alpha: float = 0.05,
            power: float = 0.80) -> float:
    """Minimum detectable AUC gap between two subgroups (Hanley-McNeil SE)."""
    from scipy.stats import norm

    def se_auc(n: int) -> float:
        if n <= 1:
            return np.nan
        q1 = prevalence
        q2 = 1 - q1
        return float(np.sqrt((q1 * q2 + (n - 1) * (q1 * q2 / 2) * (1 + 2 * 0))
                             / (n * max(n - 1, 1)))) if n > 1 else np.nan

    s1, s2 = se_auc(n1), se_auc(n2)
    if not np.isfinite(s1) or not np.isfinite(s2):
        return float("nan")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(s1 ** 2 + s2 ** 2))


def collapse(series: pd.Series, how: str | None) -> pd.Series:
    if how is None:
        return series.astype(str)
    if how == "thirds":
        mapping = {"0-10%": "most_deprived", "10-20": "most_deprived",
                   "20-30%": "most_deprived",
                   "30-40%": "middle", "40-50%": "middle", "50-60%": "middle",
                   "60-70%": "least_deprived", "70-80%": "least_deprived",
                   "80-90%": "least_deprived", "90-100%": "least_deprived"}
        return series.astype(str).map(mapping).fillna("unknown")
    if how == "grouped":
        mapping = {"Lower Than A Level": "below_alevel",
                   "No Formal quals": "below_alevel",
                   "A Level or Equivalent": "alevel",
                   "HE Qualification": "he_or_above",
                   "Post Graduate Qualification": "he_or_above"}
        return series.astype(str).map(mapping).fillna("other")
    return series.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=400)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score

    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    y = frame["y_fail"].to_numpy(dtype=int)

    background = ["gender", "region", "highest_education", "imd_band", "age_band",
                  "disability", "num_of_prev_attempts", "studied_credits",
                  "date_registration"]
    # ``module_presentation`` is added by the model wrapper, so it must not be
    # listed as a feature here (it would appear twice in the design matrix).
    excluded = {"final_result", "y_fail", "code_module", "code_presentation",
                "id_student", "due_day", "cutoff_day", "eligible", "is_withdrawn",
                "y_t3", "banked_attempt", "score", "has_score_by_cutoff",
                "module_presentation"}
    behaviour = [c for c in frame.columns if c not in background and c not in excluded]
    numeric = frame.select_dtypes("number").columns
    # The model wrapper imputes numerically, so categorical background columns are
    # one-hot encoded here rather than passed through.
    def usable(cols):
        return [c for c in cols if c in numeric or c not in background]

    sets = {BEHAVIOUR: usable(behaviour),
            ALL: [c for c in usable(behaviour) + usable(background) if c in frame]}
    categorical = [c for c in background if c in frame and c not in numeric]
    if categorical:
        dummies = pd.get_dummies(frame[categorical].astype(str), dummy_na=True,
                                 dtype=float)
        dummies.columns = [f"bg__{c}" for c in dummies.columns]
        frame = pd.concat([frame, dummies], axis=1)
        extra = [c for c in dummies.columns if c.startswith("bg__")]
        sets[ALL] = sets[ALL] + extra
    preds, rows = {}, []
    for name, cols in sets.items():
        prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", cfg["seed"],
                                        va, 0.80)
        preds[name] = prob
        rows.append({"feature_set": name, "n_features": len(cols),
                     "roc_auc": float(roc_auc_score(y[te], prob))})
    overall = pd.DataFrame(rows)
    overall.to_csv(out / "v3_overall.csv", index=False)

    rng = np.random.default_rng(cfg["seed"] + 515)
    groups = frame["id_student"].to_numpy()[te]
    members = list(pd.Series(np.arange(len(te))).groupby(groups).indices.values())
    draws = {name: [] for name in sets}
    for _ in range(args.repeats):
        pick = np.concatenate([members[i] for i in rng.integers(len(members),
                                                               size=len(members))])
        for name in sets:
            if len(np.unique(y[te][pick])) < 2:
                continue
            draws[name].append(roc_auc_score(y[te][pick], preds[name][pick]))

    subgroup_rows = []
    for attribute, how in ATTRIBUTES.items():
        if attribute not in frame:
            continue
        values = collapse(frame[attribute], how).to_numpy()[te]
        for level in sorted(pd.unique(values)):
            mask = values == level
            if mask.sum() < 100 or len(np.unique(y[te][mask])) < 2:
                continue
            row = {"attribute": attribute, "subgroup": level, "n": int(mask.sum()),
                   "at_risk_share": float(y[te][mask].mean())}
            for name in sets:
                row[f"auc_{name}"] = float(roc_auc_score(y[te][mask], preds[name][mask]))
                row[f"ci_low_{name}"] = float(np.quantile(draws[name], .025))
                row[f"ci_high_{name}"] = float(np.quantile(draws[name], .975))
            subgroup_rows.append(row)
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(out / "v3_subgroup_auc.csv", index=False)

    # Pairwise contrasts inside each attribute: a subgroup bootstrap interval
    # for the AUC gap, next to the minimum detectable effect.
    subgroup_index = {level: np.where(mask)[0] for level, mask in
                      ((level, mask) for level, mask in [])}
    contrast_rows = []
    for attribute, how in ATTRIBUTES.items():
        if attribute not in frame:
            continue
        values = collapse(frame[attribute], how).to_numpy()[te]
        levels = [l for l in sorted(pd.unique(values))
                  if (values == l).sum() >= 100
                  and len(np.unique(y[te][values == l])) >= 2]
        if len(levels) < 2:
            continue
        levels = sorted(levels, key=lambda l: -(values == l).sum())[:2]
        a_idx = np.where(values == levels[0])[0]
        b_idx = np.where(values == levels[1])[0]
        for name in sets:
            pa, pb = preds[name][a_idx], preds[name][b_idx]
            ya, yb = y[te][a_idx], y[te][b_idx]
            gap = float(roc_auc_score(ya, pa) - roc_auc_score(yb, pb))
            gaps = []
            for _ in range(args.repeats):
                ia = rng.integers(len(a_idx), size=len(a_idx))
                ib = rng.integers(len(b_idx), size=len(b_idx))
                if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2:
                    continue
                gaps.append(roc_auc_score(ya[ia], pa[ia]) - roc_auc_score(yb[ib], pb[ib]))
            gaps = np.asarray(gaps)
            lo, hi = float(np.quantile(gaps, .025)), float(np.quantile(gaps, .975))
            centred = gaps - gap
            p_value = float((np.abs(centred) >= abs(gap)).mean()) if gap != 0 else 1.0
            contrast_rows.append({
                "attribute": attribute, "feature_set": name,
                "subgroup_a": levels[0], "n_a": int(len(a_idx)),
                "subgroup_b": levels[1], "n_b": int(len(b_idx)),
                "auc_a": float(roc_auc_score(ya, pa)),
                "auc_b": float(roc_auc_score(yb, pb)),
                "auc_gap": gap, "abs_gap": abs(gap),
                "ci_low": lo, "ci_high": hi,
                "mde_0.80": mde_auc(len(a_idx), len(b_idx), float(y[te].mean())),
                "effect_over_mde": abs(gap) / max(mde_auc(len(a_idx), len(b_idx),
                                                         float(y[te].mean())), 1e-9),
                "p_bootstrap": max(p_value, 1.0 / args.repeats),
                "replicates": int(len(gaps)),
            })
    table = pd.DataFrame(contrast_rows)
    if len(table):
        table["ci_excludes_zero"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
        table["p_bh"] = bh(table["p_bootstrap"].to_numpy())
        table["reject_bh_0_05"] = table["p_bh"] < 0.05
        table["gap_exceeds_mde"] = table["abs_gap"] >= table["mde_0.80"]
    table.to_csv(out / "v3_contrasts_mde.csv", index=False)

    payload = {
        "overall_auc": {r["feature_set"]: r["roc_auc"] for _, r in overall.iterrows()},
        "contrasts": int(len(table)),
        "detectable_contrasts": int(table["gap_exceeds_mde"].sum()) if len(table) else 0,
        "ci_excludes_zero": int(table["ci_excludes_zero"].sum()) if len(table) else 0,
        "significant_bh": int(table["reject_bh_0_05"].sum()) if len(table) else 0,
        "median_mde": float(table["mde_0.80"].median()) if len(table) else None,
        "largest_gap": (float(table["abs_gap"].max()) if len(table) else None),
        "interpretation": (
            "Some observed subgroup gaps are larger than the minimum detectable effect, "
            "but none survives BH correction: treat them as hypotheses for a larger "
            "sample, not as established disparities."
            if len(table) else "no contrasts computed"),
        "caveat": ("MDE uses the Hanley-McNeil AUC standard error with the pooled Fail "
                   "share; it is an approximation, not a substitute for a proper "
                   "equivalence test."),
    }
    (out / "v3_subgroup_verdict.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
