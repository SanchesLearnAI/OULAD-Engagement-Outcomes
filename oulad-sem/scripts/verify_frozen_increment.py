"""V1 — frozen increment check (protocol: research/TMA_DEEP_PROTOCOL.md).

Reads the frozen split, fits M2 and M2+deep indicators, and applies the
pre-registered decision rule.  No feature is added or removed here: the nine
candidate columns come from the deep-exploration outputs.

Run from ``oulad-sem/``::

    python scripts/verify_frozen_increment.py --repeats 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import modeling  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
# Frozen set (research/TMA_DEEP_PROTOCOL.md). The second-wave E2 rework removed
# missed_deadlines/days_late_sum/deadlines_in_window, which were degenerate; the
# anchor submission facts replace them.
DEEP_FEATURES = ["n_silences", "n_internal_silences", "longest_silence",
                 "n_returns_after_silence", "mean_silence", "ended_silent",
                 "anchor_days_late", "anchor_submitted"]
# Checked after the run: ``longest_silence`` is byte-identical to the primary
# feature ``post_longest_inactive_run``, so a de-duplicated set is reported as a
# post-hoc supplement next to the frozen result.
DEDUP_FEATURES = ["n_silences", "mean_silence", "ended_silent", "anchor_days_late"]
THRESHOLD_AP = 0.005


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def topk_precision(y: np.ndarray, p: np.ndarray, k: float) -> float:
    n = max(int(round(len(y) * k)), 1)
    order = np.argsort(-p)
    boundary = p[order[n - 1]]
    above = p > boundary
    ties = p == boundary
    slots = n - int(above.sum())
    hit = y[above].sum() + (y[ties].mean() if ties.any() and slots > 0 else 0) * slots
    return float(hit / n)


def load(root: Path, out: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    for name in ("e9_reengagement_records.parquet", "e2_anchor_records.parquet"):
        path = out / name
        if not path.exists():
            raise FileNotFoundError(f"run the exploration pass first: {path}")
        extra = pd.read_parquet(path)
        keep = [c for c in extra.columns if c in DEEP_FEATURES + KEY]
        frame = frame.merge(extra[keep], on=KEY, how="left", validate="one_to_one")
    missing = [c for c in DEEP_FEATURES if c not in frame]
    if missing:
        raise ValueError(f"missing deep features: {missing}")
    frame[DEEP_FEATURES] = frame[DEEP_FEATURES].fillna(0.0)
    return frame


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

    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    frame = load(root, out)
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    y = frame["y_fail"].to_numpy(dtype=int)

    base = [c for c in modeling.columns_for(frame, "M2") if c in frame]
    versions = {"M2_frozen_baseline": base,
                "M2_plus_deep_frozen": base + DEEP_FEATURES,
                "M2_plus_deep_deduplicated": base + DEDUP_FEATURES}
    preds, rows = {}, []
    for name, cols in versions.items():
        prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", cfg["seed"],
                                        va, 0.80)
        preds[name] = prob
        rows.append({"model": name, "n_features": len(cols),
                     "average_precision": float(average_precision_score(y[te], prob)),
                     "roc_auc": float(roc_auc_score(y[te], prob)),
                     "brier": float(brier_score_loss(y[te], prob)),
                     "top5_precision": topk_precision(y[te], prob, 0.05),
                     "no_skill": float(y[te].mean()),
                     "threshold_rule": rule,
                     "n_test": int(len(te))})
    table = pd.DataFrame(rows)

    rng = np.random.default_rng(cfg["seed"] + 313)
    groups = frame["id_student"].to_numpy()[te]
    members = list(pd.Series(np.arange(len(te))).groupby(groups).indices.values())
    ap_delta, auc_delta, dedup_delta = [], [], []
    for _ in range(args.repeats):
        pick = np.concatenate([members[i] for i in rng.integers(len(members),
                                                               size=len(members))])
        if len(np.unique(y[te][pick])) < 2:
            continue
        ap_delta.append(average_precision_score(y[te][pick], preds["M2_plus_deep_frozen"][pick])
                        - average_precision_score(y[te][pick], preds["M2_frozen_baseline"][pick]))
        dedup_delta.append(
            average_precision_score(y[te][pick], preds["M2_plus_deep_deduplicated"][pick])
            - average_precision_score(y[te][pick], preds["M2_frozen_baseline"][pick]))
        auc_delta.append(roc_auc_score(y[te][pick], preds["M2_plus_deep_frozen"][pick])
                         - roc_auc_score(y[te][pick], preds["M2_frozen_baseline"][pick]))
    ap_delta = np.asarray(ap_delta)
    auc_delta = np.asarray(auc_delta)
    ci_low, ci_high = float(np.quantile(ap_delta, .025)), float(np.quantile(ap_delta, .975))
    point = float(ap_delta.mean())
    has_increment = bool(ci_low > 0 and point >= THRESHOLD_AP)

    dedup = np.asarray(dedup_delta)
    table.loc[len(table)] = {
        "model": "delta_deep_deduplicated_minus_baseline", "n_features": np.nan,
        "average_precision": float(dedup.mean()),
        "roc_auc": np.nan, "brier": np.nan, "top5_precision": np.nan,
        "no_skill": float(y[te].mean()),
        "threshold_rule": (f"ci_low={np.quantile(dedup, .025):+.4f};"
                           f"ci_high={np.quantile(dedup, .975):+.4f};"
                           f"reps={len(dedup)}"),
        "n_test": int(len(te))}
    table.loc[len(table)] = {
        "model": "delta_deep_minus_baseline", "n_features": np.nan,
        "average_precision": point,
        "roc_auc": float(auc_delta.mean()),
        "brier": np.nan, "top5_precision": np.nan, "no_skill": float(y[te].mean()),
        "threshold_rule": f"ci_low={ci_low:+.4f};ci_high={ci_high:+.4f};reps={len(ap_delta)}",
        "n_test": int(len(te))}
    table.to_csv(out / "v1_frozen_increment.csv", index=False)

    verdict = {
        "protocol": "research/TMA_DEEP_PROTOCOL.md",
        "protocol_sha256": sha256(root.parent / "research" / "TMA_DEEP_PROTOCOL.md")
        if (root.parent / "research" / "TMA_DEEP_PROTOCOL.md").exists() else None,
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": cfg.sha256,
        "splits_sha256": sha256(root / "data" / "processed" / "splits.json"),
        "features_sha256": sha256(root / "data" / "processed" / "features.parquet"),
        "n_records": int(len(frame)), "n_test": int(len(te)),
        "candidate_features": DEEP_FEATURES,
        "decision_rule": {
            "metric": "average_precision_delta",
            "requires_ci_excludes_zero": True,
            "min_point_estimate": THRESHOLD_AP,
        },
        "deduplicated_supplement": {
            "features": DEDUP_FEATURES,
            "reason": ("longest_silence duplicates post_longest_inactive_run and "
                       "anchor_submitted duplicates submitted_by_cutoff; this arm "
                       "removes them so the increment is not just a re-count"),
            "point": float(dedup.mean()),
            "ci_low": float(np.quantile(dedup, .025)),
            "ci_high": float(np.quantile(dedup, .975)),
        },
        "observed": {"point": point, "ci_low": ci_low, "ci_high": ci_high,
                     "replicates": int(len(ap_delta)),
                     "auc_delta_point": float(auc_delta.mean()),
                     "ap_baseline": float(table.loc[table["model"].eq(
                         "M2_frozen_baseline"), "average_precision"].iloc[0]),
                     "ap_candidate": float(table.loc[table["model"].eq(
                         "M2_plus_deep_frozen"), "average_precision"].iloc[0])},
        "has_increment": has_increment,
        "expected_before_run": "no increment",
        "matches_expectation": (not has_increment),
    }
    (out / "v1_frozen_verdict.json").write_text(json.dumps(verdict, indent=2),
                                                encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in ("observed", "has_increment",
                                             "matches_expectation")}, indent=2))


if __name__ == "__main__":
    main()
