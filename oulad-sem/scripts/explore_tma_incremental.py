"""Fourth exploration pass: do the new indicators add anything to the model?

E10  Incremental value of the delay/pattern/re-engagement indicators on top of
     the frozen M2 ladder, on the same student-grouped split, with a paired
     student-cluster bootstrap on the AP difference.

Run from ``oulad-sem/``::

    python scripts/explore_tma_incremental.py --out ../results/tma_deep_exploration_20261108
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
from src.tma import daily, modeling  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]

# Candidate increments that exist in the exploration outputs. ``profile_late``
# is never written, and the three window-deadline counters were dropped together
# with the E2 rework (the window never contains a second deadline).
EXTRA = ["n_silences", "n_internal_silences", "longest_silence",
         "n_returns_after_silence", "mean_silence", "ended_silent",
         "anchor_days_late", "sfi"]


def build_frame(root: Path, out: Path, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    joined = frame.copy()
    for name in ("e9_reengagement_records.parquet", "e2_anchor_records.parquet",
                 "e8_sfi_records.parquet"):
        path = out / name
        if not path.exists():
            continue
        extra = pd.read_parquet(path)
        keep = [c for c in extra.columns if c in EXTRA + KEY]
        joined = joined.merge(extra[keep], on=KEY, how="left", validate="one_to_one")
    if "n_silences" in joined:
        joined[["n_silences", "n_internal_silences", "longest_silence",
                "n_returns_after_silence", "mean_silence"]] = joined[
            ["n_silences", "n_internal_silences", "longest_silence",
             "n_returns_after_silence", "mean_silence"]].fillna(0)
    if "delete" == "never":
        pass
    if "ended_silent" in joined:
        joined["ended_silent"] = joined["ended_silent"].astype(float).fillna(0)
    for name in ("anchor_days_late",):
        if name in joined:
            joined[name] = joined[name].fillna(0)
    if "profile_name" not in joined:
        joined["profile_name"] = np.nan
    return joined


def run(root: Path, out: Path, seed: int, repeats: int) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score

    frame = build_frame(root, out, seed)
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    y = frame["y_fail"].to_numpy(dtype=int)

    base = modeling.columns_for(frame, "M2")
    extra = [c for c in EXTRA if c in frame]
    versions = {"M2": base, "M2_plus_deep": base + extra}
    preds, rows = {}, []
    for name, cols in versions.items():
        cols = [c for c in cols if c in frame]
        prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", seed, va, 0.80)
        preds[name] = prob
        rows.append({"model": name, "n_features": len(cols),
                     "average_precision": float(average_precision_score(y[te], prob)),
                     "roc_auc": float(roc_auc_score(y[te], prob)),
                     "no_skill": float(y[te].mean()),
                     "threshold_rule": rule})
    table = pd.DataFrame(rows)

    rng = np.random.default_rng(seed + 77)
    groups = frame["id_student"].to_numpy()[te]
    members = list(pd.Series(np.arange(len(te))).groupby(groups).indices.values())
    deltas = []
    for _ in range(repeats):
        pick = np.concatenate([members[i] for i in rng.integers(len(members),
                                                               size=len(members))])
        if len(np.unique(y[te][pick])) < 2:
            continue
        deltas.append(average_precision_score(y[te][pick], preds["M2_plus_deep"][pick])
                      - average_precision_score(y[te][pick], preds["M2"][pick]))
    if deltas:
        table.loc[len(table), :] = {
            "model": "delta_M2_plus_deep_minus_M2", "n_features": np.nan,
            "average_precision": float(np.mean(deltas)), "roc_auc": np.nan,
            "no_skill": float(np.quantile(deltas, .025)),
            "threshold_rule": (f"ci_low={np.quantile(deltas, .025):+.4f};"
                               f"ci_high={np.quantile(deltas, .975):+.4f};"
                               f"reps={len(deltas)}")}
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    table = run(root, out, cfg["seed"], args.repeats)
    table.to_csv(out / "e10_incremental.csv", index=False)
    summary = {
        "ap_m2": float(table.loc[table["model"].eq("M2"), "average_precision"].iloc[0]),
        "ap_m2_plus_deep": float(table.loc[table["model"].eq("M2_plus_deep"),
                                           "average_precision"].iloc[0]),
        "delta_ci": table.loc[table["model"].str.startswith("delta"),
                              "threshold_rule"].iloc[0]
        if table["model"].str.startswith("delta").any() else None,
    }
    (out / "summary_incremental.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
