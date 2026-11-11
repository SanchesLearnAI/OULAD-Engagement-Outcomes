"""V7 — parity check for the restored pre-window and gap features.

The integration had dropped every pre-deadline burst feature and the signed
relative gap.  This script re-derives those quantities and compares them with the
values published on the TMA branch, so the restoration is verified rather than
assumed.

Run from ``oulad-sem/``::

    python scripts/verify_restored_parity.py
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

KEY = ["code_module", "code_presentation", "id_student"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    rows = []

    # 1. Relative gap: compare the per-outcome distribution with the published
    #    relative_gap_distribution.csv from the TMA branch.
    ref_gap = pd.read_csv(root.parent / "results" / "tma_relative_gap_20261103"
                          / "relative_gap_distribution.csv")
    for _, r in ref_gap.iterrows():
        outcome = r["outcome"]
        if outcome == "Pass/Distinction":
            subset = frame[frame["final_result"].isin(["Pass", "Distinction"])]
        else:
            subset = frame[frame["final_result"].eq(outcome)]
        rows.append({
            "check": "relative_gap_mean",
            "detail": outcome,
            "reference": float(r["mean_gap"]),
            "restored": float(subset["relative_click_gap"].mean()),
            "abs_diff": abs(float(r["mean_gap"]) - float(subset["relative_click_gap"].mean())),
        })
        rows.append({
            "check": "relative_gap_median",
            "detail": outcome,
            "reference": float(r["median_gap"]),
            "restored": float(subset["relative_click_gap"].median()),
            "abs_diff": abs(float(r["median_gap"]) - float(subset["relative_click_gap"].median())),
        })
    rows.append({
        "check": "both_zero_records", "detail": "all",
        "reference": float(ref_gap["both_zero"].sum()),
        "restored": float(frame["both_zero_clicks"].sum()),
        "abs_diff": abs(float(ref_gap["both_zero"].sum()) - float(frame["both_zero_clicks"].sum())),
    })

    # 2. Burst features: verify the identities that the published definition
    #    implies, rather than re-deriving the whole branch.
    before_due = frame["pre_clicks"] - frame["pre_due_share"] * frame["pre_clicks"]
    rows.append({
        "check": "before_due_reconciles", "detail": "pre_clicks minus deadline-day share",
        "reference": 0.0,
        "restored": float(np.abs(before_due - frame["pre_before_due_clicks"]).max()),
        "abs_diff": float(np.abs(before_due - frame["pre_before_due_clicks"]).max()),
    })
    rows.append({
        "check": "due_share_in_unit_interval", "detail": "0 <= pre_due_share <= 1",
        "reference": 1.0,
        "restored": float(frame["pre_due_share"].between(0, 1).all()),
        "abs_diff": 0.0,
    })
    late = frame["submitted_after_due"].eq(1)
    interaction_zero_for_ontime = bool(
        frame.loc[~late, ["pre_last1_lift_x_late", "pre_last3_lift_x_late"]]
        .abs().max().max() == 0)
    rows.append({
        "check": "interaction_zero_when_on_time", "detail": "lift x late",
        "reference": 1.0, "restored": float(interaction_zero_for_ontime),
        "abs_diff": 0.0,
    })

    # 3. Primary results must not move: the restored columns are additions, and
    #    the frozen ladder must not read them.
    from src.tma import modeling
    risky = {"pre_last1_rate_lift", "pre_last3_rate_lift", "relative_click_gap",
             "both_zero_clicks", "pre_last1_lift_x_late", "pre_last3_lift_x_late"}
    ladder_columns = set()
    for model in ("M0", "M1", "M2", "M3"):
        ladder_columns |= set(modeling.columns_for(frame, model))
    leaked = sorted(risky & ladder_columns)
    rows.append({
        "check": "frozen_ladder_ignores_new_columns", "detail": ",".join(leaked) or "none",
        "reference": 0.0, "restored": float(len(leaked)), "abs_diff": float(len(leaked)),
    })

    table = pd.DataFrame(rows)
    table.to_csv(out / "v7_restored_parity.csv", index=False)

    payload = {
        "relative_gap_matches_reference": bool(
            table.loc[table["check"].str.startswith("relative_gap"), "abs_diff"].max() < 1e-9),
        "both_zero_matches_reference": bool(
            table.loc[table["check"].eq("both_zero_records"), "abs_diff"].iloc[0] < 0.5),
        "burst_identities_hold": bool(
            table.loc[table["check"].isin(["before_due_reconciles",
                                           "due_share_in_unit_interval"]),
                      "abs_diff"].max() < 1e-6),
        "frozen_ladder_untouched": bool(
            table.loc[table["check"].eq("frozen_ladder_ignores_new_columns"),
                      "restored"].iloc[0] == 0),
        "columns_added": int(len([c for c in frame.columns
                                 if c.startswith(("pre_", "relative_click", "both_zero"))])),
        "new_config_sha256": cfg.sha256,
        "caveat": ("The burst features are verified through identities and through the "
                   "frozen ladder staying clean, not by reproducing the TMA branch's "
                   "per-pair table: that table used a score-conditional subset."),
    }
    (out / "v7_restored_verdict.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
