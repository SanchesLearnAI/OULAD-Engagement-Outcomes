"""V2 — transfer summary (temporal plus leave-one-module-out).

Reads the frozen run's model metrics and pivots the extrapolation splits into one
auditable table, pairing each fold with the fact that only six modules have a
positive-weight TMA and that CCC has no 2013 presentation.

Run from ``oulad-sem/``::

    python scripts/verify_transfer_summary.py
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(root / "outputs" / "tables" / "t08_model_metrics.csv")
    splits = pd.read_csv(root / "outputs" / "tables" / "t05_split_summary.csv")

    transfer = metrics[metrics["split"].eq("temporal")
                       | metrics["split"].str.startswith("lomo_")].copy()
    transfer["kind"] = np.where(transfer["split"].eq("temporal"),
                                "temporal_2013_to_2014", "leave_one_module_out")
    transfer["fold"] = transfer["split"].str.replace("lomo_", "", regex=False)
    transfer.loc[transfer["split"].eq("temporal"), "fold"] = "2014B+2014J"

    # Module facts that matter for interpretation, from the frozen tables.
    presentations = pd.read_csv(root / "outputs" / "tables" / "t02_presentations.csv")
    modules = (presentations.groupby("code_module")
               .agg(presentations=("code_presentation", "nunique"),
                    due_day_median=("due_day", "median"),
                    eligible=("eligible", "sum")).reset_index())
    years = (presentations.assign(year=presentations["code_presentation"].str[:4])
             .groupby("code_module")["year"].agg(lambda s: ",".join(sorted(set(s))))
             .rename("years").reset_index())
    modules = modules.merge(years, on="code_module", how="left")
    modules["has_2013"] = modules["years"].str.contains("2013")
    # EEE and GGG have no graded assessment inside the four-week window; GGG has
    # no positive-weight TMA at all and is therefore absent from this design.
    modules["note"] = ""
    modules.loc[modules["code_module"].eq("CCC"), "note"] = (
        "no 2013 presentation, so the temporal test is also cross-module")
    modules.loc[modules["code_module"].eq("GGG"), "note"] = (
        "no positive-weight TMA, not in the design")

    joined = transfer.merge(modules, left_on="fold", right_on="code_module",
                            how="left")
    # Raw AP tracks the fold's Fail share, so the comparable quantity across
    # folds is the lift over that fold's no-skill level.
    joined["ap_lift_over_no_skill"] = (joined["average_precision"]
                                      - joined["no_skill_pr_auc"])
    joined.loc[joined["kind"].eq("temporal_2013_to_2014"), "note"] = (
        "trained 2013, tested 2014; includes CCC, which has no 2013 presentation")
    temporal_note = ("Temporal test mixes two effects: it trains on 2013 and tests on "
                     "2014, and because CCC has no 2013 presentation, 2014 CCC records "
                     "are also cross-module.")
    summary = (joined.groupby(["kind", "model"])
               .agg(folds=("split", "nunique"),
                    ap_mean=("average_precision", "mean"),
                    ap_min=("average_precision", "min"),
                    ap_max=("average_precision", "max"),
                    auc_mean=("roc_auc", "mean"),
                    ap_lift_mean=("ap_lift_over_no_skill", "mean"),
                    brier_mean=("brier", "mean"),
                    n_test_total=("n_test", "sum"))
               .reset_index())
    joined.to_csv(out / "v2_transfer_rows.csv", index=False)
    summary.to_csv(out / "v2_transfer_summary.csv", index=False)
    modules.to_csv(out / "v2_module_facts.csv", index=False)

    primary = "M2"
    temporal = joined[(joined["kind"].eq("temporal_2013_to_2014"))
                      & joined["model"].eq(primary)]
    lomo = joined[(joined["kind"].eq("leave_one_module_out"))
                  & joined["model"].eq(primary)]
    payload = {
        "primary_model": primary,
        "temporal_ap": float(temporal["average_precision"].iloc[0])
        if len(temporal) else None,
        "temporal_auc": float(temporal["roc_auc"].iloc[0]) if len(temporal) else None,
        "lomo_folds": int(lomo["split"].nunique()),
        "lomo_ap_mean": float(lomo["average_precision"].mean()) if len(lomo) else None,
        "lomo_ap_min": float(lomo["average_precision"].min()) if len(lomo) else None,
        "lomo_ap_max": float(lomo["average_precision"].max()) if len(lomo) else None,
        "weakest_module_by_ap": (lomo.sort_values("average_precision")
                                 ["fold"].iloc[0] if len(lomo) else None),
        "strongest_module_by_ap": (lomo.sort_values("average_precision", ascending=False)
                                   ["fold"].iloc[0] if len(lomo) else None),
        "weakest_module_by_lift": (lomo.sort_values("ap_lift_over_no_skill")
                                  ["fold"].iloc[0] if len(lomo) else None),
        "lomo_ap_lift_mean": float(lomo["ap_lift_over_no_skill"].mean())
        if len(lomo) else None,
        "note_on_raw_ap": ("Fold AP is not comparable across folds because the Fail "
                           "share differs; ap_lift_over_no_skill is the comparable "
                           "quantity."),
        "caveat": temporal_note,
        "smallest_fold": (lomo.merge(modules, left_on="fold", right_on="code_module",
                                     how="left", suffixes=("", "_module"))
                          .sort_values("eligible")
                          [["fold", "eligible", "average_precision"]]
                          .head(1).to_dict("records") if len(lomo) else None),
        "models": sorted(joined["model"].unique().tolist()),
    }
    (out / "v2_transfer_verdict.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
