"""Within-presentation, similar-click descriptive comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation"]


def compare_similar_clicks(frame: pd.DataFrame, output_dir: Path) -> None:
    withdrawn = frame.assign(withdrawn=(frame.final_result == "Withdrawn").astype(int)).groupby(KEY).agg(
        n=("id_student", "size"), withdrawn_n=("withdrawn", "sum"),
        median_clicks=("total_clicks", "median"), zero_click_share=("zero_click", "mean")
    ).reset_index()
    withdrawn["withdrawn_share"] = withdrawn.withdrawn_n / withdrawn.n
    withdrawn.to_csv(output_dir / "withdrawn_by_presentation.csv", index=False)
    population = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].copy()
    population["fail"] = (population.final_result == "Fail").astype(int)
    feature_columns = [
        "active_days", "longest_inactive_run", "daily_concentration", "daily_entropy",
        "resource_type_count", "resource_entropy",
    ] + [f"week_{i}_share" for i in range(4)] + [c for c in frame if c.startswith("resource_share__")]
    rows = []
    stratum_rows = []
    for bins in (5, 10, 20):
        comparisons = []
        for (module, presentation), group in population.groupby(KEY):
            # Rank bins handle repeated click counts without separating equal values.
            edges = np.unique(np.quantile(group.log_clicks, np.linspace(0, 1, bins + 1)))
            if len(edges) < 2:
                group = group.assign(click_stratum=0)
            else:
                group = group.assign(click_stratum=pd.cut(group.log_clicks, edges, labels=False, include_lowest=True, duplicates="drop"))
            for stratum, subset in group.groupby("click_stratum", dropna=False):
                positives = subset[subset.fail == 1]
                negatives = subset[subset.fail == 0]
                if positives.empty or negatives.empty:
                    continue
                click_gap = float(positives.log_clicks.mean() - negatives.log_clicks.mean())
                stratum_rows.append({"bins": bins, "module": module, "presentation": presentation,
                                     "stratum": int(stratum), "n_fail": len(positives), "n_pass": len(negatives),
                                     "log_click_gap": click_gap})
                for feature in feature_columns:
                    comparisons.append((feature, len(subset), float(positives[feature].mean() - negatives[feature].mean())))
        for feature in feature_columns:
            values = [(n, difference) for name, n, difference in comparisons if name == feature]
            if values:
                rows.append({"bins": bins, "feature": feature,
                             "fail_minus_pass": float(np.average([v for _, v in values], weights=[n for n, _ in values])),
                             "matched_rows": int(sum(n for n, _ in values)),
                             "population_rows": len(population)})
    pd.DataFrame(rows).to_csv(output_dir / "similar_click_comparison.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output_dir / "click_strata.csv", index=False)

    exceptions = []
    for (module, presentation), group in population.groupby(KEY):
        low, high = group.total_clicks.quantile([0.25, 0.75])
        for label, subset in (("high_activity_fail", group[(group.total_clicks >= high) & (group.fail == 1)]),
                              ("low_activity_success", group[(group.total_clicks <= low) & (group.fail == 0)])):
            exceptions.append({"module": module, "presentation": presentation, "group": label,
                               "count": len(subset), "share_of_presentation": len(subset) / len(group),
                               "median_clicks": float(subset.total_clicks.median()) if len(subset) else None})
    (output_dir / "exception_summary.json").write_text(json.dumps(exceptions, indent=2), encoding="utf-8")
