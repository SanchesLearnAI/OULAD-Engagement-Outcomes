"""Within-presentation, similar-click descriptive comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .features import CONCENTRATION_FEATURES

KEY = ["code_module", "code_presentation"]


def compare_similar_clicks(frame: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    withdrawn = frame.assign(withdrawn=(frame.final_result == "Withdrawn").astype(int)).groupby(KEY).agg(
        n=("id_student", "size"), withdrawn_n=("withdrawn", "sum"),
        median_clicks=("total_clicks", "median"), zero_click_share=("zero_click", "mean")
    ).reset_index()
    withdrawn["withdrawn_share"] = withdrawn.withdrawn_n / withdrawn.n
    withdrawn.to_csv(output_dir / "withdrawn_by_presentation.csv", index=False)
    eligible = frame[frame.eligible_at_day_28 == 1]
    eligible_summary = eligible.assign(withdrawn=(eligible.final_result == "Withdrawn").astype(int)).groupby(KEY).agg(
        n=("id_student", "size"), withdrawn_n=("withdrawn", "sum"), zero_click_n=("zero_click", "sum")
    ).reset_index()
    eligible_summary.to_csv(output_dir / "eligible_by_presentation.csv", index=False)
    population = frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])].copy()
    population["fail"] = (population.final_result == "Fail").astype(int)
    feature_columns = [
        "active_days", "longest_inactive_run", "daily_concentration", "daily_entropy",
        "resource_type_count", "resource_entropy",
    ] + CONCENTRATION_FEATURES + [f"week_{i}_share" for i in range(4)] + [c for c in frame if c.startswith("resource_share__")]
    rows = []
    stratum_rows = []
    bootstrap_source = []
    balanced_comparisons = []
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
                pooled_sd = np.sqrt((positives.log_clicks.var(ddof=0) + negatives.log_clicks.var(ddof=0)) / 2)
                smd = click_gap / pooled_sd if pooled_sd > 0 else 0.0
                stratum_rows.append({"bins": bins, "module": module, "presentation": presentation,
                                     "stratum": int(stratum), "n_fail": len(positives), "n_pass": len(negatives),
                                     "log_click_gap": click_gap, "click_smd": smd,
                                     "min_clicks": float(subset.total_clicks.min()), "max_clicks": float(subset.total_clicks.max())})
                if bins == 20:
                    local = subset[["id_student", "fail"] + feature_columns].copy()
                    local["stratum_key"] = f"{module}/{presentation}/{stratum}"
                    bootstrap_source.append(local)
                for feature in feature_columns:
                    difference = float(positives[feature].mean() - negatives[feature].mean())
                    comparisons.append((feature, len(subset), difference))
                    if bins == 20 and abs(smd) <= 0.1:
                        balanced_comparisons.append((feature, len(subset), difference))
        for feature in feature_columns:
            values = [(n, difference) for name, n, difference in comparisons if name == feature]
            if values:
                rows.append({"bins": bins, "feature": feature,
                             "fail_minus_pass": float(np.average([v for _, v in values], weights=[n for n, _ in values])),
                             "matched_rows": int(sum(n for n, _ in values)),
                             "population_rows": len(population)})
    pd.DataFrame(rows).to_csv(output_dir / "similar_click_comparison.csv", index=False)
    pd.DataFrame([{"feature": feature, "fail_minus_pass": float(np.average([v for name, _, v in balanced_comparisons if name == feature],
                    weights=[n for name, n, _ in balanced_comparisons if name == feature])),
                   "covered_rows": int(sum(n for name, n, _ in balanced_comparisons if name == feature)),
                   "population_rows": len(population), "maximum_absolute_click_smd": 0.1}
                  for feature in feature_columns if any(name == feature for name, _, _ in balanced_comparisons)]).to_csv(
                      output_dir / "balanced_similar_click_comparison.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output_dir / "click_strata.csv", index=False)
    strata = pd.DataFrame(stratum_rows)
    pd.DataFrame([{"bins": b, "covered_rows": int((g.n_fail + g.n_pass).sum()),
                   "weighted_absolute_click_smd": float(np.average(g.click_smd.abs(), weights=g.n_fail + g.n_pass)),
                   "strata_above_0_1": int((g.click_smd.abs() > 0.1).sum()), "strata": len(g)}
                  for b, g in strata.groupby("bins")]).to_csv(output_dir / "click_balance.csv", index=False)
    # Student-level bootstrap keeps repeated presentations together and fixed strata.
    if bootstrap_source:
        source = pd.concat(bootstrap_source, ignore_index=True)
        students = source.id_student.unique()
        rng = np.random.default_rng(seed)
        chosen_features = ["active_days", "longest_inactive_run", "week_3_share", "resource_share__forumng"]
        estimates = {feature: [] for feature in chosen_features if feature in source}
        for _ in range(100):
            counts = pd.Series(rng.choice(students, size=len(students), replace=True)).value_counts()
            weighted = source.join(counts.rename("weight"), on="id_student").dropna(subset=["weight"])
            for feature in estimates:
                groups = weighted.groupby(["stratum_key", "fail"])
                mean = groups.apply(lambda x: np.average(x[feature], weights=x.weight), include_groups=False).unstack()
                weight = groups.weight.sum().unstack().dropna()
                valid = mean.dropna().index.intersection(weight.index)
                if len(valid):
                    estimates[feature].append(float(np.average(mean.loc[valid, 1] - mean.loc[valid, 0], weights=weight.loc[valid].sum(axis=1))))
        pd.DataFrame([{"bins": 20, "feature": feature, "ci_low": float(np.quantile(values, .025)),
                       "ci_high": float(np.quantile(values, .975)), "bootstrap_repeats": len(values)}
                      for feature, values in estimates.items() if values]).to_csv(output_dir / "similar_click_ci.csv", index=False)

    exceptions = []
    for (module, presentation), group in population.groupby(KEY):
        low, high = group.total_clicks.quantile([0.25, 0.75])
        for label, subset in (("high_activity_fail", group[(group.total_clicks >= high) & (group.fail == 1)]),
                              ("low_activity_success", group[(group.total_clicks <= low) & (group.fail == 0)])):
            exceptions.append({"module": module, "presentation": presentation, "group": label,
                               "count": len(subset), "share_of_presentation": len(subset) / len(group),
                               "median_clicks": float(subset.total_clicks.median()) if len(subset) else None})
    (output_dir / "exception_summary.json").write_text(json.dumps(exceptions, indent=2), encoding="utf-8")
