"""Compare outcomes at comparable VLE volume 21/28 days after first weighted TMA.

Uses the existing node cohorts and matching rule. Row-level output stays under
the ignored output directory; only aggregate results should be committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import KEY, TRACK, SUBMIT, _fit_predict
from .matched import _component_intervals, make_pairs

CORE = ["pre_clicks", "post_clicks"] + SUBMIT + TRACK
RESOURCE_SUMMARY = ["resource_type_count", "resource_entropy",
                    "post_resource_type_count", "post_resource_entropy"]


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_resource_counts(data_dir: Path, lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One source scan, with counts for pre, post days 1–21, and days 22–28."""
    metadata = pd.read_csv(data_dir / "vle.csv", usecols=KEY[:2] + ["id_site", "activity_type"])
    site_key = KEY[:2] + ["id_site"]
    if metadata.duplicated(site_key).any() or metadata.activity_type.isna().any():
        raise ValueError("Invalid VLE resource metadata")
    lookup = lookup[KEY + ["due_day"]].drop_duplicates()
    if lookup.duplicated(KEY).any() or lookup.due_day.isna().any():
        raise ValueError("Invalid or conflicting node deadlines")
    con = duckdb.connect()
    try:
        con.register("node_lookup", lookup)
        con.register("vle_metadata", metadata)
        counts = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student,
                   coalesce(v.activity_type, '__unmatched__') AS activity_type,
                   sum(CASE WHEN e.date - l.due_day BETWEEN -13 AND 0
                       THEN e.sum_click ELSE 0 END) AS pre_clicks,
                   sum(CASE WHEN e.date - l.due_day BETWEEN 1 AND 21
                       THEN e.sum_click ELSE 0 END) AS post21_clicks,
                   sum(CASE WHEN e.date - l.due_day BETWEEN 22 AND 28
                       THEN e.sum_click ELSE 0 END) AS late7_clicks,
                   count(*) FILTER (WHERE v.activity_type IS NULL) AS unmatched_rows,
                   count(*) FILTER (WHERE e.sum_click < 0 OR e.sum_click IS NULL) AS bad_rows
            FROM read_csv_auto('{_sql_path(data_dir / 'studentVle.csv')}', header=true) e
            JOIN node_lookup l USING (code_module, code_presentation, id_student)
            LEFT JOIN vle_metadata v USING (code_module, code_presentation, id_site)
            WHERE e.date - l.due_day BETWEEN -13 AND 28
            GROUP BY 1, 2, 3, 4
        """).df()
    finally:
        con.close()
    if counts.bad_rows.sum() or counts[["pre_clicks", "post21_clicks", "late7_clicks"]].isna().any().any():
        raise ValueError("Missing or negative VLE click counts")
    return counts.drop(columns="bad_rows"), {
        "resource_types_observed": sorted(counts.activity_type.unique().tolist()),
        "unmatched_event_rows": int(counts.unmatched_rows.sum()),
        "resource_count_rows": int(len(counts)),
    }


def add_resource_features(frame: pd.DataFrame, counts: pd.DataFrame, post_days: int,
                          resource_types: list[str]) -> pd.DataFrame:
    if post_days not in (21, 28):
        raise ValueError("post_days must be 21 or 28")
    if frame.duplicated(KEY).any() or not frame.final_result.isin(["Fail", "Pass", "Distinction"]).all():
        raise ValueError("Expected unique binary node records")
    chosen = counts.copy()
    chosen["all_clicks"] = chosen.pre_clicks + chosen.post21_clicks
    chosen["post_clicks"] = chosen.post21_clicks.copy()
    if post_days == 28:
        chosen["all_clicks"] += chosen.late7_clicks
        chosen["post_clicks"] += chosen.late7_clicks
    index = pd.MultiIndex.from_frame(frame[KEY])
    result = frame.set_index(KEY).copy()
    for period, expected in [("all", frame.total_clicks.to_numpy()),
                             ("post", frame.post_clicks.to_numpy())]:
        wide = chosen.pivot(index=KEY, columns="activity_type", values=period + "_clicks")
        wide = wide.reindex(index=index, columns=resource_types, fill_value=0).fillna(0)
        values = wide.to_numpy(dtype=float)
        if not np.array_equal(values.sum(axis=1), expected.astype(float)):
            raise ValueError(f"Resource {period} click totals disagree with node features")
        shares = np.divide(values, expected[:, None],
                           out=np.zeros_like(values), where=expected[:, None] > 0)
        entropy = -(np.where(shares > 0, shares * np.log(np.where(shares > 0, shares, 1)), 0)).sum(axis=1)
        prefix = "" if period == "all" else "post_"
        result[prefix + "resource_type_count"] = (values > 0).sum(axis=1)
        result[prefix + "resource_entropy"] = entropy
        for j, name in enumerate(resource_types):
            result[prefix + f"resource_clicks__{name}"] = values[:, j]
            result[prefix + f"resource_share__{name}"] = shares[:, j]
    return result.reset_index()


def compare_node_matched(frame: pd.DataFrame, output_dir: Path, post_days: int,
                         resource_types: list[str], seed: int = 42,
                         repeats: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = CORE + RESOURCE_SUMMARY + [f"resource_share__{n}" for n in resource_types] + [
        f"post_resource_share__{n}" for n in resource_types]
    balances, effects, slices = [], [], []
    for tolerance in (0.0, .05, .10):
        pairs = make_pairs(frame, tolerance, seed)
        pairs.to_parquet(output_dir / f"pairs_{post_days}d_{int(tolerance*100):02}.parquet", index=False)
        if pairs.empty:
            balances.append({"post_days": post_days, "tolerance": tolerance, "pairs": 0})
            continue
        fail = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
        passing = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
        difference = fail[features].to_numpy(dtype=float) - passing[features].to_numpy(dtype=float)
        interval = _component_intervals(pairs, difference, seed, repeats)
        sd = np.sqrt((fail.log_clicks.var(ddof=0) + passing.log_clicks.var(ddof=0)) / 2)
        balances.append({"post_days": post_days, "tolerance": tolerance,
                         "binary_rows": len(frame), "fail_rows": int(frame.final_result.eq("Fail").sum()),
                         "pairs": len(pairs),
                         "fail_coverage": len(pairs) / frame.final_result.eq("Fail").sum(),
                         "zero_click_pairs": int(pairs.fail_clicks.eq(0).sum()),
                         "mean_relative_click_gap": float(pairs.relative_gap.mean()),
                         "max_relative_click_gap": float(pairs.relative_gap.max()),
                         "log_click_smd": float((fail.log_clicks.mean() - passing.log_clicks.mean()) / sd) if sd else 0.0,
                         "fail_click_mean": float(fail.total_clicks.mean()),
                         "pass_click_mean": float(passing.total_clicks.mean())})
        for j, name in enumerate(features):
            effects.append({"post_days": post_days, "tolerance": tolerance,
                            "feature": name, "pairs": len(pairs),
                            "fail_mean": float(fail[name].mean()),
                            "pass_mean": float(passing[name].mean()),
                            "fail_minus_pass": float(difference[:, j].mean()),
                            "ci_low": float(interval[0, j]), "ci_high": float(interval[1, j])})
        for (module, presentation), ids in pairs.groupby(KEY[:2]).indices.items():
            for j, name in enumerate(features):
                slices.append({"post_days": post_days, "tolerance": tolerance,
                               "code_module": module, "code_presentation": presentation,
                               "feature": name, "pairs": len(ids),
                               "fail_minus_pass": float(difference[ids, j].mean())})
    return pd.DataFrame(balances), pd.DataFrame(effects), pd.DataFrame(slices)


def compare_post_volume_sensitivity(frame: pd.DataFrame, post_days: int,
                                    resource_types: list[str], seed: int = 42,
                                    repeats: int = 1000) -> pd.DataFrame:
    """Exploratory subset of total-click pairs with similar positive post clicks."""
    pairs = make_pairs(frame, .05, seed)
    fail = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    post_fail = fail.post_clicks.to_numpy()
    post_pass = passing.post_clicks.to_numpy()
    ratio = np.maximum(post_fail, post_pass) / np.maximum(np.minimum(post_fail, post_pass), 1)
    features = SUBMIT + TRACK + RESOURCE_SUMMARY + [
        f"post_resource_share__{n}" for n in resource_types]
    rows = []
    for tolerance in (.05, .20):
        selected = (np.minimum(post_fail, post_pass) > 0) & (ratio <= 1 + tolerance + 1e-12)
        subset = pairs.loc[selected].reset_index(drop=True)
        if subset.empty:
            continue
        difference = (fail.loc[selected, features].to_numpy(dtype=float) -
                      passing.loc[selected, features].to_numpy(dtype=float))
        interval = _component_intervals(subset, difference, seed, repeats)
        for j, name in enumerate(features):
            rows.append({"post_days": post_days, "total_click_tolerance": .05,
                         "post_click_tolerance": tolerance, "pairs": len(subset),
                         "fraction_of_main_pairs": len(subset) / len(pairs),
                         "feature": name, "fail_minus_pass": float(difference[:, j].mean()),
                         "ci_low": float(interval[0, j]), "ci_high": float(interval[1, j])})
    return pd.DataFrame(rows)


def evaluate_resource_increment(frame: pd.DataFrame, post_days: int,
                                resource_types: list[str], seed: int = 42,
                                bootstrap: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ["module_presentation", "log_clicks", "log_clicks_squared", "pre_clicks", "post_clicks"]
    resource = RESOURCE_SUMMARY + [f"resource_share__{n}" for n in resource_types] + [
        f"post_resource_share__{n}" for n in resource_types]
    versions = {"B2_clicks_submission": base + SUBMIT,
                "B3_clicks_submission_timing": base + SUBMIT + TRACK,
                "B4_add_resources": base + SUBMIT + TRACK + resource}
    rows, intervals = [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                              random_state=split_seed).split(frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across split")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, cols) for name, cols in versions.items()}
        for name, scores in predictions.items():
            rows.append({"post_days": post_days, "seed": split_seed, "model": name,
                         "train_rows": len(train), "test_rows": len(test),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, scores))})
        if split_seed == seed:
            members = list(test.groupby("id_student").indices.values())
            rng = np.random.default_rng(seed + 17000 + post_days)
            draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                     for _ in range(bootstrap)]
            for model, comparator in [("B3_clicks_submission_timing", "B2_clicks_submission"),
                                      ("B4_add_resources", "B3_clicks_submission_timing")]:
                p, q = predictions[model], predictions[comparator]
                deltas = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                          for ix in draws if len(np.unique(y[ix])) == 2]
                intervals.append({"post_days": post_days, "model": model,
                                  "comparator": comparator,
                                  "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                                  "ci_low": float(np.quantile(deltas, .025)),
                                  "ci_high": float(np.quantile(deltas, .975))})
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def run(data_dir: Path, three: Path, four: Path, output_dir: Path,
        seed: int = 42, repeats: int = 1000, bootstrap: int = 500) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    inputs = {21: three, 28: four}
    vle_hash = hashlib.sha256((data_dir / "studentVle.csv").read_bytes()).hexdigest()
    for days, path in inputs.items():
        node_protocol = json.loads((path / "node_protocol.json").read_text(encoding="utf-8"))
        if node_protocol.get("sources_sha256", {}).get("studentVle") != vle_hash:
            raise ValueError(f"{days}-day node features use a different studentVle source")
    frames = {days: pd.read_parquet(path / "node_features.parquet")
              for days, path in inputs.items()}
    for days, frame in frames.items():
        if not frame.cutoff_day.sub(frame.due_day).eq(days).all():
            raise ValueError(f"Wrong cutoff in {days}-day input")
        frames[days] = frame[frame.final_result.isin(["Fail", "Pass", "Distinction"])].reset_index(drop=True)
    lookup = pd.concat([frame[KEY + ["due_day"]] for frame in frames.values()]).drop_duplicates()
    counts, quality = load_resource_counts(data_dir, lookup)
    resource_types = quality["resource_types_observed"]
    output_dir.mkdir(parents=True, exist_ok=True)
    balances, effects, slices, sensitivities, models, intervals = [], [], [], [], [], []
    for days, frame in frames.items():
        enriched = add_resource_features(frame, counts, days, resource_types)
        enriched.to_parquet(output_dir / f"features_{days}d.parquet", index=False)
        a, b, c = compare_node_matched(enriched, output_dir, days, resource_types, seed, repeats)
        balances.append(a); effects.append(b); slices.append(c)
        sensitivities.append(compare_post_volume_sensitivity(enriched, days, resource_types, seed, repeats))
        d, e = evaluate_resource_increment(enriched, days, resource_types, seed, bootstrap)
        models.append(d); intervals.append(e)
    for name, tables in [("balance", balances), ("effects", effects), ("presentations", slices),
                         ("post_volume_sensitivity", sensitivities),
                         ("models", models), ("ap_intervals", intervals)]:
        pd.concat(tables, ignore_index=True).to_csv(output_dir / f"{name}.csv", index=False)
    protocol = {"question": "Fail vs Pass/Distinction at comparable VLE volume after first weighted TMA",
                "post_days": [21, 28], "pre_days_inclusive": [-13, 0],
                "match_on": "same module presentation and total VLE clicks from due-13 through cutoff",
                "tolerances": [0, .05, .10], "seed": seed, "component_bootstrap": repeats,
                "post_volume_sensitivity": "Exploratory 5% total-click pairs with both positive post clicks and 5%/20% post-click ratio limits",
                "model_student_bootstrap": bootstrap,
                "binary_rows": {str(days): len(frame) for days, frame in frames.items()},
                "resource_quality": quality,
                "source_sha256": {"studentVle": vle_hash,
                                  "vle": hashlib.sha256((data_dir / "vle.csv").read_bytes()).hexdigest()},
                "node_protocol_sha256": {str(days): hashlib.sha256((path / "node_protocol.json").read_bytes()).hexdigest()
                                         for days, path in inputs.items()},
                "node_features_sha256": {str(days): hashlib.sha256((path / "node_features.parquet").read_bytes()).hexdigest()
                                         for days, path in inputs.items()},
                "limitations": "Retrospective survivors with eventual non-Withdrawn labels; exploratory pointwise intervals conditional on fixed greedy pairs; no causal inference or multiple-testing correction"}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--three", type=Path, required=True, help="Directory with 21-day node_features.parquet")
    parser.add_argument("--four", type=Path, required=True, help="Directory with 28-day node_features.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    run(args.data_dir, args.three, args.four, args.output_dir,
        args.seed, args.repeats, args.bootstrap)


if __name__ == "__main__":
    main()
