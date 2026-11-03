"""TMA-centered 14-day pre/post study with post-click matching.

Read an existing two-week assessment-node run. Student-level features, pairs,
and cluster assignments stay in the ignored output directory. Public results
contain aggregate tables only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, average_precision_score, brier_score_loss, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .assessment_node import KEY, _fit_predict
from .matched import _component_intervals, make_pairs

PRIMARY = ["post_active_days", "post_both_weeks", "post_longest_inactive_run", "post_week2_share"]
TIMING = PRIMARY + ["post_peak7_share", "post_entropy_norm", "post_first_active_day"]
SUBMIT = ["submitted_by_cutoff", "submitted_after_due"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _longest_zero_run(clicks: np.ndarray) -> np.ndarray:
    run = np.zeros(len(clicks), dtype=int)
    longest = run.copy()
    for day in range(clicks.shape[1]):
        run = np.where(clicks[:, day] > 0, 0, run + 1)
        longest = np.maximum(longest, run)
    return longest


def _time_features(post: np.ndarray) -> dict[str, np.ndarray]:
    total = post.sum(axis=1)
    shares = np.divide(post, total[:, None], out=np.zeros_like(post, dtype=float), where=total[:, None] > 0)
    entropy = -(shares * np.log(np.where(shares > 0, shares, 1))).sum(axis=1)
    cumulative = np.pad(np.cumsum(post, axis=1), ((0, 0), (1, 0)))
    peak7 = (cumulative[:, 7:] - cumulative[:, :-7]).max(axis=1)
    return {
        "post_longest_inactive_run": _longest_zero_run(post),
        "post_week2_share": np.divide(post[:, 7:].sum(axis=1), total,
                                        out=np.zeros(len(post)), where=total > 0),
        "post_peak7_share": np.divide(peak7, total, out=np.zeros(len(post)), where=total > 0),
        "post_entropy_norm": entropy / np.log(14),
        "post_first_active_day": np.where((post > 0).any(axis=1), (post > 0).argmax(axis=1) + 1, 15),
    }


def build_behavior_features(node: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Join original VLE events to the node cohort and reconcile all click totals."""
    frame = node[node.final_result.isin(["Fail", "Pass", "Distinction"])].reset_index(drop=True).copy()
    if frame.duplicated(KEY).any() or not frame.cutoff_day.sub(frame.due_day).eq(14).all():
        raise ValueError("Expected unique first-TMA +14-day node records")
    site_key = KEY[:2] + ["id_site"]
    metadata = pd.read_csv(data_dir / "vle.csv", usecols=site_key + ["activity_type"])
    if metadata.duplicated(site_key).any() or metadata.activity_type.isna().any():
        raise ValueError("Invalid VLE metadata")
    con = duckdb.connect()
    try:
        con.register("node_lookup", frame[KEY + ["due_day"]])
        con.register("vle_metadata", metadata)
        events = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student,
                   e.date - l.due_day AS relative_day,
                   coalesce(v.activity_type, '__unmatched__') AS activity_type,
                   sum(e.sum_click) AS clicks,
                   count(*) FILTER (WHERE v.activity_type IS NULL) AS unmatched_rows,
                   count(*) FILTER (WHERE e.sum_click IS NULL OR e.sum_click < 0) AS bad_rows
            FROM read_csv_auto('{_sql_path(data_dir / 'studentVle.csv')}', header=true) e
            JOIN node_lookup l USING (code_module, code_presentation, id_student)
            LEFT JOIN vle_metadata v USING (code_module, code_presentation, id_site)
            WHERE e.date - l.due_day BETWEEN -13 AND 14
            GROUP BY 1, 2, 3, 4, 5
        """).df()
    finally:
        con.close()
    if events.bad_rows.sum() or events.clicks.isna().any():
        raise ValueError("Invalid VLE clicks")
    index = pd.MultiIndex.from_frame(frame[KEY])
    daily = events.groupby(KEY + ["relative_day"], as_index=False).clicks.sum()
    wide = (daily.pivot(index=KEY, columns="relative_day", values="clicks")
            .reindex(index=index, columns=list(range(-13, 15))).fillna(0))
    all_days = wide.to_numpy(dtype=float)
    pre, post = all_days[:, :14], all_days[:, 14:]
    if not np.array_equal(pre.sum(axis=1), frame.pre_clicks.to_numpy(dtype=float)):
        raise ValueError("Pre clicks disagree with two-week node features")
    if not np.array_equal(post.sum(axis=1), frame.post_clicks.to_numpy(dtype=float)):
        raise ValueError("Post clicks disagree with two-week node features")
    if not np.array_equal((post > 0).sum(axis=1), frame.post_active_days.to_numpy(dtype=float)):
        raise ValueError("Post active days disagree with node features")
    for name, values in _time_features(post).items():
        frame[name] = values
    frame["log_pre_clicks"] = np.log1p(frame.pre_clicks)
    frame["log_post_clicks"] = np.log1p(frame.post_clicks)
    frame["log_post_clicks_squared"] = frame.log_post_clicks ** 2
    resource_types = sorted(events.activity_type.unique().tolist())
    resource = (events[events.relative_day.between(1, 14)]
                .groupby(KEY + ["activity_type"], as_index=False).clicks.sum())
    resource_wide = (resource.pivot(index=KEY, columns="activity_type", values="clicks")
                     .reindex(index=index, columns=resource_types).fillna(0))
    values = resource_wide.to_numpy(dtype=float)
    if not np.array_equal(values.sum(axis=1), frame.post_clicks.to_numpy(dtype=float)):
        raise ValueError("Post resource totals disagree with node features")
    shares = np.divide(values, frame.post_clicks.to_numpy(dtype=float)[:, None],
                       out=np.zeros_like(values), where=frame.post_clicks.to_numpy()[:, None] > 0)
    frame["post_resource_type_count"] = (values > 0).sum(axis=1)
    frame["post_resource_entropy"] = -(shares * np.log(np.where(shares > 0, shares, 1))).sum(axis=1)
    for j, name in enumerate(resource_types):
        frame[f"post_resource_share__{name}"] = shares[:, j]
    quality = {"binary_records": len(frame), "fail_records": int(frame.final_result.eq("Fail").sum()),
               "zero_post_click_records": int(frame.post_clicks.eq(0).sum()),
               "unmatched_resource_event_rows": int(events.unmatched_rows.sum()),
               "resource_types_observed": resource_types,
               "event_resource_day_rows": len(events)}
    return frame, post, quality


def _matching_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # make_pairs matches on total_clicks; this adapter names the intended
    # post-deadline quantity explicitly while retaining all other features.
    return frame.assign(total_clicks=frame.post_clicks)


def compare_pairs(frame: pd.DataFrame, post: np.ndarray, output_dir: Path,
                  seed: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resource_cols = sorted(c for c in frame if c.startswith("post_resource_share__"))
    features = ["pre_clicks", "pre_active_days", "last3_share"] + SUBMIT + TIMING + [
        "post_resource_type_count", "post_resource_entropy"] + resource_cols
    balances, effects, slices, pre_sensitivity, post_sensitivity, profile = [], [], [], [], [], []
    main_pairs = pd.DataFrame()
    for tolerance in (0.0, .05, .10):
        pairs = make_pairs(_matching_frame(frame), tolerance, seed)
        pairs.to_parquet(output_dir / f"post_pairs_{int(tolerance*100):02}.parquet", index=False)
        if pairs.empty:
            continue
        fail = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
        passing = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
        differences = fail[features].to_numpy(dtype=float) - passing[features].to_numpy(dtype=float)
        intervals = _component_intervals(pairs, differences, seed, repeats)
        post_sd = np.sqrt((fail.log_post_clicks.var(ddof=0) + passing.log_post_clicks.var(ddof=0)) / 2)
        pre_sd = np.sqrt((fail.log_pre_clicks.var(ddof=0) + passing.log_pre_clicks.var(ddof=0)) / 2)
        balances.append({"tolerance": tolerance, "binary_records": len(frame),
                         "fail_records": int(frame.final_result.eq("Fail").sum()),
                         "pairs": len(pairs), "fail_coverage": len(pairs) / frame.final_result.eq("Fail").sum(),
                         "zero_post_pairs": int(pairs.fail_clicks.eq(0).sum()),
                         "mean_post_click_gap": float(pairs.relative_gap.mean()),
                         "post_log_click_smd": float((fail.log_post_clicks.mean() - passing.log_post_clicks.mean()) / post_sd) if post_sd else 0,
                         "pre_log_click_smd": float((fail.log_pre_clicks.mean() - passing.log_pre_clicks.mean()) / pre_sd) if pre_sd else 0,
                         "fail_pre_click_mean": float(fail.pre_clicks.mean()),
                         "pass_pre_click_mean": float(passing.pre_clicks.mean()),
                         "fail_post_click_mean": float(fail.post_clicks.mean()),
                         "pass_post_click_mean": float(passing.post_clicks.mean())})
        for j, name in enumerate(features):
            effects.append({"tolerance": tolerance, "feature": name, "pairs": len(pairs),
                            "fail_mean": float(fail[name].mean()), "pass_mean": float(passing[name].mean()),
                            "fail_minus_pass": float(differences[:, j].mean()),
                            "ci_low": float(intervals[0, j]), "ci_high": float(intervals[1, j])})
        if tolerance != .05:
            continue
        main_pairs = pairs
        positive = pairs.fail_clicks.gt(0).to_numpy()
        positive_pairs = pairs.loc[positive].reset_index(drop=True)
        positive_diff = differences[positive]
        positive_interval = _component_intervals(positive_pairs, positive_diff, seed, repeats)
        for name in PRIMARY + SUBMIT:
            j = features.index(name)
            post_sensitivity.append({"subset": "positive_post_clicks", "pairs": len(positive_pairs),
                                     "feature": name, "fail_minus_pass": float(positive_diff[:, j].mean()),
                                     "ci_low": float(positive_interval[0, j]),
                                     "ci_high": float(positive_interval[1, j])})
        for (module, presentation), ids in pairs.groupby(KEY[:2]).indices.items():
            for name in PRIMARY + SUBMIT:
                slices.append({"code_module": module, "code_presentation": presentation,
                               "feature": name, "pairs": len(ids),
                               "fail_minus_pass": float((fail.iloc[ids][name].to_numpy() - passing.iloc[ids][name].to_numpy()).mean())})
        pre_fail, pre_pass = fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy()
        ratio = np.maximum(pre_fail, pre_pass) / np.maximum(np.minimum(pre_fail, pre_pass), 1)
        for threshold in (.20, .50):
            selected = ((pre_fail == 0) & (pre_pass == 0)) | ((np.minimum(pre_fail, pre_pass) > 0) & (ratio <= 1 + threshold + 1e-12))
            subset = pairs.loc[selected].reset_index(drop=True)
            if subset.empty:
                continue
            sub_diff = differences[selected]
            sub_interval = _component_intervals(subset, sub_diff, seed, repeats)
            for name in PRIMARY + SUBMIT:
                j = features.index(name)
                pre_sensitivity.append({"pre_click_tolerance": threshold, "pairs": len(subset),
                                        "fraction_of_main_pairs": len(subset) / len(pairs),
                                        "feature": name, "fail_minus_pass": float(sub_diff[:, j].mean()),
                                        "ci_low": float(sub_interval[0, j]), "ci_high": float(sub_interval[1, j])})
        for day in range(14):
            for label, ix in (("Fail", pairs.fail_index.to_numpy()), ("Pass/Distinction", pairs.pass_index.to_numpy())):
                values = post[ix, day]
                profile.append({"post_day": day + 1, "outcome": label, "pairs": len(pairs),
                                "mean_clicks": float(values.mean()), "active_fraction": float((values > 0).mean())})
    return (pd.DataFrame(balances), pd.DataFrame(effects), pd.DataFrame(slices),
            pd.DataFrame(pre_sensitivity), pd.DataFrame(post_sensitivity),
            pd.DataFrame(profile), main_pairs)


def _rf_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int) -> np.ndarray:
    numeric = [c for c in columns if c != "module_presentation"]
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("context", OneHotEncoder(handle_unknown="ignore"), ["module_presentation"])])
    model = make_pipeline(prep, RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                                       random_state=seed, n_jobs=-1))
    model.fit(train[columns], train.final_result.eq("Fail").astype(int))
    return model.predict_proba(test[columns])[:, 1]


def evaluate_models(frame: pd.DataFrame, seed: int, bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    resource = ["post_resource_type_count", "post_resource_entropy"] + sorted(
        c for c in frame if c.startswith("post_resource_share__"))[:-1]
    baseline = ["module_presentation", "log_pre_clicks", "pre_active_days", "last3_share",
                "log_post_clicks", "log_post_clicks_squared"] + SUBMIT
    versions = {"M0_pre_post_volume_submission": baseline,
                "M1_add_post_timing": baseline + TIMING,
                "M2_add_post_resources": baseline + TIMING + resource}
    rows, contrasts = [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across split")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, cols) for name, cols in versions.items()}
        if split_seed == seed:
            predictions["RF_M0"] = _rf_predict(train, test, baseline, split_seed)
            predictions["RF_M1"] = _rf_predict(train, test, baseline + TIMING, split_seed)
        for name, scores in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train), "test_rows": len(test),
                         "test_students": test.id_student.nunique(), "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, scores)),
                         "brier": float(brier_score_loss(y, scores))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 21400)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for model, comparator in [("M1_add_post_timing", "M0_pre_post_volume_submission"),
                                  ("M2_add_post_resources", "M1_add_post_timing"),
                                  ("RF_M1", "RF_M0")]:
            p, q = predictions[model], predictions[comparator]
            deltas = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                      for ix in draws if len(np.unique(y[ix])) == 2]
            contrasts.append({"model": model, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ci_low": float(np.quantile(deltas, .025)),
                              "ci_high": float(np.quantile(deltas, .975))})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def explore_clusters(frame: pd.DataFrame, post: np.ndarray, pairs: pd.DataFrame,
                     output_dir: Path, seed: int, repeats: int, max_pairs: int = 800
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive = pairs[pairs.fail_clicks.gt(0)].reset_index(drop=True)
    if len(positive) < 30:
        return tuple(pd.DataFrame() for _ in range(5))
    rng = np.random.default_rng(seed)
    chosen = positive.iloc[np.sort(rng.choice(len(positive), size=min(max_pairs, len(positive)), replace=False))].reset_index(drop=True)
    n = len(chosen)
    indices = np.concatenate([chosen.fail_index.to_numpy(), chosen.pass_index.to_numpy()])
    sample = frame.iloc[indices].reset_index(drop=True).copy()
    cluster_features = ["post_active_days", "post_longest_inactive_run", "post_week2_share",
                        "post_peak7_share", "post_entropy_norm"]
    x = StandardScaler().fit_transform(sample[cluster_features])
    candidates, fitted = [], {}
    for algorithm in ("kmeans", "agglomerative"):
        for k in range(2, 6):
            make = (lambda s: KMeans(n_clusters=k, n_init=10, random_state=s)) if algorithm == "kmeans" else (
                lambda s: AgglomerativeClustering(n_clusters=k))
            labels = make(seed).fit_predict(x)
            counts = np.bincount(labels)
            if counts.min() < max(5, int(np.ceil(.03 * len(sample)))):
                continue
            stability = []
            for repeat in range(3):
                sub = np.sort(rng.choice(len(sample), size=int(.8 * len(sample)), replace=False))
                rerun = make(seed + repeat + 1).fit_predict(x[sub])
                stability.append(float(adjusted_rand_score(labels[sub], rerun)))
            silhouette = float(silhouette_score(x, labels, sample_size=min(800, len(sample)), random_state=seed))
            candidates.append({"algorithm": algorithm, "k": k, "silhouette": silhouette,
                               "stability_ari": float(np.mean(stability)),
                               "min_cluster_share": float(counts.min() / len(sample))})
            fitted[(algorithm, k)] = labels
    if not candidates:
        return tuple(pd.DataFrame() for _ in range(5))
    best = max(candidates, key=lambda row: (row["silhouette"], row["stability_ari"]))
    labels = fitted[(best["algorithm"], best["k"])]
    sample["cluster"] = labels
    sample[KEY + ["cluster", "post_clicks", "final_result"]].to_parquet(output_dir / "cluster_sample.parquet", index=False)
    profile = sample.groupby("cluster").agg(n=("id_student", "size"),
        mean_post_clicks=("post_clicks", "mean"), median_post_clicks=("post_clicks", "median"),
        mean_pre_clicks=("pre_clicks", "mean"), submitted_by_cutoff=("submitted_by_cutoff", "mean"))
    profile = profile.join(sample.groupby("cluster")[cluster_features + ["post_resource_type_count"]].mean()).reset_index()
    context = (sample.groupby(["cluster", "code_module", "code_presentation"]).size()
               .rename("n").reset_index())
    differences = np.column_stack([(labels[:n] == k).astype(float) - (labels[n:] == k).astype(float)
                                   for k in range(best["k"])])
    interval = _component_intervals(chosen, differences, seed, repeats)
    contrast = pd.DataFrame([{"cluster": k, "sample_pairs": n,
                              "fail_minus_pass_membership": float(differences[:, k].mean()),
                              "ci_low": float(interval[0, k]), "ci_high": float(interval[1, k])}
                             for k in range(best["k"])])
    daily = []
    for k in range(best["k"]):
        rows = post[indices[labels == k]]
        for day in range(14):
            daily.append({"cluster": k, "post_day": day + 1, "n": len(rows),
                          "mean_clicks": float(rows[:, day].mean()),
                          "active_fraction": float((rows[:, day] > 0).mean())})
    return pd.DataFrame(candidates), profile, context, contrast, pd.DataFrame(daily)


def run(data_dir: Path, node_dir: Path, output_dir: Path, seed: int = 42,
        repeats: int = 1000, bootstrap: int = 500, cluster_pairs: int = 800) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    protocol = json.loads((node_dir / "node_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("post_days") != 14:
        raise ValueError("Expected a two-week node run")
    for name, source_hash in protocol["sources_sha256"].items():
        if _sha256(data_dir / f"{name}.csv") != source_hash:
            raise ValueError(f"Node source differs: {name}")
    node = pd.read_parquet(node_dir / "node_features.parquet")
    frame, post, quality = build_behavior_features(node, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "features.parquet", index=False)
    balance, effects, slices, pre_sensitivity, post_sensitivity, daily, pairs = compare_pairs(
        frame, post, output_dir, seed, repeats)
    model, intervals = evaluate_models(frame, seed, bootstrap)
    candidates, profiles, context, contrast, cluster_daily = explore_clusters(
        frame, post, pairs, output_dir, seed, repeats, cluster_pairs)
    for name, table in {"balance": balance, "effects": effects, "presentations": slices,
                        "pre_volume_sensitivity": pre_sensitivity, "daily_profile": daily,
                        "positive_post_sensitivity": post_sensitivity,
                        "models": model, "ap_intervals": intervals,
                        "cluster_candidates": candidates, "cluster_profile": profiles,
                        "cluster_context": context, "cluster_pair_contrasts": contrast,
                        "cluster_daily_profile": cluster_daily}.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    record = {"question": "post-TMA behavior at comparable post-deadline clicks",
              "window": {"pre": [-13, 0], "post": [1, 14]}, "seed": seed,
              "pair_bootstrap": repeats, "model_bootstrap": bootstrap,
              "cluster_sample_max_pairs": cluster_pairs, "quality": quality,
              "node_protocol_sha256": _sha256(node_dir / "node_protocol.json"),
              "node_features_sha256": _sha256(node_dir / "node_features.parquet"),
              "source_sha256": {"studentVle": _sha256(data_dir / "studentVle.csv"),
                                "vle": _sha256(data_dir / "vle.csv")},
              "code_sha256": _sha256(Path(__file__)),
              "limitations": "retrospective survivors; matched outcomes are artificial; cluster is exploratory; repeated test use; no causal effect"}
    (output_dir / "protocol.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--cluster-pairs", type=int, default=800)
    args = parser.parse_args()
    run(args.data_dir, args.node_dir, args.output_dir, args.seed,
        args.repeats, args.bootstrap, args.cluster_pairs)


if __name__ == "__main__":
    main()
