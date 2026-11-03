"""Retrospective first-weighted-TMA score control for matched +21/+28 behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assessment_node import BATCH, KEY, _first_weighted_tma
from .matched import _component_intervals

BEHAVIOR = ["pre_active_days", "post_active_days", "last3_share",
            "post_active_week_fraction", "post_minus_pre_log", "submitted_after_due",
            "post_resource_type_count", "post_resource_entropy",
            "post_resource_share__forumng", "post_resource_share__quiz"]


def attach_scores(node: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Rank final recorded scores within eligible, submitted cohort per assessment.

    Date restriction concerns submission, not grade release (unrecorded in OULAD).
    Include eventual Withdrawn records in the rank reference population.
    """
    selected = _first_weighted_tma(pd.read_csv(data_dir / "assessments.csv", na_values=["?"]))
    grades = pd.read_csv(data_dir / "studentAssessment.csv", na_values=["?"])
    if grades.duplicated(["id_assessment", "id_student"]).any() or node.duplicated(KEY).any():
        raise ValueError("Duplicate grade or node key")
    joined = node[KEY + ["final_result", "cutoff_day"]].merge(
        selected[BATCH + ["id_assessment"]], on=BATCH, validate="many_to_one")
    if len(joined) != len(node):
        raise ValueError("Missing weighted TMA in node cohort")
    joined = joined.merge(grades[["id_assessment", "id_student", "date_submitted", "is_banked", "score"]],
                          on=["id_assessment", "id_student"], how="left", validate="one_to_one")
    if joined.is_banked.eq(1).any():
        raise ValueError("Banked first TMA in node cohort")
    if joined.score.dropna().lt(0).any() or joined.score.dropna().gt(100).any():
        raise ValueError("First TMA score outside 0–100")
    joined["first_tma_submitted_by_cutoff"] = joined.date_submitted.le(joined.cutoff_day)
    joined["score_recorded_for_submission"] = joined.first_tma_submitted_by_cutoff & joined.score.notna()
    joined["first_tma_score"] = joined.score.where(joined.score_recorded_for_submission)
    joined["first_tma_rank_pct"] = joined.groupby("id_assessment").first_tma_score.rank(
        method="average", pct=True)
    # Keep score availability separate from the question of when marks were released.
    return joined[KEY + ["first_tma_score", "first_tma_rank_pct",
                         "first_tma_submitted_by_cutoff", "score_recorded_for_submission"]]


def make_joint_pairs(frame: pd.DataFrame, score_gap: float = 5,
                     click_tolerance: float = .05, seed: int = 42) -> pd.DataFrame:
    """Exact clicks first, then <=5% clicks; all pairs within <=5 score points."""
    if frame.duplicated(KEY).any() or frame.first_tma_score.isna().any():
        raise ValueError("Joint matching requires unique scored binary records")
    rng = np.random.default_rng(seed)
    records = []
    for (module, presentation), group in frame.groupby(BATCH):
        failures = rng.permutation(group.index[group.final_result.eq("Fail")]).tolist()
        passes = rng.permutation(group.index[~group.final_result.eq("Fail")]).tolist()
        pass_clicks = frame.loc[passes, "total_clicks"].to_numpy(dtype=float)
        pass_scores = frame.loc[passes, "first_tma_score"].to_numpy(dtype=float)
        available = np.ones(len(passes), dtype=bool)
        remaining = []
        for exact in (True, False):
            if not exact:
                failures = remaining
            for i in failures:
                clicks = float(frame.at[i, "total_clicks"])
                score = float(frame.at[i, "first_tma_score"])
                score_distance = np.abs(pass_scores - score)
                candidate = available & (score_distance <= score_gap + 1e-12)
                if exact:
                    candidate &= pass_clicks == clicks
                else:
                    smallest = np.minimum(pass_clicks, clicks)
                    largest = np.maximum(pass_clicks, clicks)
                    candidate &= ((smallest == 0) & (largest == 0)) | (
                        (smallest > 0) & (largest / np.maximum(smallest, 1) <= 1 + click_tolerance + 1e-12))
                options = np.flatnonzero(candidate)
                if not len(options):
                    if exact:
                        remaining.append(i)
                    continue
                distances = np.abs(np.log1p(pass_clicks[options]) - np.log1p(clicks))
                j = int(options[np.lexsort((score_distance[options], distances))[0]])
                available[j] = False
                k = passes[j]
                other = pass_clicks[j]
                records.append({"code_module": module, "code_presentation": presentation,
                                "fail_index": i, "pass_index": k,
                                "fail_student": int(frame.at[i, "id_student"]),
                                "pass_student": int(frame.at[k, "id_student"]),
                                "relative_click_gap": max(clicks, other) / min(clicks, other) - 1
                                if min(clicks, other) > 0 else 0.0,
                                "score_gap": score - pass_scores[j]})
    return pd.DataFrame(records, columns=["code_module", "code_presentation", "fail_index",
                                          "pass_index", "fail_student", "pass_student",
                                          "relative_click_gap", "score_gap"])


def paired_effects(frame: pd.DataFrame, pairs: pd.DataFrame, features: list[str],
                   post_days: int, comparison: str, repeats: int = 1000,
                   seed: int = 42) -> pd.DataFrame:
    if pairs.empty:
        raise ValueError("No pairs for score comparison")
    fail = frame.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = frame.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    difference = fail[features].to_numpy(dtype=float) - passing[features].to_numpy(dtype=float)
    interval = _component_intervals(pairs, difference, seed, repeats)
    return pd.DataFrame([{"post_days": post_days, "comparison": comparison,
                          "pairs": len(pairs), "feature": feature,
                          "fail_mean": float(fail[feature].mean()),
                          "pass_mean": float(passing[feature].mean()),
                          "fail_minus_pass": float(difference[:, j].mean()),
                          "ci_low": float(interval[0, j]),
                          "ci_high": float(interval[1, j])}
                         for j, feature in enumerate(features)])


def run(data_dir: Path, three: Path, four: Path, matched_dir: Path,
        output_dir: Path, seed: int = 42, repeats: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    inputs = {21: three, 28: four}
    grade_hash = hashlib.sha256((data_dir / "studentAssessment.csv").read_bytes()).hexdigest()
    assessment_hash = hashlib.sha256((data_dir / "assessments.csv").read_bytes()).hexdigest()
    for days, path in inputs.items():
        protocol = json.loads((path / "node_protocol.json").read_text(encoding="utf-8"))
        source = protocol.get("sources_sha256", {})
        if source.get("studentAssessment") != grade_hash or source.get("assessments") != assessment_hash:
            raise ValueError(f"{days}-day node cohort uses a different assessment source")
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage, quintiles, effects, balances, presentations = [], [], [], [], []
    for days, path in inputs.items():
        node = pd.read_parquet(path / "node_features.parquet")
        scores = attach_scores(node, data_dir)
        binary = pd.read_parquet(matched_dir / f"features_{days}d.parquet")
        binary = binary.merge(scores, on=KEY, validate="one_to_one")
        original = pd.read_parquet(matched_dir / f"features_{days}d.parquet")
        if not binary[KEY].equals(original[KEY]):
            raise AssertionError("Score join changed pair row order")
        binary.to_parquet(output_dir / f"scored_features_{days}d.parquet", index=False)
        for outcome, group in binary.groupby("final_result"):
            coverage.append({"post_days": days, "outcome": outcome, "rows": len(group),
                             "submitted_by_cutoff": int(group.first_tma_submitted_by_cutoff.sum()),
                             "score_recorded_for_submission": int(group.score_recorded_for_submission.sum()),
                             "score_mean_when_recorded": float(group.first_tma_score.mean())})
        scored = binary[binary.first_tma_score.notna()].copy()
        scored["rank_quintile"] = pd.cut(scored.first_tma_rank_pct,
                                           bins=[0, .2, .4, .6, .8, 1], include_lowest=True,
                                           labels=[1, 2, 3, 4, 5])
        for rank, group in scored.groupby("rank_quintile", observed=True):
            quintiles.append({"post_days": days, "rank_quintile": int(rank),
                              "rows": len(group), "fail_rate": float(group.final_result.eq("Fail").mean()),
                              "score_mean": float(group.first_tma_score.mean())})
        click_pairs = pd.read_parquet(matched_dir / f"pairs_{days}d_05.parquet")
        both = binary.first_tma_score.iloc[click_pairs.fail_index.to_numpy()].notna().to_numpy() & (
            binary.first_tma_score.iloc[click_pairs.pass_index.to_numpy()].notna().to_numpy())
        scored_click_pairs = click_pairs.loc[both].reset_index(drop=True)
        effects.append(paired_effects(binary, scored_click_pairs,
                                      ["first_tma_score", "first_tma_rank_pct"], days,
                                      "click_matched_both_scored", repeats, seed))
        scored = scored.drop(columns="rank_quintile").reset_index(drop=True)
        pairs = make_joint_pairs(scored, seed=seed)
        pairs.to_parquet(output_dir / f"joint_pairs_{days}d.parquet", index=False)
        if pairs.empty:
            raise ValueError("No joint click-and-score pairs")
        matched_fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
        matched_pass = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
        x = matched_fail.post_clicks.to_numpy()
        y = matched_pass.post_clicks.to_numpy()
        post_ratio = np.maximum(x, y) / np.maximum(np.minimum(x, y), 1)
        post_similar = (np.minimum(x, y) > 0) & (post_ratio <= 1.20 + 1e-12)
        post_pairs = pairs.loc[post_similar].reset_index(drop=True)
        balances.append({"post_days": days, "scored_binary_rows": len(scored),
                         "scored_fail_rows": int(scored.final_result.eq("Fail").sum()),
                         "click_matched_both_scored_pairs": len(scored_click_pairs),
                         "joint_pairs": len(pairs),
                         "joint_post20_pairs": len(post_pairs),
                         "joint_fail_coverage": len(pairs) / scored.final_result.eq("Fail").sum(),
                         "mean_abs_score_gap": float(pairs.score_gap.abs().mean()),
                         "max_abs_score_gap": float(pairs.score_gap.abs().max()),
                         "mean_relative_click_gap": float(pairs.relative_click_gap.mean()),
                         "max_relative_click_gap": float(pairs.relative_click_gap.max())})
        effects.append(paired_effects(scored, pairs,
                                      ["first_tma_score", "first_tma_rank_pct"] + BEHAVIOR,
                                      days, "joint_click_score_matched", repeats, seed))
        if not post_pairs.empty:
            effects.append(paired_effects(scored, post_pairs, BEHAVIOR,
                                          days, "joint_click_score_post20_subset", repeats, seed))
        fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
        passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
        for (module, presentation), ids in pairs.groupby(BATCH).indices.items():
            for name in BEHAVIOR:
                presentations.append({"post_days": days, "code_module": module,
                                      "code_presentation": presentation, "pairs": len(ids),
                                      "feature": name,
                                      "fail_minus_pass": float((fail.iloc[ids][name].to_numpy() -
                                                                passing.iloc[ids][name].to_numpy()).mean())})
    pd.DataFrame(coverage).to_csv(output_dir / "score_coverage.csv", index=False)
    pd.DataFrame(quintiles).to_csv(output_dir / "rank_quintiles.csv", index=False)
    pd.DataFrame(balances).to_csv(output_dir / "joint_balance.csv", index=False)
    pd.concat(effects, ignore_index=True).to_csv(output_dir / "score_effects.csv", index=False)
    pd.DataFrame(presentations).to_csv(output_dir / "joint_presentations.csv", index=False)
    (output_dir / "protocol.json").write_text(json.dumps({
        "question": "Does TMA-relative behavior differ at comparable clicks and first-TMA score?",
        "post_days": [21, 28], "score_definition": "Final recorded score for first positive-weight TMA, only if submitted by cutoff; actual grade release date unknown",
        "rank_definition": "Average-tie percentile within assessment among all cutoff-eligible records with recorded score, including eventual Withdrawn",
        "joint_matching": "same module presentation; score difference <=5 points; exact clicks first, then <=5% ratio; no reuse",
        "post_volume_sensitivity": "Exploratory subset of joint pairs with positive post clicks within 20% ratio",
        "seed": seed, "component_bootstrap": repeats,
        "source_sha256": {"studentAssessment": grade_hash,
                          "assessments": assessment_hash},
        "limitations": "Retrospective score control; grade release timing unavailable; first weighted TMA contributes to final outcome; exploratory fixed-pair intervals, no causal interpretation"
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--three", type=Path, required=True)
    parser.add_argument("--four", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.data_dir, args.three, args.four, args.matched_dir,
        args.output_dir, args.seed, args.repeats)


if __name__ == "__main__":
    main()
