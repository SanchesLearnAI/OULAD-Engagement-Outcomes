"""Retrospective first-TMA mark and pre/post click-gap interaction study.

The dataset has submission dates but no mark-release dates. Scores here are
final recorded marks for submissions made by the +14-day cutoff, not verified
as information available to a live predictor at that time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .assessment_node import KEY, _fit_predict
from .matched import _component_intervals
from .tma_relative_gap_model import VERSIONS as GAP_VERSIONS, relative_click_gap
from .tma_score_probe import attach_scores, make_joint_pairs

BASE = GAP_VERSIONS["M1_add_post_timing"]
SCORE = ["first_tma_score", "score_recorded_for_submission"]
VERSIONS = {"M1_timing": BASE,
            "M1_timing_gap": BASE + ["relative_click_gap"],
            "S0_score": BASE + SCORE,
            "S1_score_gap": BASE + SCORE + ["relative_click_gap"],
            "S2_score_gap_interaction": BASE + SCORE + ["relative_click_gap", "score_x_gap"]}
PAIR_FEATURES = ["first_tma_score", "relative_click_gap", "pre_clicks", "post_clicks",
                 "post_active_days", "post_both_weeks", "post_longest_inactive_run", "post_week2_share"]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_score_frame(features: pd.DataFrame, node: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if features.duplicated(KEY).any() or not features.final_result.isin(["Fail", "Pass", "Distinction"]).all():
        raise ValueError("Expected unique binary two-week features")
    if not features.cutoff_day.sub(features.due_day).eq(14).all():
        raise ValueError("Expected TMA +14-day cutoff")
    scores = attach_scores(node, data_dir)
    frame = features.merge(scores, on=KEY, validate="one_to_one", sort=False)
    if len(frame) != len(features) or not frame[KEY].equals(features[KEY]):
        raise ValueError("Score join changed two-week cohort or row order")
    if not frame.first_tma_score.dropna().between(0, 100).all():
        raise ValueError("First TMA mark outside 0–100")
    frame["relative_click_gap"] = relative_click_gap(frame.pre_clicks, frame.post_clicks)
    frame["score_x_gap"] = frame.first_tma_score.div(100) * frame.relative_click_gap
    return frame


def evaluate_models(frame: pd.DataFrame, seed: int, bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, contrasts = [], []
    comparisons = [("S0_score", "M1_timing"),
                   ("S1_score_gap", "S0_score"),
                   ("S2_score_gap_interaction", "S1_score_gap"),
                   ("S2_score_gap_interaction", "S0_score")]
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[left], frame.iloc[right]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across train and test")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, columns) for name, columns in VERSIONS.items()}
        for name, scores in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train),
                         "test_rows": len(test), "test_students": test.id_student.nunique(),
                         "test_score_coverage": float(test.first_tma_score.notna().mean()),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, scores)),
                         "brier": float(brier_score_loss(y, scores))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 11700)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for model, comparator in comparisons:
            p, q = predictions[model], predictions[comparator]
            deltas = [average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], q[ix])
                      for ix in draws if len(np.unique(y[ix])) == 2]
            contrasts.append({"model": model, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ci_low": float(np.quantile(deltas, .025)),
                              "ci_high": float(np.quantile(deltas, .975))})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def describe_score_gap(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = (frame.groupby("final_result").agg(
        records=("id_student", "size"),
        submitted_by_cutoff=("first_tma_submitted_by_cutoff", "sum"),
        score_recorded=("score_recorded_for_submission", "sum"),
        mean_recorded_score=("first_tma_score", "mean")).reset_index())
    scored = frame[frame.first_tma_score.notna()].copy()
    scored["score_band"] = pd.cut(scored.first_tma_rank_pct, [0, .3, .7, 1],
                                  labels=["low", "middle", "high"], include_lowest=True)
    both_zero = scored.pre_clicks.eq(0) & scored.post_clicks.eq(0)
    scored["gap_band"] = np.select(
        [both_zero, scored.relative_click_gap.le(-.5), scored.relative_click_gap.le(0)],
        ["both_zero", "post_much_lower", "post_lower_or_equal"], default="post_higher")
    strata = (scored.groupby(["score_band", "gap_band"], observed=True).agg(
        records=("id_student", "size"),
        fail_records=("final_result", lambda value: int(value.eq("Fail").sum())),
        mean_first_tma_score=("first_tma_score", "mean"),
        mean_pre_clicks=("pre_clicks", "mean"),
        mean_post_clicks=("post_clicks", "mean"),
        mean_relative_gap=("relative_click_gap", "mean")).reset_index())
    strata["fail_fraction"] = strata.fail_records / strata.records
    return coverage, strata[strata.records.ge(30)].reset_index(drop=True)


def compare_joint_pairs(frame: pd.DataFrame, output_dir: Path,
                        seed: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame[frame.first_tma_score.notna()].reset_index(drop=True).copy()
    scored["total_clicks"] = scored.post_clicks
    pairs = make_joint_pairs(scored, score_gap=5, click_tolerance=.05, seed=seed)
    if pairs.empty:
        raise ValueError("No joint post-click and score pairs")
    pairs.to_parquet(output_dir / "joint_score_post_click_pairs.parquet", index=False)
    fail = scored.iloc[pairs.fail_index.to_numpy()].reset_index(drop=True)
    passing = scored.iloc[pairs.pass_index.to_numpy()].reset_index(drop=True)
    mean_rank = (fail.first_tma_rank_pct + passing.first_tma_rank_pct) / 2
    bands = pd.cut(mean_rank, [0, .3, .7, 1], labels=["low", "middle", "high"], include_lowest=True)
    positive = fail.post_clicks.gt(0).to_numpy()
    pre_fail, pre_pass = fail.pre_clicks.to_numpy(), passing.pre_clicks.to_numpy()
    smaller, larger = np.minimum(pre_fail, pre_pass), np.maximum(pre_fail, pre_pass)
    pre_close = ((smaller == 0) & (larger == 0)) | ((smaller > 0) & (larger / np.maximum(smaller, 1) <= 1.2 + 1e-12))
    subsets = [("all_scored", np.ones(len(pairs), dtype=bool)), ("positive_post", positive),
               ("positive_post_pre20", positive & pre_close)]
    subsets += [(f"positive_post_{band}_mark", positive & bands.eq(band).to_numpy())
                for band in ("low", "middle", "high")]
    balances, effects = [], []
    for name, selected in subsets:
        subset = pairs.loc[selected].reset_index(drop=True)
        if len(subset) < 30:
            continue
        a, b = fail.loc[selected].reset_index(drop=True), passing.loc[selected].reset_index(drop=True)
        differences = a[PAIR_FEATURES].to_numpy(dtype=float) - b[PAIR_FEATURES].to_numpy(dtype=float)
        ci = _component_intervals(subset, differences, seed, repeats)
        balances.append({"subset": name, "pairs": len(subset),
                         "mean_abs_score_gap": float(subset.score_gap.abs().mean()),
                         "mean_relative_post_click_gap": float(subset.relative_click_gap.mean()),
                         "fail_score_mean": float(a.first_tma_score.mean()),
                         "pass_score_mean": float(b.first_tma_score.mean()),
                         "fail_post_click_mean": float(a.post_clicks.mean()),
                         "pass_post_click_mean": float(b.post_clicks.mean()),
                         "fail_pre_click_mean": float(a.pre_clicks.mean()),
                         "pass_pre_click_mean": float(b.pre_clicks.mean())})
        for j, feature in enumerate(PAIR_FEATURES):
            effects.append({"subset": name, "feature": feature, "pairs": len(subset),
                            "fail_mean": float(a[feature].mean()), "pass_mean": float(b[feature].mean()),
                            "fail_minus_pass": float(differences[:, j].mean()),
                            "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j])})
    return pd.DataFrame(balances), pd.DataFrame(effects)


def plot_model_ap(models: pd.DataFrame, output_path: Path) -> None:
    """Show score and interaction ablations on the same five student splits."""
    import matplotlib.pyplot as plt

    order = [("M1: timing", "M1_timing", "#3856B3"),
             ("+ first TMA mark", "S0_score", "#B87916"),
             ("+ relative click gap", "S1_score_gap", "#B87916"),
             ("+ mark × gap", "S2_score_gap_interaction", "#B87916")]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for y, (_, model, color) in enumerate(order):
        values = models.loc[models.model.eq(model)].sort_values("seed").average_precision.to_numpy()
        ax.scatter(values, y + np.linspace(-.11, .11, len(values)), s=42, color=color, alpha=.65)
        ax.scatter([values.mean()], [y], marker="D", s=78, facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=3)
    ax.set_yticks(range(len(order)), [label for label, _, _ in order])
    ax.set_ylim(len(order) - .5, -.5)
    ax.set_xlabel("Fail average precision (focused scale)")
    ax.set_title("First-TMA mark and click-shift interaction: retrospective prediction",
                 loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E8EE")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.2, .025, "Dots: five overlapping student-grouped tests; open diamond: mean. Mark-release dates are unknown.",
             fontsize=9, color="#4A5260")
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def run(data_dir: Path, node_dir: Path, two_week_dir: Path, output_dir: Path,
        reference_models: Path | None = None, seed: int = 42,
        bootstrap: int = 500, repeats: int = 1000) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    node_path = node_dir / "node_features.parquet"
    feature_path = two_week_dir / "features.parquet"
    source_protocol = json.loads((two_week_dir / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol["node_features_sha256"] != _sha256(node_path):
        raise ValueError("Two-week features and node records do not share a source")
    node_protocol = json.loads((node_dir / "node_protocol.json").read_text(encoding="utf-8"))
    for name in ("assessments", "studentAssessment"):
        if node_protocol["sources_sha256"][name] != _sha256(data_dir / f"{name}.csv"):
            raise ValueError(f"Assessment source differs: {name}")
    frame = build_score_frame(pd.read_parquet(feature_path), pd.read_parquet(node_path), data_dir)
    models, intervals = evaluate_models(frame, seed, bootstrap)
    if reference_models is not None:
        prior = pd.read_csv(reference_models)
        baseline = prior[prior.model.eq("M1_timing_plus_relative_gap")]
        current = models[models.model.eq("M1_timing_gap")]
        merged = current.merge(baseline, on="seed", suffixes=("_new", "_old"), validate="one_to_one")
        if len(merged) != 5 or not np.allclose(merged.average_precision_new, merged.average_precision_old,
                                              rtol=0, atol=1e-12) or not np.allclose(
                                                  merged.brier_new, merged.brier_old, rtol=0, atol=1e-12):
            raise ValueError("Score study baseline disagrees with relative-gap model")
    coverage, strata = describe_score_gap(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_balance, pair_effects = compare_joint_pairs(frame, output_dir, seed, repeats)
    for name, table in {"models": models, "ap_intervals": intervals,
                        "score_coverage": coverage, "score_gap_strata": strata,
                        "joint_balance": pair_balance, "joint_effects": pair_effects}.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    plot_model_ap(models, output_dir / "score_gap_model_ap.png")
    protocol = {"question": "Does first-TMA mark interact with pre/post click shift in retrospective Fail prediction?",
                "score": "final recorded first weighted TMA mark, only if submitted by +14; release date unavailable",
                "score_groups": "within-assessment percentile among eligible submitted students, including eventual Withdrawn; low <=30%, middle 30-70%, high >70%; descriptive only; public strata with <30 records suppressed",
                "relative_gap": "(post_clicks-pre_clicks)/(post_clicks+pre_clicks); both zero -> 0",
                "interaction": "first_tma_score/100 * relative_gap; missing when mark unavailable",
                "models": VERSIONS, "seed": seed, "student_bootstrap": bootstrap,
                "pair_bootstrap": repeats, "pairs": "same course/presentation; first-TMA score within 5 points; post clicks exact then within 5%; no reuse",
                "input_sha256": {"node_features.parquet": _sha256(node_path),
                                 "features.parquet": _sha256(feature_path),
                                 "assessments.csv": _sha256(data_dir / "assessments.csv"),
                                 "studentAssessment.csv": _sha256(data_dir / "studentAssessment.csv")},
                "reference_models_sha256": _sha256(reference_models) if reference_models else None,
                "code_sha256": _sha256(Path(__file__)),
                "limitations": "post hoc, grade release unknown, score partly determines final result, selected scored cohort in matching, repeated test use, no causal effect"}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--node-dir", type=Path, required=True)
    parser.add_argument("--two-week-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-models", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.data_dir, args.node_dir, args.two_week_dir, args.output_dir,
        args.reference_models, args.seed, args.bootstrap, args.repeats)


if __name__ == "__main__":
    main()
