"""Test whether a signed pre/post TMA click gap improves the two-week models.

The gap is an engineered transformation of click counts already present in the
baseline. It supplies a nonlinear representation, not new observations.
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
from .tma_two_week import SUBMIT, TIMING

BASELINE = ["module_presentation", "log_pre_clicks", "pre_active_days", "last3_share",
            "log_post_clicks", "log_post_clicks_squared"] + SUBMIT
VERSIONS = {"M0_pre_post_volume_submission": BASELINE,
            "M0_plus_relative_gap": BASELINE + ["relative_click_gap"],
            "M1_add_post_timing": BASELINE + TIMING,
            "M1_timing_plus_relative_gap": BASELINE + TIMING + ["relative_click_gap"]}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_click_gap(pre_clicks: np.ndarray, post_clicks: np.ndarray) -> np.ndarray:
    """(post-pre)/(post+pre), in [-1, 1]; return 0 when both windows are zero."""
    pre = np.asarray(pre_clicks, dtype=float)
    post = np.asarray(post_clicks, dtype=float)
    if pre.shape != post.shape or np.any(~np.isfinite(pre)) or np.any(~np.isfinite(post)):
        raise ValueError("Pre/post clicks need identical shapes and finite values")
    if np.any(pre < 0) or np.any(post < 0):
        raise ValueError("Click counts cannot be negative")
    total = pre + post
    return np.divide(post - pre, total, out=np.zeros_like(total), where=total > 0)


def evaluate(frame: pd.DataFrame, seed: int = 42,
             bootstrap: int = 500) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.duplicated(KEY).any() or not frame.final_result.isin(["Fail", "Pass", "Distinction"]).all():
        raise ValueError("Expected unique, binary two-week cohort")
    required = sorted({column for columns in VERSIONS.values() for column in columns} - {"relative_click_gap"})
    if any(column not in frame for column in required + ["pre_clicks", "post_clicks"]):
        raise ValueError("Missing two-week model features")
    frame = frame.copy()
    frame["relative_click_gap"] = relative_click_gap(frame.pre_clicks, frame.post_clicks)
    frame["both_zero_clicks"] = frame.pre_clicks.eq(0) & frame.post_clicks.eq(0)
    rows, contrasts = [], []
    for split_seed in (seed, seed + 101, seed + 102, seed + 103, seed + 104):
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(
            frame, groups=frame.id_student))
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        if not set(train.id_student).isdisjoint(test.id_student):
            raise AssertionError("Student overlap across split")
        y = test.final_result.eq("Fail").astype(int).to_numpy()
        predictions = {name: _fit_predict(train, test, columns) for name, columns in VERSIONS.items()}
        for name, scores in predictions.items():
            rows.append({"seed": split_seed, "model": name, "train_rows": len(train),
                         "test_rows": len(test), "test_students": test.id_student.nunique(),
                         "fail_prevalence": float(y.mean()),
                         "average_precision": float(average_precision_score(y, scores)),
                         "brier": float(brier_score_loss(y, scores))})
        if split_seed != seed:
            continue
        members = list(test.groupby("id_student").indices.values())
        rng = np.random.default_rng(seed + 7300)
        draws = [np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
                 for _ in range(bootstrap)]
        for model, comparator in (("M0_plus_relative_gap", "M0_pre_post_volume_submission"),
                                  ("M1_timing_plus_relative_gap", "M1_add_post_timing")):
            p, q = predictions[model], predictions[comparator]
            ap_deltas, brier_deltas = [], []
            for indices in draws:
                if len(np.unique(y[indices])) < 2:
                    continue
                ap_deltas.append(average_precision_score(y[indices], p[indices]) -
                                 average_precision_score(y[indices], q[indices]))
                brier_deltas.append(brier_score_loss(y[indices], p[indices]) -
                                    brier_score_loss(y[indices], q[indices]))
            contrasts.append({"model": model, "comparator": comparator,
                              "delta_ap": float(average_precision_score(y, p) - average_precision_score(y, q)),
                              "ap_ci_low": float(np.quantile(ap_deltas, .025)),
                              "ap_ci_high": float(np.quantile(ap_deltas, .975)),
                              "delta_brier": float(brier_score_loss(y, p) - brier_score_loss(y, q)),
                              "brier_ci_low": float(np.quantile(brier_deltas, .025)),
                              "brier_ci_high": float(np.quantile(brier_deltas, .975))})
    distribution = (frame.assign(outcome=np.where(frame.final_result.eq("Fail"), "Fail", "Pass/Distinction"))
                    .groupby("outcome").agg(records=("id_student", "size"),
                                             mean_gap=("relative_click_gap", "mean"),
                                             median_gap=("relative_click_gap", "median"),
                                             both_zero=("both_zero_clicks", "sum"))
                    .reset_index())
    return pd.DataFrame(rows), pd.DataFrame(contrasts), distribution


def plot_ap_deltas(models: pd.DataFrame, output_path: Path) -> None:
    """Display same-split AP changes from adding the engineered gap."""
    import matplotlib.pyplot as plt

    pivot = models.pivot(index="seed", columns="model", values="average_precision")
    comparisons = [("M0 + gap", "M0_plus_relative_gap", "M0_pre_post_volume_submission", "#3856B3"),
                   ("M1 + gap", "M1_timing_plus_relative_gap", "M1_add_post_timing", "#B87916")]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for y, (label, added, baseline, color) in enumerate(comparisons):
        values = (pivot[added] - pivot[baseline]).to_numpy()
        offsets = np.linspace(-.12, .12, len(values))
        ax.scatter(values, y + offsets, s=55, color=color, alpha=.75, zorder=3)
        ax.scatter([values.mean()], [y], marker="D", s=80, facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=4)
    ax.axvline(0, color="#5C6572", linewidth=1)
    ax.set_yticks([0, 1], ["M0 + relative gap", "M1 + relative gap"])
    ax.set_ylim(1.45, -.45)
    ax.set_xlabel("Change in Fail average precision (with gap − without gap)")
    ax.set_title("Relative pre/post click gap: five student-grouped holdouts", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E8EE")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.13, .025, "Dots: individual splits; open diamond: mean. Splits overlap and are not independent replications.",
             fontsize=9, color="#4A5260")
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def run(input_dir: Path, output_dir: Path, reference_models: Path | None = None,
        seed: int = 42, bootstrap: int = 500) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_dir}")
    source = input_dir / "features.parquet"
    frame = pd.read_parquet(source)
    models, contrasts, distribution = evaluate(frame, seed, bootstrap)
    if reference_models is not None:
        original = pd.read_csv(reference_models)
        old = original[original.model.isin(["M0_pre_post_volume_submission", "M1_add_post_timing"])]
        new = models[models.model.isin(old.model)]
        merged = new.merge(old, on=["seed", "model"], suffixes=("_new", "_old"), validate="one_to_one")
        if len(merged) != 10 or not np.allclose(merged.average_precision_new, merged.average_precision_old,
                                               atol=1e-12, rtol=0) or not np.allclose(
                                                   merged.brier_new, merged.brier_old, atol=1e-12, rtol=0):
            raise ValueError("Baseline predictions do not reconcile to original two-week models")
    output_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(output_dir / "relative_gap_models.csv", index=False)
    contrasts.to_csv(output_dir / "relative_gap_intervals.csv", index=False)
    distribution.to_csv(output_dir / "relative_gap_distribution.csv", index=False)
    plot_ap_deltas(models, output_dir / "relative_gap_ap.png")
    protocol = {"question": "incremental predictive value of relative pre/post TMA click gap",
                "feature": "(post_clicks - pre_clicks) / (post_clicks + pre_clicks); both zero -> 0",
                "windows": {"pre": [-13, 0], "post": [1, 14]},
                "models": VERSIONS, "seed": seed, "student_bootstrap": bootstrap,
                "input_sha256": {"features.parquet": _sha256(source),
                                 "two_week_protocol.json": _sha256(input_dir / "protocol.json")},
                "reference_models_sha256": _sha256(reference_models) if reference_models else None,
                "code_sha256": _sha256(Path(__file__)),
                "limitations": "post hoc feature engineering on repeatedly explored OULAD data; no new information beyond counts; no causal interpretation"}
    (output_dir / "relative_gap_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-models", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.reference_models, args.seed, args.bootstrap)


if __name__ == "__main__":
    main()
