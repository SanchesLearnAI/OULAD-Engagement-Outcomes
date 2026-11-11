"""Build the compact behavior-evidence figure from committed aggregate tables.

This combines four matching specifications for display, without treating their
pairs as one sample or their intervals as a joint test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "main": ROOT / "oulad-sem/outputs/tables/t06_primary_holm.csv",
    "pre": ROOT / "oulad-sem/outputs/tables/t06_pre_volume_sensitivity.csv",
    "score": ROOT / "results/tma_score_gap_20261103/joint_effects.csv",
}
FEATURES = (
    ("post_active_days", "Active days", "days", 1),
    ("post_longest_inactive_run", "Longest inactive run", "days", 1),
    ("post_week2_share", "Second-week click share", "percentage points", 100),
)
SPECIFICATIONS = (
    ("Post clicks", "main", None, 4768),
    ("Post + pre clicks", "pre", .2, 664),
    ("Post clicks + score*", "score", "positive_post", 2770),
    ("Post + pre + score*", "score", "positive_post_pre20", 410),
)


def select_row(table: pd.DataFrame, source: str, feature: str, condition) -> pd.Series:
    if source == "pre":
        rows = table[table.feature.eq(feature) & table.pre_click_tolerance.eq(condition)]
    elif source == "score":
        rows = table[table.feature.eq(feature) & table.subset.eq(condition)]
    else:
        rows = table[table.metric.eq(feature)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {source} row for {feature}, found {len(rows)}")
    return rows.iloc[0]


def build(output_dir: Path) -> None:
    tables = {name: pd.read_csv(path) for name, path in SOURCES.items()}
    rows = []
    for feature, label, unit, scale in FEATURES:
        for spec, source, condition, expected_pairs in SPECIFICATIONS:
            row = select_row(tables[source], source, feature, condition)
            n = int(row.pairs)
            if n != expected_pairs:
                raise ValueError(f"{spec}: expected {expected_pairs} pairs, found {n}")
            if not row.ci_low <= row.fail_minus_pass <= row.ci_high:
                raise ValueError(f"{spec}: invalid interval for {feature}")
            rows.append({
                "feature": feature, "specification": spec, "pairs": n,
                "estimate": float(row.fail_minus_pass) * scale,
                "ci_low": float(row.ci_low) * scale,
                "ci_high": float(row.ci_high) * scale,
                "unit": unit, "source": str(SOURCES[source].relative_to(ROOT)).replace("\\", "/"),
                "holm_p_main": float(row.p_holm) if source == "main" else np.nan,
            })
    evidence = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output_dir / "paired_persistence_evidence.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    positions = np.arange(len(SPECIFICATIONS))[::-1]
    for ax, (feature, label, unit, _) in zip(axes, FEATURES):
        selected = evidence[evidence.feature.eq(feature)]
        for position, row in zip(positions, selected.itertuples()):
            marker_style = {"marker": "o", "markersize": 7,
                            "mec": "#263747", "mew": 1.1,
                            "mfc": "#315f84" if position == positions[0] else "white"}
            ax.errorbar(row.estimate, position,
                        xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                        fmt="none", ecolor="#315f84", elinewidth=1.8, capsize=3)
            ax.plot(row.estimate, position, linestyle="none", **marker_style)
        ax.axvline(0, color="#555555", linewidth=1.1)
        ax.set_title(label)
        ax.set_xlabel(f"Fail − Pass ({unit})")
        ax.set_yticks(positions)
        if ax is axes[0]:
            ax.set_yticklabels([f"{r.specification}\n(n={r.pairs:,})"
                                for r in selected.itertuples()])
        else:
            ax.tick_params(labelleft=False)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        values = np.r_[selected.ci_low.to_numpy(), selected.ci_high.to_numpy(), 0]
        span = values.max() - values.min()
        ax.set_xlim(values.min() - .12 * span, values.max() + .12 * span)
    fig.suptitle("Similar post-TMA click volume, different activity timing", fontsize=14)
    fig.text(.5, -.015,
             "All post clicks within 5%; pre clicks within 20%; score within 5 points. * Re-matched positive-click pairs. 95% student-component bootstrap intervals.",
             ha="center", fontsize=9, color="#4c5661")
    fig.savefig(output_dir / "paired_persistence_evidence.png", dpi=180,
                bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir)
