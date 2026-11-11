"""Render the five-split AP comparison from aggregate metrics.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(metrics_path: Path, output: Path) -> None:
    metrics = pd.read_csv(metrics_path).sort_values("seed")
    positions = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    ax.bar(positions, metrics.delta_ap, width=.6, color="#6756aa")
    mean_delta = metrics.delta_ap.mean()
    ax.axhline(mean_delta, color="#555555", linewidth=1.3, linestyle="--",
               label=f"Five-split mean +{mean_delta:.4f}")
    for position, delta in zip(positions, metrics.delta_ap):
        ax.annotate(f"{delta:+.4f}", (position, delta), xytext=(0, 4 if delta >= 0 else -4),
                    textcoords="offset points", ha="center",
                    va="bottom" if delta >= 0 else "top", fontsize=9)
    ax.set_xticks(positions, metrics.seed.astype(str))
    ax.set_ylim(min(0, metrics.delta_ap.min() * 1.25),
                max(0, metrics.delta_ap.max() * 1.25))
    ax.set_xlabel("Student-group split seed")
    ax.set_ylabel("Fail AP difference (XGBoost − logistic)")
    ax.set_title("XGBoost gains were small and positive on all five splits")
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.metrics, args.output)
