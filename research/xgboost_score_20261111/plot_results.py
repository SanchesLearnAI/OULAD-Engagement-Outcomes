"""Render the five-split AP comparison from aggregate metrics.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(metrics_path: Path, output: Path) -> None:
    metrics = pd.read_csv(metrics_path).sort_values("seed")
    positions = range(len(metrics))
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    ax.plot(positions, metrics.baseline_ap, "o-", linewidth=2.2,
            color="#555555", label=f"Score logistic regression (mean {metrics.baseline_ap.mean():.4f})")
    ax.plot(positions, metrics.xgboost_ap, "o-", linewidth=2.2,
            color="#6756aa", label=f"GPU XGBoost (mean {metrics.xgboost_ap.mean():.4f})")
    ax.set_xticks(list(positions), metrics.seed.astype(str))
    ax.set_ylim(.695, .735)
    ax.set_xlabel("Student-group split seed")
    ax.set_ylabel("Fail average precision (AP)")
    ax.set_title("Score-inclusive logistic baseline vs GPU XGBoost")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.metrics, args.output)
