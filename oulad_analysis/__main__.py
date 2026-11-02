"""CLI: python -m oulad_analysis --data-dir ... --output-dir ..."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cluster import cluster_behaviors
from .compare import compare_similar_clicks
from .features import build_features
from .model import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="First-four-weeks OULAD analysis")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with studentInfo.csv, studentVle.csv and vle.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or existing output directory")
    parser.add_argument("--stage", choices=("all", "features", "compare", "cluster", "model"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "features"):
        frame = build_features(args.data_dir, args.output_dir)
    else:
        feature_path = args.output_dir / "features.parquet"
        if not feature_path.is_file():
            parser.error(f"Feature file missing: {feature_path}; run --stage features first")
        frame = pd.read_parquet(feature_path)
    if args.stage in ("all", "compare"):
        compare_similar_clicks(frame, args.output_dir)
    if args.stage in ("all", "cluster"):
        cluster_behaviors(frame, args.output_dir, args.seed)
    if args.stage in ("all", "model"):
        evaluate_models(frame, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
