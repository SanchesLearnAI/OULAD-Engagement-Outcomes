"""CLI: python -m oulad_analysis --data-dir ... --output-dir ..."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .cluster import cluster_behaviors
from .compare import compare_similar_clicks
from .features import build_features
from .model import evaluate_models
from .timing import describe_timing
from .matched import compare_matched


def main() -> None:
    parser = argparse.ArgumentParser(description="First-four-weeks OULAD analysis")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with studentInfo.csv, studentVle.csv and vle.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or existing output directory")
    parser.add_argument("--stage", choices=("all", "features", "matched", "compare", "cluster", "model"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate-events", action="store_true", help="Sensitivity run: drop exact duplicate event rows")
    args = parser.parse_args()
    existing = args.output_dir.exists() and any(args.output_dir.iterdir())
    stage_outputs = {"matched": "matched_balance.csv", "compare": "similar_click_comparison.csv", "cluster": "cluster_choice.json", "model": "model_results.json"}
    if existing and (args.stage in ("all", "features") or not (args.output_dir / "features.parquet").is_file()
                     or (args.output_dir / stage_outputs[args.stage]).exists()):
        parser.error(f"Output directory contains results that this stage would overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.data_dir / f"{name}.csv" for name in ("studentInfo", "studentVle", "vle", "studentRegistration")}
    git_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    manifest = {"stage": args.stage, "seed": args.seed, "deduplicate_events": args.deduplicate_events,
                "python": platform.python_version(), "inputs": {},
                "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "duckdb", "pyarrow")},
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "working_tree_dirty": bool(git_status.stdout.strip())}
    for name, path in inputs.items():
        if path.is_file():
            with path.open("rb") as stream:
                manifest["inputs"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not existing:
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.stage in ("all", "features"):
        frame = build_features(args.data_dir, args.output_dir, args.deduplicate_events)
        describe_timing(frame, args.output_dir)
    else:
        feature_path = args.output_dir / "features.parquet"
        if not feature_path.is_file():
            parser.error(f"Feature file missing: {feature_path}; run --stage features first")
        frame = pd.read_parquet(feature_path)
    if args.stage in ("all", "matched"):
        compare_matched(frame, args.output_dir, args.seed)
    if args.stage in ("all", "compare"):
        compare_similar_clicks(frame, args.output_dir, args.seed)
    if args.stage in ("all", "cluster"):
        cluster_behaviors(frame, args.output_dir, args.seed)
    if args.stage in ("all", "model"):
        evaluate_models(frame, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
