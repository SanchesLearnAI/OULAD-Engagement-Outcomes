"""Mode-aware data access.  After S02, simulation stages read only the sample."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import project_root

KEY = ["code_module", "code_presentation", "id_student"]
NA = ["?"]


def cache_path(name: str) -> Path:
    return project_root() / "data" / "interim" / "cache" / f"{name}.parquet"


def read_raw_cached(name: str, columns=None) -> pd.DataFrame:
    p = cache_path(name)
    if p.exists():
        return pd.read_parquet(p, columns=columns)
    raw = project_root() / "data" / "raw" / "oulad" / f"{name}.csv"
    return pd.read_csv(raw, na_values=NA, usecols=columns)


def master_path(ctx) -> Path:
    return ctx.paths.interim_dir / "master.parquet"


def read_master(ctx) -> pd.DataFrame:
    return pd.read_parquet(master_path(ctx))


def read_vle(ctx) -> pd.DataFrame:
    if ctx.mode == "simulation":
        p = ctx.paths.interim_dir / "studentVle.parquet"
    else:
        p = cache_path("studentVle")
    return pd.read_parquet(p)


def read_student_assessment(ctx) -> pd.DataFrame:
    if ctx.mode == "simulation":
        p = ctx.paths.interim_dir / "studentAssessment.parquet"
    else:
        p = cache_path("studentAssessment_enriched")
    return pd.read_parquet(p)


def read_assessments() -> pd.DataFrame:
    return read_raw_cached("assessments")


def read_vle_catalog() -> pd.DataFrame:
    return read_raw_cached("vle")


def read_features(ctx, cutoff: int) -> pd.DataFrame:
    p = ctx.paths.processed_dir / f"features_c{cutoff}.parquet"
    return pd.read_parquet(p)


def read_splits(ctx) -> dict:
    import json
    p = ctx.paths.processed_dir / "splits.json"
    return json.loads(p.read_text())
