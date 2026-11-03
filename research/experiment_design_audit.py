"""Reproduce aggregate feasibility facts for the experiment design (no student export)."""
from pathlib import Path
import json
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
OUT = ROOT / "research" / "experiment_design_20261103"


def main():
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(REPO / "data" / "oulad.zip") as archive:
        info = pd.read_csv(archive.open("studentInfo.csv"))
        vle = pd.read_csv(archive.open("vle.csv"))
        courses = pd.read_csv(archive.open("courses.csv"))
    feature_path = REPO / "outputs" / "timing_concentration_20261102" / "features.parquet"
    features = pd.read_parquet(feature_path)
    keys = ["code_module", "code_presentation", "id_student"]
    assert not info.duplicated(keys).any()
    assert not features.duplicated(keys).any()
    assert len(info.merge(features[keys], on=keys, validate="one_to_one")) == len(info)
    module_counts = info.groupby("id_student").code_module.nunique().value_counts().sort_index()
    assert module_counts.sum() == info.id_student.nunique()
    summary = {
        "enrollment_records": len(info),
        "unique_students": info.id_student.nunique(),
        "distinct_modules_per_student": {str(k): int(v) for k, v in module_counts.items()},
        "catalog_resource_types": sorted(vle.activity_type.unique().tolist()),
        "observed_day0_27_resource_types": [c.removeprefix("resource_share__") for c in features if c.startswith("resource_share__")],
        "course_duration_days": [int(courses.module_presentation_length.min()), int(courses.module_presentation_length.max())],
        "feature_input": str(feature_path),
        "scope": "Catalog is full-course metadata, not a verified historical release calendar. Click summaries include all final outcomes and zero-click enrollments.",
    }
    (OUT / "data_facts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    features.groupby(["code_module", "code_presentation"]).total_clicks.agg(["size", "mean", "median"]).to_csv(OUT / "course_click_context.csv")
    pd.crosstab([vle.code_module, vle.code_presentation], vle.activity_type).to_csv(OUT / "resource_catalog_counts.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
