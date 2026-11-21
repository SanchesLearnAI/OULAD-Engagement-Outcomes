"""Build the score-free XGBoost input: M2's 16 features, five grouped splits.

This is the deployable counterpart to ``xgboost_score_20261111``: the same tree
model, but only on features observable at ``d+14``, so the comparison against
the frozen M2 is like-for-like and contains no mark information.

The output NPZ holds student-level rows and must stay in a git-ignored
directory. Only aggregate metrics are committed.

Run from the repository root::

    python research/xgboost_noscore_20261112/prepare_input.py --output outputs/xgb_noscore.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from oulad_analysis.assessment_node import _fit_predict  # noqa: E402

# Exactly the frozen M2 ladder: (PRE + POST_VOLUME + SUBMIT + TIMING + DISTANCE).
# Kept as a literal so this script does not import the deliverable package.
FEATURES = [
    "log_pre_clicks", "pre_active_days", "last3_share",
    "log_post_clicks", "log_post_clicks_squared",
    "submitted_by_cutoff", "submitted_after_due",
    "post_active_days", "post_both_weeks", "post_longest_inactive_run",
    "post_week2_share", "post_peak7_share", "post_entropy_norm",
    "post_first_active_day", "days_early", "days_late",
]
SEEDS = (42, 143, 144, 145, 146)
FORBIDDEN = ("score", "final_result", "y_fail")


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def prepare(features_path: Path, output: Path) -> dict:
    frame = pd.read_parquet(features_path)
    leaked = [c for c in FEATURES if any(t in c.lower() for t in FORBIDDEN)]
    if leaked:
        raise ValueError(f"Score or outcome column in the deployable set: {leaked}")
    missing = [c for c in FEATURES if c not in frame]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    if len(frame) != 19806 or int(frame.y_fail.sum()) != 6092:
        raise ValueError("Unexpected cohort; this benchmark requires the frozen cohort")

    numeric = frame[FEATURES].to_numpy(dtype=float)
    y = frame.y_fail.to_numpy(dtype=np.int64)
    students = frame.id_student.to_numpy(dtype=np.int64)
    category = pd.get_dummies(frame.module_presentation, dtype=float)
    if category.shape[1] != 19:
        raise ValueError(f"Expected 19 module-presentation categories, got {category.shape[1]}")
    names = FEATURES + [f"module_presentation={c}" for c in category.columns]

    payload = {"y": y, "student": students, "feature_names": np.asarray(names)}
    baselines = {}
    for seed in SEEDS:
        train, test = next(GroupShuffleSplit(
            n_splits=1, test_size=.2, random_state=seed)
            .split(frame, groups=students))
        if set(students[train]) & set(students[test]):
            raise AssertionError("Student appears in both train and test")
        # The published M2 baseline on this split, refitted here so the NPZ ships
        # its own reference vector.
        baseline = _fit_predict(frame.iloc[train], frame.iloc[test],
                                FEATURES + ["module_presentation"])
        ap = average_precision_score(y[test], baseline)
        baselines[str(seed)] = float(ap)
        transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        transform.fit(numeric[train])
        # Fitting the unsupervised transform on the full outer train set keeps
        # this identical to the earlier experiment's inner-validation setup; no
        # label information enters it.
        x = np.concatenate([transform.transform(numeric), category.to_numpy()],
                           axis=1).astype("float32")
        if x.shape[1] != len(FEATURES) + 19:
            raise AssertionError("Unexpected design-matrix width")
        payload.update({f"x_{seed}": x, f"train_{seed}": train,
                        f"test_{seed}": test, f"baseline_{seed}": baseline})
        print(f"seed={seed} test={len(test)} baseline_ap={ap:.9f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    provenance = {
        "purpose": "score-free XGBoost arm on the frozen M2 feature set",
        "features": FEATURES,
        "n_features_numeric": len(FEATURES),
        "n_category": int(category.shape[1]),
        "seeds": list(SEEDS),
        "n_records": int(len(frame)),
        "n_fail": int(y.sum()),
        "baseline_model": "M2 logistic regression (no scores)",
        "baseline_ap_by_seed": baselines,
        "features_sha256": sha256(features_path),
        "script_sha256": sha256(Path(__file__)),
        "npz_sha256": sha256(output),
        "score_information_used": False,
    }
    (output.with_suffix(".json")).write_text(json.dumps(provenance, indent=2),
                                             encoding="utf-8")
    print(json.dumps({"baseline_ap_by_seed": baselines,
                      "npz_sha256": provenance["npz_sha256"]}, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=REPO / "oulad-sem/data/processed/features.parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.features, args.output)
