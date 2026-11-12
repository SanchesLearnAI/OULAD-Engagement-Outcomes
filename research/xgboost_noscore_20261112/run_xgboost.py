"""Train GPU XGBoost on the score-free M2 feature set and compare with the frozen baseline.

Same five student-grouped 80/20 splits as ``xgboost_score_20261111``, same
hyper-parameter search, but no mark information: this is the deployable arm.

Run on a CUDA host from the unzipped input directory::

    python run_xgboost.py --input xgb_noscore.npz --output-dir xgb_noscore_run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

SEEDS = (42, 143, 144, 145, 146)
CANDIDATES = ((3, .03), (5, .03), (3, .05))


def sha256(path: Path) -> str:
    """``hashlib.file_digest`` needs Python 3.11; the GPU host runs 3.8."""
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def classifier(seed: int, depth: int, rate: float, iterations: int,
               early_stopping: bool, device: str = "cuda") -> XGBClassifier:
    return XGBClassifier(
        n_estimators=iterations, max_depth=depth, learning_rate=rate,
        tree_method="hist", device=device, subsample=.85,
        colsample_bytree=.85, reg_lambda=1.0, eval_metric="aucpr",
        early_stopping_rounds=35 if early_stopping else None,
        importance_type="gain", random_state=seed, n_jobs=8)


def assert_device(model: XGBClassifier, expected: str) -> str:
    config = json.loads(model.get_booster().save_config())
    observed = config["learner"]["generic_param"]["device"]
    if expected == "cuda" and not observed.startswith("cuda"):
        raise RuntimeError(f"XGBoost did not train on CUDA: {observed}")
    return observed


def bootstrap_interval(y, prediction, baseline, students) -> dict:
    members = [np.flatnonzero(students == student) for student in np.unique(students)]
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(400):
        chosen = np.concatenate([members[i] for i in
                                 rng.integers(len(members), size=len(members))])
        sampled = y[chosen]
        if 0 < sampled.sum() < len(sampled):
            deltas.append(average_precision_score(sampled, prediction[chosen]) -
                          average_precision_score(sampled, baseline[chosen]))
    return {"seed": 42,
            "delta_ap": float(average_precision_score(y, prediction) -
                              average_precision_score(y, baseline)),
            "ci_low": float(np.quantile(deltas, .025)),
            "ci_high": float(np.quantile(deltas, .975)),
            "bootstrap_repeats": 400, "resampling_unit": "id_student"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu; cpu is allowed here so the arm can run "
                             "without a GPU, but the CUDA run is the reference")
    parser.add_argument("--max-rounds", type=int, default=600)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    y = data["y"]
    students = data["student"]
    names = [str(n) for n in data["feature_names"]]
    observed_device = None
    rows, importance_rows, intervals = [], [], []
    for seed in SEEDS:
        x = data[f"x_{seed}"]
        train = data[f"train_{seed}"]
        test = data[f"test_{seed}"]
        baseline = data[f"baseline_{seed}"]
        if set(students[train]) & set(students[test]):
            raise AssertionError("Student appears in both train and test")

        best = None
        for depth, rate in CANDIDATES:
            outer_tr, inner_va = next(GroupShuffleSplit(
                n_splits=1, test_size=.2, random_state=seed + 7)
                .split(x[train], groups=students[train]))
            model = classifier(seed, depth, rate, args.max_rounds, True, args.device)
            model.fit(x[train][outer_tr], y[train][outer_tr],
                      eval_set=[(x[train][inner_va], y[train][inner_va])],
                      verbose=False)
            observed_device = assert_device(model, args.device)
            score = average_precision_score(y[train][inner_va],
                                            model.predict_proba(x[train][inner_va])[:, 1])
            if best is None or score > best["score"]:
                best = {"score": float(score), "depth": depth, "rate": rate,
                        "rounds": int(model.best_iteration or args.max_rounds)}

        final = classifier(seed, best["depth"], best["rate"], best["rounds"] + 1,
                           False, args.device)
        final.fit(x[train], y[train], verbose=False)
        prob = final.predict_proba(x[test])[:, 1]
        rows.append({
            "seed": seed, "n_test": int(len(test)),
            "baseline_ap": float(average_precision_score(y[test], baseline)),
            "xgboost_ap": float(average_precision_score(y[test], prob)),
            "delta_ap": float(average_precision_score(y[test], prob) -
                              average_precision_score(y[test], baseline)),
            "baseline_roc_auc": float(roc_auc_score(y[test], baseline)),
            "xgboost_roc_auc": float(roc_auc_score(y[test], prob)),
            "baseline_brier": float(brier_score_loss(y[test], baseline)),
            "xgboost_brier": float(brier_score_loss(y[test], prob)),
            "validation_ap": best["score"], "max_depth": best["depth"],
            "learning_rate": best["rate"], "n_estimators": best["rounds"],
        })
        if seed == 42:
            intervals.append(bootstrap_interval(y[test], prob, baseline,
                                                students[test]))
        gains = final.feature_importances_
        for name, gain in zip(names, gains):
            importance_rows.append({"seed": seed, "feature": name,
                                    "gain_importance": float(gain)})
        print(f"seed={seed} baseline={rows[-1]['baseline_ap']:.6f} "
              f"xgboost={rows[-1]['xgboost_ap']:.6f} "
              f"delta={rows[-1]['delta_ap']:+.6f}", flush=True)

    save_csv(args.output_dir / "metrics.csv", rows)
    save_csv(args.output_dir / "interval_seed42.csv", intervals)
    save_csv(args.output_dir / "importance_by_seed.csv", importance_rows)
    summary = {}
    for row in importance_rows:
        summary.setdefault(row["feature"], []).append(row["gain_importance"])
    save_csv(args.output_dir / "importance_summary.csv", [
        {"feature": f, "mean_gain_importance": float(np.mean(v)),
         "min_gain_importance": float(np.min(v)),
         "max_gain_importance": float(np.max(v))}
        for f, v in sorted(summary.items(), key=lambda kv: -np.mean(kv[1]))])
    provenance = {
        "baseline": "M2 logistic regression, 16 score-free features",
        "seeds": list(SEEDS), "positive_class": "Fail",
        "n_records": int(len(y)), "n_fail": int(y.sum()),
        "feature_count_numeric": len(names) - 19,
        "feature_count_category": 19,
        "score_information_used": False,
        "selection": "inner student-group validation sklearn AP; final outer-train refit",
        "candidate_depth_rate": [list(c) for c in CANDIDATES],
        "device": observed_device, "requested_device": args.device,
        "xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "input_sha256": sha256(args.input),
        "test_previously_inspected": True,
        "limitations": ("Retrospective comparison on a repeatedly inspected split; "
                        "the baseline itself was selected after seeing these data. "
                        "A positive delta is an exploratory gain, not deployment value."),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2),
                                                     encoding="utf-8")
    mean_delta = float(np.mean([r["delta_ap"] for r in rows]))
    print(json.dumps({"mean_baseline_ap": float(np.mean([r["baseline_ap"] for r in rows])),
                      "mean_xgboost_ap": float(np.mean([r["xgboost_ap"] for r in rows])),
                      "mean_delta_ap": mean_delta,
                      "all_five_positive": bool(all(r["delta_ap"] > 0 for r in rows)),
                      "device": observed_device}, indent=2))


if __name__ == "__main__":
    main()
