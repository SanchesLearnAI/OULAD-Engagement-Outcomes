"""V5 — sequence-style baselines on the frozen split (no PyTorch available).

PyTorch could not be installed in this environment (the 138 MB wheel stalls on the
local link), so the protocol's fallback applies: a neural model from scikit-learn
on the **ordered** 14-day click vector, plus two order-blind controls.  These are
not recurrent networks, and the results file records that limitation.

Run from ``oulad-sem/``::

    python scripts/verify_sequence_models.py --repeats 200
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import daily, modeling  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]


def topk_precision(y: np.ndarray, p: np.ndarray, k: float) -> float:
    n = max(int(round(len(y) * k)), 1)
    order = np.argsort(-p)
    boundary = p[order[n - 1]]
    above = p > boundary
    ties = p == boundary
    slots = n - int(above.sum())
    hit = y[above].sum() + (y[ties].mean() if ties.any() and slots > 0 else 0) * slots
    return float(hit / n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    post_days = int(cfg["study"]["post_days"])

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    y = frame["y_fail"].to_numpy(dtype=int)

    master = pd.read_parquet(root / "data" / "interim" / "master.parquet")
    cohort = binary_cohort(master)
    events = daily.daily_events(root / "data" / "raw" / "oulad" / "studentVle.csv",
                                cohort[KEY + ["due_day"]], pre_days=14,
                                post_days=post_days)
    matrix = daily.click_matrix(events, pd.MultiIndex.from_frame(frame[KEY]), 1, post_days)
    log_daily = np.log1p(matrix)

    context = pd.get_dummies(frame["code_module"].astype(str), dtype=float)
    context.columns = [f"module_{c}" for c in context.columns]
    daily_cols = [f"d{i+1:02d}" for i in range(post_days)]
    sequence_design = pd.concat(
        [pd.DataFrame(log_daily, columns=daily_cols, index=frame.index),
         context.reset_index(drop=True)], axis=1)

    # Order-blind controls: same information, scrambled day order, plus a bag
    # summary (total and active days) so the comparison isolates ordering.
    rng = np.random.default_rng(cfg["seed"])
    shuffled = pd.DataFrame(log_daily[:, rng.permutation(post_days)], columns=daily_cols,
                            index=frame.index)
    bag = pd.DataFrame({"total_log": log_daily.sum(axis=1),
                        "active_days": (matrix > 0).sum(axis=1).astype(float)},
                       index=frame.index)

    def fit_mlp(X: pd.DataFrame, seed: int) -> np.ndarray:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64, 32),
                                            max_iter=args.epochs, early_stopping=True,
                                            n_iter_no_change=5, random_state=seed))
        model.fit(X.iloc[tr], y[tr])
        return model.predict_proba(X.iloc[te])[:, 1]

    # Fair version of the same test: keep the M2 non-daily features and add the
    # ordered daily vector, so the contrast isolates the representation rather
    # than starving the neural model of the submission/timing block.
    m2_all = [c for c in modeling.columns_for(frame, "M2") if c in frame]
    m2_non_daily = [c for c in m2_all if not c.startswith(("log_post", "post_clicks"))]
    m2_block = frame[m2_non_daily].reset_index(drop=True).astype(float)
    rows, preds = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, X in (("mlp_ordered_daily", sequence_design),
                        ("mlp_shuffled_daily", pd.concat([shuffled, context], axis=1)),
                        ("mlp_bag_summary", pd.concat([bag, context], axis=1)),
                        ("mlp_ordered_plus_M2", pd.concat(
                            [sequence_design.reset_index(drop=True), m2_block], axis=1)),
                        ("mlp_bag_plus_M2", pd.concat(
                            [bag.reset_index(drop=True), context.reset_index(drop=True),
                             m2_block], axis=1))):
            prob = fit_mlp(X, cfg["seed"])
            preds[name] = prob
            rows.append({"model": name, "n_features": int(X.shape[1]),
                         "average_precision": float(average_precision_score(y[te], prob)),
                         "roc_auc": float(roc_auc_score(y[te], prob)),
                         "brier": float(brier_score_loss(y[te], prob)),
                         "top5_precision": topk_precision(y[te], prob, 0.05),
                         "no_skill": float(y[te].mean()), "n_test": int(len(te))})

    m2_cols = [c for c in modeling.columns_for(frame, "M2") if c in frame]
    m2_prob, m2_thr, m2_rule = modeling._fit(frame, m2_cols, tr, te, "logistic",
                                             cfg["seed"], va, 0.80)
    preds["M2_logistic_reference"] = m2_prob
    rows.append({"model": "M2_logistic_reference", "n_features": len(m2_cols),
                 "average_precision": float(average_precision_score(y[te], m2_prob)),
                 "roc_auc": float(roc_auc_score(y[te], m2_prob)),
                 "brier": float(brier_score_loss(y[te], m2_prob)),
                 "top5_precision": topk_precision(y[te], m2_prob, 0.05),
                 "no_skill": float(y[te].mean()), "n_test": int(len(te))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "v5_sequence_models.csv", index=False)

    # Paired bootstrap on the two contrasts that matter.
    groups = frame["id_student"].to_numpy()[te]
    members = list(pd.Series(np.arange(len(te))).groupby(groups).indices.values())
    boot = np.random.default_rng(cfg["seed"] + 808)
    contrasts = []
    for a, b, label in (("mlp_ordered_daily", "M2_logistic_reference",
                         "ordered_mlp_minus_M2"),
                        ("mlp_ordered_daily", "mlp_shuffled_daily",
                         "ordered_minus_shuffled"),
                        ("mlp_ordered_plus_M2", "M2_logistic_reference",
                         "ordered_mlp_plus_M2_minus_M2"),
                        ("mlp_ordered_plus_M2", "mlp_bag_plus_M2",
                         "ordered_plus_M2_minus_bag_plus_M2")):
        deltas = []
        for _ in range(args.repeats):
            pick = np.concatenate([members[i] for i in boot.integers(len(members),
                                                                   size=len(members))])
            if len(np.unique(y[te][pick])) < 2:
                continue
            deltas.append(average_precision_score(y[te][pick], preds[a][pick])
                          - average_precision_score(y[te][pick], preds[b][pick]))
        deltas = np.asarray(deltas)
        contrasts.append({"contrast": label, "point": float(deltas.mean()),
                          "ci_low": float(np.quantile(deltas, .025)),
                          "ci_high": float(np.quantile(deltas, .975)),
                          "replicates": int(len(deltas))})
    contrast_table = pd.DataFrame(contrasts)
    contrast_table.to_csv(out / "v5_sequence_contrasts.csv", index=False)

    payload = {
        "backend": "scikit-learn MLPClassifier (PyTorch unavailable: 138 MB wheel did "
                   "not finish downloading on this link)",
        "is_recurrent": False,
        "epochs": args.epochs,
        "models": table[["model", "average_precision", "roc_auc", "brier",
                         "top5_precision"]].round(4).to_dict("records"),
        "contrasts": contrast_table.round(4).to_dict("records"),
        "ordered_beats_shuffled": bool(
            contrast_table.set_index("contrast").loc["ordered_minus_shuffled", "ci_low"] > 0),
        "mlp_beats_m2": bool(
            contrast_table.set_index("contrast").loc["ordered_mlp_minus_M2", "ci_low"] > 0),
        "mlp_with_M2_features_beats_M2": bool(
            contrast_table.set_index("contrast")
            .loc["ordered_mlp_plus_M2_minus_M2", "ci_low"] > 0),
        "order_helps_when_M2_features_present": bool(
            contrast_table.set_index("contrast")
            .loc["ordered_plus_M2_minus_bag_plus_M2", "ci_low"] > 0),
        "caveat": ("A 14-day MLP on ordered daily clicks is not a GRU or Transformer, so "
                   "this cannot confirm or refute published sequence-model numbers; it "
                   "only tests whether day ordering adds information beyond volume within "
                   "this design."),
    }
    (out / "v5_sequence_verdict.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
