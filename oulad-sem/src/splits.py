"""Grouped, temporal and leave-one-module-out splits."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def _keys_hash(df: pd.DataFrame) -> str:
    s = "\n".join(f"{a}|{b}|{c}" for a, b, c in
                  zip(df["code_module"], df["code_presentation"], df["id_student"]))
    return hashlib.sha256(s.encode()).hexdigest()


def _class_dist(y: pd.Series) -> dict:
    n = int(len(y))
    pos = int(y.sum())
    return {"n": n, "positive": pos, "negative": n - pos,
            "positive_rate": float(pos / n) if n else float("nan")}


def build_splits(master: pd.DataFrame, config: dict, seed: int):
    cohort = master[master["y_t1"].notna()].reset_index(drop=True)
    n = len(cohort)
    y = cohort["y_t1"].astype(int)
    groups = cohort["id_student"].to_numpy()

    # 60/20/20 grouped split.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train, temp = next(gss.split(cohort, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(temp.reshape(-1, 1), y.iloc[temp],
                                        groups[temp]))
    val, test = temp[val_rel], temp[test_rel]

    # 5-fold GroupKFold.
    folds = []
    gkf = GroupKFold(n_splits=int(config["splits"]["n_splits"]))
    for i, (tr, te) in enumerate(gkf.split(cohort, y, groups)):
        folds.append({"fold": i, "train": tr.tolist(), "test": te.tolist()})

    # Forward temporal split.
    tr_pres = list(config["splits"]["temporal_train"])
    te_pres = list(config["splits"]["temporal_test"])
    t_train = cohort.index[cohort["code_presentation"].isin(tr_pres)].to_numpy()
    t_test = cohort.index[cohort["code_presentation"].isin(te_pres)].to_numpy()
    temporal_other = int((~cohort["code_presentation"].isin(tr_pres + te_pres)).sum())
    # Hard assertion: no id_student may be in both sides.  Drop overlapping
    # students from the training side only; the test fold stays intact.
    te_students = set(cohort["id_student"].iloc[t_test].tolist())
    keep = ~cohort["id_student"].iloc[t_train].isin(te_students).to_numpy()
    temporal_dropped = int((~keep).sum())
    t_train = t_train[keep]

    # Leave-one-module-out, also group-aware for the same hard assertion.
    lomo = []
    for mod in sorted(cohort["code_module"].unique()):
        te = cohort.index[cohort["code_module"] == mod].to_numpy()
        tr = cohort.index[cohort["code_module"] != mod].to_numpy()
        te_students = set(cohort["id_student"].iloc[te].tolist())
        keep = ~cohort["id_student"].iloc[tr].isin(te_students).to_numpy()
        dropped = int((~keep).sum())
        tr = tr[keep]
        lomo.append({"module": str(mod), "train": tr.tolist(), "test": te.tolist(),
                     "n_dropped_student_overlap": dropped})

    splits = {
        "schema_version": "1.0",
        "cohort": "T1",
        "target": ("Fail=1, Pass/Distinction=0; Withdrawn excluded"
                   if config.get("cohort", {}).get("positive_class", "fail") == "fail"
                   else "Pass/Distinction=1, Fail=0; Withdrawn excluded"),
        "n": int(n),
        "cohort_keys_sha256": _keys_hash(cohort),
        "seed": int(seed),
        "main": {
            "train": train.tolist(), "val": val.tolist(), "test": test.tolist(),
            "class_distribution": {
                "train": _class_dist(y.iloc[train]),
                "val": _class_dist(y.iloc[val]),
                "test": _class_dist(y.iloc[test]),
            },
        },
        "group_kfold": folds,
        "temporal": {
            "train_presentations": tr_pres, "test_presentations": te_pres,
            "train": t_train.tolist(), "test": t_test.tolist(),
            "n_excluded_other_presentations": temporal_other,
            "n_dropped_student_overlap": temporal_dropped,
            "class_distribution": {
                "train": _class_dist(y.iloc[t_train]),
                "test": _class_dist(y.iloc[t_test]),
            },
        },
        "lomo": lomo,
    }

    info = {
        "n": int(n),
        "cohort_keys_sha256": splits["cohort_keys_sha256"],
        "main": {k: len(splits["main"][k]) for k in ("train", "val", "test")},
        "main_positive_rate": {k: splits["main"]["class_distribution"][k]["positive_rate"]
                               for k in ("train", "val", "test")},
        "n_temporal_train": len(t_train), "n_temporal_test": len(t_test),
        "temporal_dropped_student_overlap": temporal_dropped,
        "n_lomo_modules": len(lomo),
        "lomo_dropped_student_overlap": int(sum(
            f["n_dropped_student_overlap"] for f in lomo)),
        "temporal_other_excluded": temporal_other,
    }
    return cohort, splits, info
