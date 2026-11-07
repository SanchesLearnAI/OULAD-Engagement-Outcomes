"""S05 — grouped splits + temporal + leave-one-module-out, with hard overlap checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import KEY, read_master
from src.logging_utils import atomic_write_json
from src.splits import build_splits


def _overlap(df, a, b):
    sa = set(df["id_student"].iloc[a].tolist())
    sb = set(df["id_student"].iloc[b].tolist())
    return len(sa & sb)


def run(ctx, st):
    master = read_master(ctx)
    cohort, splits, info = build_splits(master, ctx.cfg.data, ctx.cfg["seed"])
    n = len(cohort)
    y = cohort["y_t1"].astype(int).to_numpy()

    p = ctx.paths.processed_dir / "splits.json"
    atomic_write_json(p, splits)
    st.artifact(p, rows=n, message="splits.json")

    main = splits["main"]
    with st.step("main_split"):
        tr, va, te = main["train"], main["val"], main["test"]
        st.check("main_partition", len(tr) + len(va) + len(te) == n,
                 f"{len(tr)}+{len(va)}+{len(te)} == {n}")
        st.check("main_train_val_student_overlap",
                 _overlap(cohort, tr, va) == 0, "train/val overlap=0")
        st.check("main_train_test_student_overlap",
                 _overlap(cohort, tr, te) == 0, "train/test overlap=0")
        st.check("main_val_test_student_overlap",
                 _overlap(cohort, va, te) == 0, "val/test overlap=0")
        st.check("main_class_distribution_recorded",
                 all(not np.isnan(main["class_distribution"][k]["positive_rate"])
                     for k in ("train", "val", "test")),
                 "positive rate recorded for train/val/test")
        st.metric("main_split", {
            "main_n_train": len(tr), "main_n_val": len(va), "main_n_test": len(te),
            "main_positive_rate": {k: main["class_distribution"][k]["positive_rate"]
                                   for k in ("train", "val", "test")},
        })

    with st.step("group_kfold"):
        seen = np.zeros(n, dtype=int)
        max_overlap = 0
        for f in splits["group_kfold"]:
            trf, tef = f["train"], f["test"]
            seen[tef] += 1
            max_overlap = max(max_overlap, _overlap(cohort, trf, tef))
        st.check("group_fold_disjoint", max_overlap == 0,
                 f"max train/test student overlap={max_overlap}")
        st.check("group_fold_covers_all", bool((seen == 1).all()),
                 f"test-fold coverage min={seen.min()} max={seen.max()}",
                 metrics={"min": int(seen.min()), "max": int(seen.max())})
        st.metric("group_kfold", {"kfold_folds": len(splits["group_kfold"]),
                                  "kfold_max_student_overlap": int(max_overlap)})

    with st.step("temporal"):
        t = splits["temporal"]
        trp = set(t["train_presentations"]); tep = set(t["test_presentations"])
        st.check("temporal_presentations_disjoint", not (trp & tep),
                 f"train={sorted(trp)} test={sorted(tep)}")
        st.check("temporal_student_overlap",
                 _overlap(cohort, t["train"], t["test"]) == 0,
                 "train/test student overlap=0")
        st.check("temporal_test_presentations_expected",
                 set(cohort["code_presentation"].iloc[t["test"]].unique()) <= tep,
                 f"test presentations={sorted(tep)}")
        st.metric("temporal", {"temporal_n_train": len(t["train"]),
                               "temporal_n_test": len(t["test"]),
                               "temporal_dropped_student_overlap":
                                   t["n_dropped_student_overlap"],
                               "excluded_other_presentations": t["n_excluded_other_presentations"]})

    with st.step("lomo"):
        max_ov = 0
        for f in splits["lomo"]:
            max_ov = max(max_ov, _overlap(cohort, f["train"], f["test"]))
        st.check("lomo_student_overlap", max_ov == 0,
                 f"max overlap={max_ov}")
        st.metric("lomo", {"lomo_folds": len(splits["lomo"]),
                           "lomo_max_student_overlap": int(max_ov),
                           "lomo_dropped_student_overlap": int(sum(
                               f["n_dropped_student_overlap"] for f in splits["lomo"]))})

    with st.step("ungrouped_control"):
        rng = np.random.default_rng(ctx.cfg["seed"])
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = int(0.6 * n); n_va = int(0.2 * n)
        tr = idx[:n_tr]; va = idx[n_tr:n_tr + n_va]; te = idx[n_tr + n_va:]
        train_students = set(cohort["id_student"].iloc[tr].tolist())
        test_rows_overlap = int(cohort["id_student"].iloc[te].isin(train_students).sum())
        overlap_students = len(train_students &
                               set(cohort["id_student"].iloc[te].tolist()))
        share = test_rows_overlap / max(len(te), 1)
        st.metric("ungrouped_control", {
            "students_in_train_and_test": overlap_students,
            "test_rows_overlapping": test_rows_overlap,
            "test_row_overlap_share": float(share),
            "note": "row-level control; expected ~852 students / 15.7% in full OULAD",
        })
        if ctx.mode == "full":
            st.check("ungrouped_control_positive", overlap_students > 0,
                     "row-level split does leak students; grouped split must not")
        else:
            st.check("ungrouped_control_recorded", True,
                     f"simulation sample has {overlap_students} repeat students "
                     "across presentations; the control is recorded, not expected "
                     "to reproduce the full-data 852/15.7%")

    # Summary table
    rows = []
    def add(name, tr, te):
        rows.append({"split": name, "n_train": len(tr), "n_test": len(te),
                     "train_positive_rate": float(y[tr].mean()),
                     "test_positive_rate": float(y[te].mean()),
                     "student_overlap": _overlap(cohort, tr, te)})
    add("main_train_test", main["train"], main["test"])
    add("main_train_val", main["train"], main["val"])
    add("temporal", splits["temporal"]["train"], splits["temporal"]["test"])
    for f in splits["group_kfold"]:
        add(f"group_kfold_{f['fold']}", f["train"], f["test"])
    for f in splits["lomo"]:
        add(f"lomo_{f['module']}", f["train"], f["test"])
    tab = pd.DataFrame(rows)
    tp = ctx.paths.tables_dir / "t05_split_summary.csv"
    tab.to_csv(tp, index=False)
    st.artifact(tp, rows=len(tab), cols=tab.shape[1], message="split summary")
    st.metric("splits", info)
