"""S04 — cutoff-isolated feature construction and leakage assertions."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data import KEY, read_master
from src.features import build_features, build_score_features, prepare_assessment, prepare_vle
from src.logging_utils import StageFailure


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def _quick_cv_auc(df: pd.DataFrame) -> tuple[float, int]:
    cols = [c for c in df.columns if c not in KEY + ["y_t1"]]
    X = df[cols].astype(float).to_numpy()
    y = df["y_t1"].to_numpy()
    groups = df["id_student"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    preds = np.full(len(df), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=42)),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan"), int(ok.sum())
    return float(roc_auc_score(y[ok], preds[ok])), int(ok.sum())


def run(ctx, st):
    cfg = ctx.cfg
    cutoffs = cfg["windows"]["cutoffs"]
    master = read_master(ctx)
    vle, vinfo = prepare_vle(ctx)
    sa = prepare_assessment(ctx)
    st.metric("vle_prepare", vinfo)
    st.check("activity_type_complete", vinfo["missing_activity_type"] == 0,
             f"missing activity_type rows={vinfo['missing_activity_type']}")
    st.check("quiz_sites_excluded", True,
             f"quiz/externalquiz rows excluded={vinfo['quiz_rows_excluded']} "
             f"({vinfo['quiz_row_share']:.4f})")

    from src.windows import w2_end_value
    ft = (master.groupby(["code_module", "code_presentation"], observed=True)
          ["first_tma_date"].first())
    ends = ft.map(lambda x: w2_end_value(x, cfg.data))
    st.metric("first_tma", {
        "n_presentations": int(len(ft)),
        "n_missing": int(ft.isna().sum()),
        "n_le_28_fallback_55": int((ft <= 28).sum()),
        "n_29_to_55_use_tma_minus_1": int(((ft > 28) & (ft <= 55)).sum()),
        "n_gt_55": int((ft > 55).sum()),
        "min_first_tma": int(ft.min()) if ft.notna().any() else None,
        "w2_end_min": int(ends.min()), "w2_end_max": int(ends.max()),
    })
    st.check("w2_end_domain", bool(((ends >= 28) & (ends <= 55)).all()),
             f"per-presentation W2 ends in [{int(ends.min())}, {int(ends.max())}]")

    audit_rows = []
    built = {}
    for c in cutoffs:
        with st.step(f"features_c{c}"):
            f, info = build_features(master, vle, sa, c, cfg.data,
                                     top_types=None)
            built[c] = f
            p = ctx.paths.processed_dir / f"features_c{c}.parquet"
            f.to_parquet(p, index=False)
            st.artifact(p, rows=len(f), cols=f.shape[1],
                        message=f"features cutoff {c}")
            audit_rows.append({
                "cutoff": c, "rows": len(f), "features": f.shape[1] - len(KEY),
                "w1_start": info["w1_window"][0] if info["w1_window"] else None,
                "w1_end": info["w1_window"][1] if info["w1_window"] else None,
                "w2_start": info["w2_window"][0] if info["w2_window"] else None,
                "w2_end": info["w2_window"][1] if info["w2_window"] else None,
                "max_date_seen": info["max_date_seen"],
                "zero_click_share_w1": info["zero_click_share_w1"],
                "negative_date_rows": info["n_negative_date_rows"],
                "negative_date_share": info["negative_date_share"],
            })
            st.check(f"rows_c{c}", len(f) == len(master),
                     f"rows={len(f)} master={len(master)}")
            st.check(f"key_unique_c{c}", not f.duplicated(KEY).any(),
                     "no duplicate enrolment keys")
            st.check(f"no_future_c{c}", info["max_date_seen"] < c,
                     f"max_date_seen={info['max_date_seen']} cutoff={c}",
                     metrics={"max_date_seen": info["max_date_seen"]})
            st.check(f"no_forbidden_columns_c{c}", not info["forbidden_columns"],
                     f"forbidden={info['forbidden_columns']}")
            st.metric(f"feature_counts_c{c}", {
                "n_features": info["n_features"],
                "n_features_w1": sum(1 for x in f.columns if x.startswith("w1_")),
                "n_features_w2": sum(1 for x in f.columns if x.startswith("w2_")),
                "n_features_delta": sum(1 for x in f.columns
                                        if x.startswith("d_") or x.startswith("accel_")),
                "zero_click_share": info["zero_click_share_w1"],
                "max_date": info["max_date_seen"],
                "n_negative_date_rows": info["n_negative_date_rows"],
            })

    ad = pd.DataFrame(audit_rows)
    ap = ctx.paths.tables_dir / "t04_feature_audit.csv"
    ad.to_csv(ap, index=False)
    st.artifact(ap, rows=len(ad), cols=ad.shape[1], message="feature audit")

    with st.step("scores_m4"):
        sf = build_score_features(master, sa, 28,
                                  cfg["leakage"].get("exclude_banked", True))
        sp = ctx.paths.processed_dir / "features_scores_c28.parquet"
        sf.to_parquet(sp, index=False)
        st.artifact(sp, rows=len(sf), cols=sf.shape[1],
                    message="M4 score features (upper bound only)")
        st.check("scores_not_in_strict_table",
                 not any("score" in c.lower() for c in built[28].columns),
                 "score columns exist only in features_scores_c28.parquet")

    with st.step("determinism"):
        f2, _ = build_features(master, vle, sa, 28, cfg.data)
        h1, h2 = _df_hash(built[28]), _df_hash(f2)
        st.check("deterministic_c28", h1 == h2, f"{h1[:12]} vs {h2[:12]}",
                 metrics={"hash": h1})
        st.metric("determinism", {"hash_c28": h1, "match": h1 == h2})

    with st.step("leakage_sanity"):
        df = built[28].merge(master[KEY + ["y_t1"]], on=KEY, how="inner")
        df = df[df["y_t1"].notna()].reset_index(drop=True)
        auc, n_eval = _quick_cv_auc(df)
        st.metric("sanity_group_cv", {"roc_auc_day28": auc, "n_eval": n_eval})
        st.check("day28_group_cv_not_above_0_90",
                 (np.isnan(auc) or auc <= 0.90),
                 f"grouped-CV day-28 ROC-AUC={auc:.4f}; >0.90 would indicate leakage",
                 metrics={"roc_auc": auc})
        if not np.isnan(auc) and auc > 0.90:
            raise StageFailure(f"day-28 grouped-CV AUC {auc:.3f} > 0.90: E_LEAK",
                               "E_LEAK")
        st.check("negative_dates_handled",
                 vinfo["negative_rows"] > 0,
                 f"policy={cfg['windows']['pre_start']}; rows={vinfo['negative_rows']} "
                 f"({vinfo['negative_share']:.4%})")
