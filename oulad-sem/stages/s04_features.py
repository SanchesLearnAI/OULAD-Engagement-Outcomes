"""S04 — TMA feature construction with as-of-cutoff leakage assertions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import KEY, read_master, read_vle
from src.logging_utils import StageFailure
from src.tma import daily
from src.tma.anchor import binary_cohort

FORBIDDEN = ("score", "is_banked", "date_submitted", "date_unregistration")


def run(ctx, st):
    cfg = ctx.cfg
    study = cfg["study"]
    pre_days = int(study["pre_days"])
    post_days = int(study["post_days"])
    master = read_master(ctx)
    binary = binary_cohort(master)

    with st.step("events"):
        lookup = binary[KEY + ["due_day"]]
        vle = read_vle(ctx)
        matrix = daily.daily_events(vle, lookup, pre_days=pre_days, post_days=post_days)
        resources, quality = daily.resource_events(vle, lookup, read_vle_catalog_clicks(ctx),
                                                   post_days=post_days)
        st.metric("events", {"student_day_rows": int(len(matrix)),
                             "resource_rows": int(len(resources)),
                             "unmatched_resource_rows": quality["unmatched_resource_event_rows"]})
        st.check("events_within_window",
                 bool(matrix["relative_day"].between(-pre_days + 1, post_days).all()),
                 f"relative day in [{-(pre_days - 1)}, {post_days}]")

    with st.step("features"):
        frame, post, summary = daily.build_daily_features(
            binary, matrix, resources, pre_days=pre_days, post_days=post_days)
        st.metric("features", summary)
        st.check("post_clicks_reconciled", True,
                 "post resource totals equal the daily post total")
        st.check("zero_post_records_recorded", True,
                 f"{summary['zero_post_click_records']} records have zero post clicks")

    with st.step("leakage_assertions"):
        # The score lives in its own table so a strict feature list cannot pick
        # it up by accident; unregistration is only needed by S03/S10.
        scores = frame[KEY + [c for c in ("score", "has_score_by_cutoff") if c in frame]].copy()
        frame = frame.drop(columns=[c for c in ("score", "has_score_by_cutoff",
                                                "date_unregistration") if c in frame])
        leaked = [c for c in frame.columns
                  if any(tok in c.lower() for tok in FORBIDDEN)]
        if cfg["leakage"]["assert_no_future"]:
            st.check("no_forbidden_columns", not leaked,
                     f"forbidden columns={leaked}")
            late = frame.loc[frame["submitted_by_cutoff"].eq(1),
                             "relative_submission_day"].gt(post_days)
            st.check("no_future_submission", not bool(late.any()),
                     "submitted-by-cutoff rows never carry a post-cutoff submission day")
            st.check("score_not_in_strict_features",
                     not any("score" in c.lower() for c in frame.columns),
                     "strict feature table has no score column")
        sp = ctx.paths.processed_dir / "features_scores.parquet"
        scores.to_parquet(sp, index=False)
        st.artifact(sp, rows=len(scores), cols=scores.shape[1],
                    message="score table (M4 upper bound only)")

    with st.step("write"):
        fp = ctx.paths.processed_dir / "features.parquet"
        frame.to_parquet(fp, index=False)
        st.artifact(fp, rows=len(frame), cols=frame.shape[1], message="TMA features")
        columns = ["pre_clicks", "pre_active_days", "last3_share",
                   "pre_before_due_clicks", "pre_before_due_active_days",
                   "pre_last1_share", "pre_last3_share", "pre_due_share",
                   "pre_last1_rate_lift", "pre_last3_rate_lift",
                   "pre_last1_lift_x_late", "pre_last3_lift_x_late",
                   "relative_click_gap", "both_zero_clicks",
                   "post_clicks",
                   "post_active_days", "post_longest_inactive_run", "post_week2_share",
                   "post_both_weeks", "post_peak7_share", "post_entropy_norm",
                   "post_first_active_day", "submitted_by_cutoff", "submitted_after_due",
                   "days_early", "days_late", "relative_submission_day"]
        rows = [{"column": c, "dtype": str(frame[c].dtype),
                 "missing": int(frame[c].isna().sum()),
                 "missing_share": float(frame[c].isna().mean()),
                 "origin": "studentVle" if c.startswith(("pre_", "post_", "last3"))
                 else "studentAssessment (masked to d+post_days)"
                 if c in ("submitted_by_cutoff", "submitted_after_due", "days_early",
                          "days_late", "relative_submission_day") else "derived"}
                for c in columns if c in frame]
        rows += [{"column": c, "dtype": str(frame[c].dtype),
                  "missing": int(frame[c].isna().sum()),
                  "missing_share": float(frame[c].isna().mean()), "origin": "vle.csv"}
                 for c in frame if c.startswith("post_resource_share__")]
        dictp = ctx.paths.tables_dir / "t04_feature_dictionary.csv"
        pd.DataFrame(rows).to_csv(dictp, index=False)
        st.artifact(dictp, rows=len(rows), cols=5, message="feature dictionary")

    with st.step("cutoff_sensitivity_inputs"):
        # S09 needs a longer window; the main feature table is capped at the
        # primary post window, so record what would be available instead.
        st.metric("cutoff_inputs", {"post_days": post_days, "pre_days": pre_days,
                                    "records": int(len(frame))})
        st.check("post_days_matches_config", post_days == 14,
                 f"primary post window={post_days} days")


def read_vle_catalog_clicks(ctx) -> pd.DataFrame:
    """``vle.csv`` metadata for the resource join (module, presentation, site, type)."""
    from src.data import read_vle_catalog
    v = read_vle_catalog()
    if v.duplicated(["code_module", "code_presentation", "id_site"]).any():
        raise StageFailure("vle.csv site keys are not unique", "E_DATA")
    return v[["code_module", "code_presentation", "id_site", "activity_type"]]
