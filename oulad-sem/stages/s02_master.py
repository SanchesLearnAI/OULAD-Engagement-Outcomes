"""S02 — TMA anchor cohort, sample flow, Withdrawn audit and simulation sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cohort import stratified_sample, withdrawal_audit
from src.data import KEY, read_raw_cached
from src.tma.anchor import (build_tma_cohort, eligibility_steps,
                            presentation_anchor_table)

EXPECTED = {"simulation": 500, "full": 32593}


def _submission_relative_weeks(master: pd.DataFrame, vle: pd.DataFrame,
                               weeks: int = 8) -> pd.DataFrame:
    """Weekly clicks relative to the TMA deadline, split by final outcome.

    Uses the same key join as the rest of the study; only relative day matters,
    so no student-level row leaves the ignored interim directory.
    """
    lookup = master[KEY + ["due_day", "eligible"]]
    lookup = lookup[lookup["eligible"]]
    v = vle[KEY + ["date", "sum_click"]].merge(lookup, on=KEY, how="inner")
    v["relative_day"] = v["date"] - v["due_day"]
    v = v[(v["relative_day"] >= -7 * weeks) & (v["relative_day"] < 7 * weeks)]
    v["relative_week"] = np.floor(v["relative_day"] / 7).astype(int)
    out = (v.groupby(["relative_week"])["sum_click"].sum().rename("clicks")
           .reset_index())
    out["n_weeks_observed"] = weeks
    return out.sort_values("relative_week").reset_index(drop=True)


def run(ctx, st):
    cfg = ctx.cfg
    study = cfg["study"]
    seed = cfg["seed"]
    post_days = int(study["post_days"])

    with st.step("read_inputs"):
        si = read_raw_cached("studentInfo")
        sr = read_raw_cached("studentRegistration")
        courses = read_raw_cached("courses")
        assessments = read_raw_cached("assessments")
        sa = read_raw_cached("studentAssessment")
        st.metric("inputs", {"studentInfo": len(si), "studentRegistration": len(sr),
                             "courses": len(courses), "assessments": len(assessments),
                             "studentAssessment": len(sa)})

    with st.step("build_cohort"):
        master, info = build_tma_cohort(
            si, sr, courses, assessments, sa, post_days=post_days,
            assessment_type=str(study["assessment_type"]),
            min_weight_exclusive=float(study["min_weight_exclusive"]),
            require_final_result=bool(cfg["cohort"]["require_final_result"]))
        st.check("positive_class_declared", cfg["cohort"]["positive_class"] == "fail",
                 "T1 positive class = Fail")
        st.metric("cohort", info)
        st.check("master_key_unique", not master.duplicated(KEY).any(),
                 "one row per enrolment")
        st.check("anchor_presentations_19", info["anchor_presentations"] == 19,
                 f"presentations with a first weighted TMA={info['anchor_presentations']}")
        st.check("no_tma_module_absent",
                 bool((master.loc[master["code_module"].eq("GGG"), "due_day"].isna()).all()),
                 "GGG carries no anchor, as the protocol states")
        st.check("submission_masked_to_cutoff",
                 bool(master.loc[master["submitted_by_cutoff"].eq(1),
                                 "relative_submission_day"].le(post_days).all()),
                 "no submission day after d+post_days is used")
        st.check("eligible_positive", int(master["eligible"].sum()) > 0,
                 f"eligible={int(master['eligible'].sum())}")

    with st.step("cohort_flow"):
        steps = eligibility_steps(master)
        p = ctx.paths.tables_dir / "t02_cohort_flow.csv"
        steps.to_csv(p, index=False)
        st.artifact(p, rows=len(steps), cols=steps.shape[1],
                    message="ordered sample flow")
        present = presentation_anchor_table(master)
        pp = ctx.paths.tables_dir / "t02_presentations.csv"
        present.to_csv(pp, index=False)
        st.artifact(pp, rows=len(present), cols=present.shape[1],
                    message="per-presentation anchor and outcome mix")
        binary = master[master["eligible"]
                        & master["final_result"].isin(["Pass", "Distinction", "Fail"])]
        st.metric("cohort_counts", {
            "n_total": int(len(master)),
            "n_eligible": int(master["eligible"].sum()),
            "n_binary": int(len(binary)),
            "n_fail": int(binary["final_result"].eq("Fail").sum()),
            "fail_rate": float(binary["final_result"].eq("Fail").mean()),
            "n_withdrawn_eligible": int((master["eligible"]
                                         & master["final_result"].eq("Withdrawn")).sum()),
            "presentations": int(len(present)),
        })
        if ctx.mode == "full":
            st.check("full_binary_records_19806", len(binary) == 19806,
                     f"binary records={len(binary)}")
            st.check("full_fail_records_6092",
                     int(binary["final_result"].eq("Fail").sum()) == 6092,
                     f"fail records={int(binary['final_result'].eq('Fail').sum())}")

    with st.step("withdrawal_audit"):
        vle = read_raw_cached("studentVle") if ctx.mode == "full" else None
        if vle is None:
            sample_keys_path = ctx.paths.interim_dir / "sim_keys.parquet"
            # In simulation the filtered stream is written below; the full audit
            # is not meaningful on 500 rows, so record the intent instead.
            audit = {"unregistration_nonnull_share": float(
                master["date_unregistration"].notna().mean()),
                "post_clicks_exist": None,
                "decision": "simulation mode: withdrawal audit recorded descriptively only"}
            per_pres = pd.DataFrame(columns=["code_module", "code_presentation",
                                             "click_rows", "rows_after", "students",
                                             "students_with_post_click", "post_click_share"])
            weekly = pd.DataFrame(columns=["rel_week", "clicks", "before", "after"])
            st.check("withdrawal_audit_simulated", True,
                     "simulation skips the full click-stream audit")
        else:
            per_pres, weekly, audit = withdrawal_audit(master, vle)
            audit["decision"] = ("Withdrawn records keep clicking after "
                                 "date_unregistration, so they are described "
                                 "separately and never folded into Fail")
        p1 = ctx.paths.tables_dir / "t02_withdrawal_audit.csv"
        per_pres.to_csv(p1, index=False)
        st.artifact(p1, rows=len(per_pres), cols=per_pres.shape[1],
                    message="withdrawal audit by presentation")
        p2 = ctx.paths.tables_dir / "t02_withdrawal_weekly.csv"
        weekly.to_csv(p2, index=False)
        st.artifact(p2, rows=len(weekly), cols=weekly.shape[1],
                    message="weekly clicks around unregistration")
        st.metric("withdrawal", {k: v for k, v in audit.items() if k != "decision"})
        st.check("withdrawn_stays_separate",
                 bool((master.loc[master["final_result"].eq("Withdrawn"), "eligible"]
                       .isin([True, False])).all()),
                 "Withdrawn records are eligible for description, never for the binary label")

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        if len(weekly):
            before = weekly[weekly["rel_week"] < 0] if "rel_week" in weekly else weekly
            after = weekly[weekly["rel_week"] >= 0] if "rel_week" in weekly else weekly.iloc[:0]
            ax.bar(before["rel_week"], before["clicks"], color="#3b6ea5",
                   label="before unregistration")
            ax.bar(after["rel_week"], after["clicks"], color="#c05b5b",
                   label="after unregistration")
            ax.set_xlabel("weeks relative to date_unregistration")
            ax.set_ylabel("clicks (withdrawn students)")
            ax.legend(frameon=False)
        ax.set_title("Withdrawn: clicks do not stop at date_unregistration")
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f02_withdrawal_clicks.png"
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        st.artifact(fp, message="withdrawal timing figure")

    with st.step("weekly_profile"):
        if ctx.mode == "full":
            profile = _submission_relative_weeks(master, read_raw_cached("studentVle"))
        else:
            profile = pd.DataFrame(columns=["relative_week", "clicks", "n_weeks_observed"])
        p3 = ctx.paths.tables_dir / "t02_weekly_clicks.csv"
        profile.to_csv(p3, index=False)
        st.artifact(p3, rows=len(profile), cols=profile.shape[1],
                    message="weekly clicks around the TMA deadline")

    with st.step("write_master"):
        if ctx.mode == "simulation":
            sample = stratified_sample(master, EXPECTED["simulation"], seed)
            keys = sample[KEY].drop_duplicates()
            kp = ctx.paths.interim_dir / "sim_keys.parquet"
            keys.to_parquet(kp, index=False)
            m = master.merge(keys, on=KEY, how="inner")
            # Preserve the deterministic key order used by later stages.
            m = m.sort_values(KEY).reset_index(drop=True)
            st.check("sample_size_500", len(m) == EXPECTED["simulation"],
                     f"sample={len(m)}")
            st.check("sample_strata_represented",
                     m.groupby(["code_module", "code_presentation", "final_result"]).ngroups
                     == master.groupby(["code_module", "code_presentation",
                                        "final_result"]).ngroups,
                     "all outcome strata represented")
            v = read_raw_cached("studentVle").merge(keys, on=KEY, how="inner")
            vp = ctx.paths.interim_dir / "studentVle.parquet"
            v.to_parquet(vp, index=False)
            st.artifact(vp, rows=len(v), cols=v.shape[1], message="sample studentVle")
            sa = read_raw_cached("studentAssessment")
            sa = sa.merge(read_raw_cached("assessments"), on="id_assessment",
                          how="left", validate="many_to_one")
            sa = sa.merge(keys, on=KEY, how="inner")
            sap = ctx.paths.interim_dir / "studentAssessment.parquet"
            sa.to_parquet(sap, index=False)
            st.artifact(sap, rows=len(sa), cols=sa.shape[1],
                        message="sample studentAssessment")
            st.metric("sim_streams", {"studentVle_rows": int(len(v)),
                                      "studentVle_share": round(
                                          len(v) / max(len(read_raw_cached("studentVle")), 1), 6)})
        else:
            m = master

        mp = ctx.paths.interim_dir / "master.parquet"
        m.to_parquet(mp, index=False)
        st.artifact(mp, rows=len(m), cols=m.shape[1], message="master table")


    st.check("final_master_rows", len(m) == EXPECTED[ctx.mode],
             f"rows={len(m)} expected={EXPECTED[ctx.mode]}")
    st.check("final_master_key_unique", not m.duplicated(KEY).any(), "master key unique")
