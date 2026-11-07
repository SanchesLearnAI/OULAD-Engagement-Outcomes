"""S02 — master table, cohort flags, Withdrawn audit, simulation sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cohort import (add_at_risk_flags, build_master, cohort_flow,
                        stratified_sample, withdrawal_audit)
from src.data import KEY, read_raw_cached

EXPECTED = {"simulation": 500, "full": 32593}


def run(ctx, st):
    cfg = ctx.cfg
    cutoffs = cfg["windows"]["cutoffs"]
    seed = cfg["seed"]

    with st.step("read_inputs"):
        si = read_raw_cached("studentInfo")
        sr = read_raw_cached("studentRegistration")
        courses = read_raw_cached("courses")
        assessments = read_raw_cached("assessments")
        sa = read_raw_cached("studentAssessment")
        vle = read_raw_cached("studentVle")
        st.metric("inputs", {"studentInfo": len(si), "studentRegistration": len(sr),
                             "assessments": len(assessments),
                             "studentAssessment": len(sa), "studentVle": len(vle)})

    with st.step("build_master"):
        positive_class = str(cfg["cohort"].get("positive_class", "fail"))
        master, info = build_master(si, sr, courses, assessments,
                                    positive_class=positive_class)
        st.check("positive_class_declared", positive_class in ("fail", "pass"),
                 f"T1 positive class = {positive_class}")
        master = add_at_risk_flags(master, cutoffs)
        st.metric("master", {
            "rows": int(len(master)),
            "join_loss": info["join_loss"],
            "courses_missing": info["courses_missing"],
            "n_withdrawn": info["n_withdrawn"],
            "n_repeaters": info["n_repeaters"],
            "first_tma_missing": info["first_tma_missing"],
            "n_t1": info["n_t1"],
            "final_result_counts": info["final_result_counts"],
        })
        st.check("master_key_unique", not master.duplicated(KEY).any(),
                 "one row per enrolment")
        st.check("final_result_no_missing", master["final_result"].notna().all(),
                 "final_result has no missing values")
        st.check("first_tma_missing_recorded", True,
                 f"{info['first_tma_missing']} enrolments have no TMA/CMA; "
                 "W2 falls back to day 55")

    with st.step("withdrawal_audit"):
        per_pres, weekly, audit = withdrawal_audit(master, vle)
        p1 = ctx.paths.tables_dir / "t02_withdrawal_audit.csv"
        per_pres.to_csv(p1, index=False)
        st.artifact(p1, rows=len(per_pres), cols=per_pres.shape[1],
                    message="withdrawal audit by presentation")
        p2 = ctx.paths.tables_dir / "t02_withdrawal_weekly.csv"
        weekly.to_csv(p2, index=False)
        st.artifact(p2, rows=len(weekly), cols=weekly.shape[1],
                    message="weekly clicks around unregistration")
        audit["unregistration_nonnull_share"] = float(
            master["date_unregistration"].notna().mean())
        at_risk_cfg = bool(cfg["cohort"].get("at_risk_at_cutoff", False))
        sens_cfg = bool(cfg["cohort"].get("at_risk_sensitivity", True))
        audit["at_risk_at_cutoff_config"] = at_risk_cfg
        audit["at_risk_sensitivity_config"] = sens_cfg
        audit["at_risk_filter_applied"] = bool(at_risk_cfg and
                                               not audit["post_clicks_exist"])
        audit["decision"] = (
            "main cohort keeps all T1 (at_risk_at_cutoff=false) and the at-risk "
            "cohort is a sensitivity only (at_risk_sensitivity=true). "
            "Post-unregistration clicks also rule out silent filtering. "
            "Withdrawn remains a separate outcome for T3/competing risk."
        )
        st.check("config_at_risk_consistent", (not at_risk_cfg) and sens_cfg,
                 f"at_risk_at_cutoff={at_risk_cfg}, at_risk_sensitivity={sens_cfg}")
        st.metric("withdrawal", audit)
        st.check("withdrawal_audit_generated", p1.exists() and p2.exists(),
                 "audit tables exist")
        st.check("unregistration_share_recorded", True,
                 f"{audit['unregistration_nonnull_share']:.4f} of enrolments "
                 "have a non-null date_unregistration")
        st.check("at_risk_decision_recorded", "at_risk_filter_applied" in audit,
                 audit["decision"])
        if audit["post_clicks_exist"]:
            # This is an empirical finding, not a pipeline error: keep the
            # stage green but make the decision explicit in checks/metrics.
            st.check(
                "post_unregistration_clicks_recorded", True,
                f"{audit['post_unregistration_click_rows']} click rows after "
                f"date_unregistration ({audit['students_with_post_click']} students); "
                "at-risk filtering withheld from the main cohort")

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        if len(weekly):
            before = weekly[weekly["rel_week"] < 0]
            after = weekly[weekly["rel_week"] >= 0]
            ax.bar(before["rel_week"], before["clicks"], color="#3b6ea5",
                   label="before unregistration")
            ax.bar(after["rel_week"], after["clicks"], color="#c05b5b",
                   label="after unregistration")
            ax.set_xlabel("weeks relative to date_unregistration")
            ax.set_ylabel("clicks (withdrawn students)")
            ax.axvline(-0.5, color="#333", lw=0.9, ls=":")
            ax.legend(frameon=False)
        ax.set_title("Withdrawn: clicks do not stop at date_unregistration")
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f02_withdrawal_clicks.png"
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        st.artifact(fp, message="withdrawal timing figure")

    with st.step("cohort_flow"):
        st.metric("cohort_counts_full", {
            "n_total": int(len(master)),
            "n_t1": int(master["y_t1"].notna().sum()),
            "n_t2": int(len(master)),
            "n_t3": {str(k): int(v) for k, v in master["y_t3"].value_counts().items()},
        })

    with st.step("write_master"):
        if ctx.mode == "simulation":
            sample = stratified_sample(master, EXPECTED["simulation"], seed)
            sp = ctx.paths.interim_dir / "sim_sample.parquet"
            sample.to_parquet(sp, index=False)
            st.artifact(sp, rows=len(sample), cols=sample.shape[1],
                        message="simulation sample keys")
            st.check("sample_size_500", len(sample) == 500,
                     f"sample={len(sample)}")
            st.check("sample_strata_represented",
                     sample.groupby(["code_module", "code_presentation",
                                     "final_result"]).ngroups ==
                     master.groupby(["code_module", "code_presentation",
                                     "final_result"]).ngroups,
                     "all 88 strata represented")
            keys = sample[KEY].drop_duplicates()
            m = master.merge(keys, on=KEY, how="inner").sort_values(KEY).reset_index(drop=True)
            st.check("sim_master_rows_500", len(m) == 500, f"rows={len(m)}")
            # Filter click / assessment streams once; later stages read only these.
            v = vle.merge(keys, on=KEY, how="inner")
            vp = ctx.paths.interim_dir / "studentVle.parquet"
            v.to_parquet(vp, index=False)
            st.artifact(vp, rows=len(v), cols=v.shape[1],
                        message="sample studentVle")
            st.metric("sim_streams", {
                "studentVle_rows": int(len(v)),
                "studentVle_share": round(len(v) / max(len(vle), 1), 5),
            })
            alloc = (sample.groupby(["code_module", "code_presentation", "final_result"])
                     .size().rename("alloc").reset_index())
            ap = ctx.paths.tables_dir / "t02_sample_allocation.csv"
            alloc.to_csv(ap, index=False)
            st.artifact(ap, rows=len(alloc), cols=alloc.shape[1],
                        message="sample allocation")
        else:
            m = master
            v = None

        mp = ctx.paths.interim_dir / "master.parquet"
        m.to_parquet(mp, index=False)
        st.artifact(mp, rows=len(m), cols=m.shape[1], message="master table")

        # Cohort flow must describe the rows later stages actually read.
        flow = cohort_flow(m, cutoffs)
        fp = ctx.paths.tables_dir / "t02_cohort_flow.csv"
        flow.to_csv(fp, index=False)
        st.artifact(fp, rows=len(flow), cols=flow.shape[1], message="cohort flow")
        st.metric("cohort_counts", {
            "n_total": int(len(m)),
            "n_t1": int(m["y_t1"].notna().sum()),
            "n_t2": int(len(m)),
            "n_t3": {str(k): int(v) for k, v in m["y_t3"].value_counts().items()},
        })

        # Enrich studentAssessment with assessment metadata; cache full version
        # for the full run, write the filtered version for simulation.
        a = read_raw_cached("assessments")
        sa = read_raw_cached("studentAssessment")
        sae = sa.merge(a, on="id_assessment", how="left", validate="many_to_one")
        cache = ctx.paths.root / "data" / "interim" / "cache" / "studentAssessment_enriched.parquet"
        if not cache.exists():
            sae.to_parquet(cache, index=False)
        st.artifact(cache, rows=len(sae), cols=sae.shape[1],
                    message="enriched studentAssessment cache")
        if ctx.mode == "simulation":
            keys = m[KEY].drop_duplicates()
            sae = sae.merge(keys, on=KEY, how="inner")
        sap = ctx.paths.interim_dir / "studentAssessment.parquet"
        sae.to_parquet(sap, index=False)
        st.artifact(sap, rows=len(sae), cols=sae.shape[1],
                    message="studentAssessment with presentation metadata")
        st.metric("assessment_stream", {"rows": int(len(sae))})

    # Final gate on the rows actually written.
    st.check("final_master_rows", len(m) == EXPECTED[ctx.mode],
             f"rows={len(m)} expected={EXPECTED[ctx.mode]}",
             metrics={"rows": int(len(m))})
    st.check("final_master_key_unique", not m.duplicated(KEY).any(),
             "master key unique")
    st.check("unregistration_share_not_presupposed",
             audit["unregistration_nonnull_share"] < 1.0,
             f"{audit['unregistration_nonnull_share']:.4f}")
    st.extra["at_risk_filter_applied"] = audit["at_risk_filter_applied"]
    st.extra["withdrawal_audit"] = audit
