"""S01 — raw-table audit: keys, dates, labels, activity types, missingness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import project_root

TABLES = ["courses", "assessments", "vle", "studentInfo",
          "studentRegistration", "studentAssessment", "studentVle"]


def run(ctx, st):
    root = project_root()
    raw = ctx.paths.raw_dir
    cache = root / "data" / "interim" / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    tables = {}
    with st.step("read_tables"):
        for name in TABLES:
            df = pd.read_csv(raw / f"{name}.csv", na_values=["?"])
            tables[name] = df
            st.metric(f"shape_{name}", {"rows": int(len(df)), "cols": int(df.shape[1])},
                      f"{name} shape")
        # Cached parquet copies keep later stages from re-parsing 10.6M CSV rows.
        for name, df in tables.items():
            p = cache / f"{name}.parquet"
            if not p.exists():
                df.to_parquet(p, index=False)
            st.artifact(p, rows=len(df), cols=df.shape[1], message=f"cache {name}")

    with st.step("schema_audit"):
        rows = []
        for name, df in tables.items():
            for col in df.columns:
                s = df[col]
                rows.append({
                    "table": name,
                    "rows": len(df),
                    "cols": df.shape[1],
                    "column": col,
                    "dtype": str(s.dtype),
                    "missing": int(s.isna().sum()),
                    "missing_share": float(s.isna().mean()),
                    "n_unique": int(s.nunique(dropna=True)),
                })
        audit = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t01_raw_audit.csv"
        audit.to_csv(p, index=False)
        st.artifact(p, rows=len(audit), cols=audit.shape[1], message="raw audit")
        st.metric("missing", {
            "imd_band_missing": int(tables["studentInfo"]["imd_band"].isna().sum()),
            "date_registration_missing": int(tables["studentRegistration"]["date_registration"].isna().sum()),
            "date_unregistration_missing": int(tables["studentRegistration"]["date_unregistration"].isna().sum()),
            "score_missing": int(tables["studentAssessment"]["score"].isna().sum()),
        })

    with st.step("key_checks"):
        si = tables["studentInfo"]
        key = ["code_module", "code_presentation", "id_student"]
        dup = int(si.duplicated(key).sum())
        st.check("studentinfo_key_unique", dup == 0, f"duplicates={dup}",
                 metrics={"duplicates": dup})
        st.metric("students", {
            "n_unique_students": int(si["id_student"].nunique()),
            "n_enrolments": int(len(si)),
            "repeat_students": int((si.groupby("id_student").size() > 1).sum()),
        })
        a = tables["assessments"]
        st.check("assessment_id_unique", int(a["id_assessment"].duplicated().sum()) == 0,
                 "id_assessment must be unique for studentAssessment joins")
        v = tables["vle"]
        st.metric("vle_catalog", {
            "n_sites": int(len(v)),
            "n_activity_types": int(v["activity_type"].nunique()),
            "dup_site_keys": int(v.duplicated(["code_module", "code_presentation",
                                              "id_site"]).sum()),
        })

    with st.step("labels"):
        counts = si["final_result"].value_counts(dropna=False)
        p = ctx.paths.tables_dir / "t01_final_result_counts.csv"
        counts.rename_axis("final_result").reset_index(name="n").to_csv(p, index=False)
        st.artifact(p, rows=len(counts), cols=2, message="final_result counts")
        st.metric("final_result_counts", {str(k): int(v) for k, v in counts.items()})
        expected = {"Pass", "Fail", "Withdrawn", "Distinction"}
        st.check("final_result_exact_4", set(si["final_result"].dropna()) == expected,
                 f"classes={sorted(si['final_result'].dropna().unique())}")

    with st.step("dates"):
        sv = tables["studentVle"]
        dmin, dmax = int(sv["date"].min()), int(sv["date"].max())
        neg = sv["date"] < 0
        n_neg = int(neg.sum())
        share = float(neg.mean())
        st.metric("studentVle_dates", {
            "date_min": dmin, "date_max": dmax,
            "negative_rows": n_neg, "negative_share": round(share, 6),
            "negative_students": int(sv.loc[neg, "id_student"].nunique()),
        })
        st.check("negative_dates_present", dmin < 0,
                 f"date_min={dmin}; negative rows={n_neg} ({share:.4%})")
        st.metric("date_range", {"min": dmin, "max": dmax})

    with st.step("activity_and_banked"):
        at = tables["vle"]["activity_type"].value_counts()
        p = ctx.paths.tables_dir / "t01_activity_types.csv"
        at.rename_axis("activity_type").reset_index(name="n_sites").to_csv(p, index=False)
        st.artifact(p, rows=len(at), cols=2, message="activity types")
        banked = tables["studentAssessment"]["is_banked"].value_counts(dropna=False)
        st.metric("banked", {str(k): int(v) for k, v in banked.items()})

    with st.step("click_distribution"):
        sv = tables["studentVle"]
        per = sv.groupby(["code_module", "code_presentation", "id_student"],
                         observed=True)["sum_click"].sum()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        vals = np.log1p(per.values)
        ax.hist(vals, bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.3)
        ax.axvline(float(np.median(vals)), color="#b23b3b", ls="--", lw=1.2,
                   label=f"median={np.expm1(np.median(vals)):.0f} clicks")
        ax.set_xlabel("log1p(total clicks per enrolment)")
        ax.set_ylabel("enrolments")
        ax.set_title("OULAD per-enrolment click volume (all dates)")
        ax.legend(frameon=False)
        fig.tight_layout()
        p = ctx.paths.figures_dir / "f01_click_distribution.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        st.artifact(p, message="click distribution figure")
        st.metric("clicks", {
            "median_total": float(per.median()),
            "p99_total": float(per.quantile(0.99)),
            "zero_click_enrolments": int((per == 0).sum()),
        })
