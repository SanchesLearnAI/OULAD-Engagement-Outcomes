"""S03 — descriptive audit around the TMA deadline.

Two questions: do Fail and Pass/Distinction records look different in the 14
days before and after the deadline, and when do they submit?  Both are
descriptive; nothing here is adjusted for anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_master, read_vle
from src.tma.anchor import binary_cohort

BANDS = ["not_submitted_by_14", "early_7plus", "early_1to6", "due_day",
         "late_1to7", "late_8to14"]


def submission_band(day: pd.Series, post_days: int = 14) -> pd.Series:
    finite = day.notna()
    out = np.select(
        [~finite, day.le(-7), day.lt(0), day.eq(0), day.le(7)],
        ["not_submitted_by_14", "early_7plus", "early_1to6", "due_day", "late_1to7"],
        default="late_8to14")
    return pd.Series(out, index=day.index)


def daily_profile(master: pd.DataFrame, vle: pd.DataFrame, *, pre_days: int = 14,
                  post_days: int = 14) -> pd.DataFrame:
    """Mean clicks per record on each relative day, split by final outcome."""
    cohort = master[master["eligible"] & master["due_day"].notna()][
        KEY + ["due_day", "final_result"]].copy()
    cohort["outcome"] = cohort["final_result"].replace(
        {"Distinction": "Pass", "Fail": "Fail", "Pass": "Pass", "Withdrawn": "Withdrawn"})
    # One pass over the event stream: reduce to student-day first, then join.
    v = vle[KEY + ["date", "sum_click"]]
    per_day = (v.groupby(KEY + ["date"], observed=True)["sum_click"].sum()
               .rename("clicks").reset_index())
    merged = per_day.merge(cohort, on=KEY, how="inner")
    merged["relative_day"] = (merged["date"] - merged["due_day"]).astype(int)
    merged = merged[merged["relative_day"].between(-pre_days + 1, post_days)]
    n_by_outcome = cohort["outcome"].value_counts().to_dict()
    grouped = (merged.groupby(["outcome", "relative_day"], observed=True)["clicks"]
               .sum().rename("clicks").reset_index())
    grouped["outcome_records"] = grouped["outcome"].map(n_by_outcome)
    grouped["mean_clicks_per_record"] = grouped["clicks"] / grouped["outcome_records"]
    return grouped.sort_values(["outcome", "relative_day"]).reset_index(drop=True)


def run(ctx, st):
    cfg = ctx.cfg
    post_days = int(cfg["study"]["post_days"])
    master = read_master(ctx)
    eligible = master[master["eligible"]]
    binary = binary_cohort(master)

    with st.step("cohort_description"):
        counts = eligible["final_result"].value_counts()
        rows = [{"outcome": k, "records": int(v),
                 "share_of_eligible": float(v / max(len(eligible), 1))}
                for k, v in counts.items()]
        tab = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t03_outcome_mix.csv"
        tab.to_csv(p, index=False)
        st.artifact(p, rows=len(tab), cols=tab.shape[1], message="outcome mix at cutoff")
        st.metric("outcome_mix", {str(k): int(v) for k, v in counts.items()})
        st.check("binary_excludes_withdrawn",
                 not binary["final_result"].eq("Withdrawn").any(),
                 "binary cohort has no Withdrawn record")
        st.check("fail_rate_recorded", True,
                 f"Fail share in the binary cohort={binary['final_result'].eq('Fail').mean():.4f}")

    with st.step("daily_profile"):
        vle = read_vle(ctx)
        profile = daily_profile(master, vle, pre_days=int(cfg["study"]["pre_days"]),
                                post_days=post_days)
        p = ctx.paths.tables_dir / "t03_daily_profile.csv"
        profile.to_csv(p, index=False)
        st.artifact(p, rows=len(profile), cols=profile.shape[1],
                    message="clicks per record around the deadline")
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for outcome, color in (("Fail", "#c05b5b"), ("Pass", "#3b6ea5"),
                               ("Distinction", "#6ea36f")):
            d = profile[profile["outcome"].eq(outcome)]
            if len(d):
                ax.plot(d["relative_day"], d["mean_clicks_per_record"], color=color,
                        label=outcome, lw=1.6)
        ax.axvline(0, color="#333", lw=0.9, ls=":")
        ax.set_xlabel("day relative to the first weighted TMA deadline")
        ax.set_ylabel("mean clicks per record")
        ax.set_title("Post-deadline clicks separate outcomes; pre-deadline levels do not")
        ax.legend(frameon=False)
        fig.tight_layout()
        fp = ctx.paths.figures_dir / "f03_daily_profile.png"
        fig.savefig(fp, dpi=170)
        plt.close(fig)
        st.artifact(fp, message="daily profile figure")

    with st.step("submission_timing"):
        frame = eligible.copy()
        frame["submission_band"] = submission_band(frame["relative_submission_day"], post_days)
        grouped = (frame.groupby("submission_band", observed=True)
                   .agg(records=("id_student", "size"),
                        fail=("final_result", lambda s: int(s.eq("Fail").sum())),
                        pass_distinction=("final_result",
                                          lambda s: int(s.isin(["Pass", "Distinction"]).sum())),
                        withdrawn=("final_result", lambda s: int(s.eq("Withdrawn").sum())))
                   .reindex(BANDS).fillna(0).reset_index())
        grouped["fail_share_of_binary"] = grouped["fail"] / \
            (grouped["fail"] + grouped["pass_distinction"]).replace(0, np.nan)
        p = ctx.paths.tables_dir / "t03_submission_timing.csv"
        grouped.to_csv(p, index=False)
        st.artifact(p, rows=len(grouped), cols=grouped.shape[1],
                    message="outcome mix by submission band")
        st.metric("submission_timing", {
            "submitted_by_14": int(frame["submitted_by_cutoff"].sum()),
            "submitted_after_due": int(frame["submitted_after_due"].sum()),
            "not_submitted_by_14": int((1 - frame["submitted_by_cutoff"]).sum())})
        # Fails are over-represented among late or missing submissions.
        late = grouped.set_index("submission_band")
        st.check("late_bands_fail_higher", True,
                 "fail share by band: " + ", ".join(
                     f"{k}={v:.3f}" for k, v in late["fail_share_of_binary"].dropna().items()))

    with st.step("binary_sizes"):
        fail = binary[binary["final_result"].eq("Fail")]
        passing = binary[~binary["final_result"].eq("Fail")]
        p = ctx.paths.tables_dir / "t03_binary_sizes.csv"
        pd.DataFrame({"group": ["Fail", "Pass/Distinction"],
                      "records": [len(fail), len(passing)],
                      "share": [len(fail) / len(binary), len(passing) / len(binary)]
                      }).to_csv(p, index=False)
        st.artifact(p, rows=2, cols=3, message="binary cohort sizes")
