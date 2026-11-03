"""Descriptive timing coverage and concentration; zero activity is separate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .features import CONCENTRATION_FEATURES


def describe_timing(frame: pd.DataFrame, output_dir: Path) -> None:
    cohorts = {"all": frame,
               "eligible_day28": frame[frame.eligible_at_day_28 == 1],
               "binary": frame[frame.final_result.isin(["Pass", "Distinction", "Fail"])]}
    rows = []
    quantiles = []
    for name, cohort in cohorts.items():
        active = cohort[cohort.total_clicks > 0]
        rows.append({"cohort": name, "rows": len(cohort), "zero_click_rows": int((cohort.total_clicks == 0).sum()),
                     "nonzero_rows": len(active),
                     "all_four_weeks_active": int((active.active_weeks == 4).sum()),
                     "one_calendar_week_at_least_half": int((active.max_week_share >= .5).sum()),
                     "one_rolling_week_at_least_half": int((active.peak_7day_share >= .5).sum()),
                     "weekly_evenness_at_least_0_9": int((active.weekly_evenness >= .9).sum())})
        for feature in CONCENTRATION_FEATURES + ["active_days", "daily_concentration"]:
            quantiles.append({"cohort": name, "feature": feature,
                              **{f"q{int(q*100)}": float(active[feature].quantile(q)) for q in (.25, .5, .75)}})
    pd.DataFrame(rows).to_csv(output_dir / "timing_summary.csv", index=False)
    pd.DataFrame(quantiles).to_csv(output_dir / "timing_quantiles.csv", index=False)
    active = cohorts["binary"].query("total_clicks > 0")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = active.active_weeks.value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0) * 100
    axes[0].bar(counts.index, counts.values)
    axes[0].set(xlabel="Weeks with positive clicks", ylabel="Student-course records (%)", xticks=[1,2,3,4])
    for x, y in counts.items():
        axes[0].text(x, y + .7, f"{y:.1f}%", ha="center")
    axes[0].set_ylim(0, max(10, counts.max()*1.15))
    axes[1].hist(active.peak_7day_share, bins=[.25 + .05*i for i in range(16)],
                 weights=[100 / len(active)] * len(active) if len(active) else None)
    axes[1].set(xlabel="Largest share within any consecutive 7 days", ylabel="Student-course records (%)", xlim=(.25,1))
    fig.suptitle(f"Days 0–27: nonzero-click Pass/Distinction/Fail records (n={len(active):,})")
    fig.tight_layout()
    fig.savefig(output_dir / "timing_distribution.png", dpi=160)
    plt.close(fig)
