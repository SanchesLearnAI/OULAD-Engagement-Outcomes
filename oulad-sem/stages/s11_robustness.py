"""S11 — robustness: windows, tolerances, duplicate events and a spec curve.

Every row is a sensitivity analysis of the main design, not a new main result.
The +21/+28 windows are reported twice: with post-click matching (comparable to
the main design) and with whole-window click matching (comparable to the earlier
+21/+28 study).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import KEY, read_master, read_vle
from src.tma import daily, effects, pairing
from src.tma.anchor import binary_cohort

SPECS = [
    ("primary", {"post_days": 14, "tolerance": 0.05, "match_on": "post_clicks",
                 "dedupe": False}),
    ("tolerance_0", {"post_days": 14, "tolerance": 0.00, "match_on": "post_clicks",
                     "dedupe": False}),
    ("tolerance_10", {"post_days": 14, "tolerance": 0.10, "match_on": "post_clicks",
                      "dedupe": False}),
    ("window_21_post", {"post_days": 21, "tolerance": 0.05, "match_on": "post_clicks",
                        "dedupe": False}),
    ("window_28_post", {"post_days": 28, "tolerance": 0.05, "match_on": "post_clicks",
                        "dedupe": False}),
    ("window_21_total", {"post_days": 21, "tolerance": 0.05, "match_on": "total_clicks",
                         "dedupe": False}),
    ("window_28_total", {"post_days": 28, "tolerance": 0.05, "match_on": "total_clicks",
                         "dedupe": False}),
    ("dedupe_events", {"post_days": 14, "tolerance": 0.05, "match_on": "post_clicks",
                       "dedupe": True}),
]
METRIC_SUMMARY = ["submitted_by_cutoff", "submitted_after_due", "post_active_days",
                  "post_longest_inactive_run", "post_week2_share"]


def _features(vle: pd.DataFrame, cohort: pd.DataFrame, post_days: int,
              dedupe: bool) -> pd.DataFrame:
    source = vle
    if dedupe:
        source = vle.drop_duplicates(subset=KEY + ["id_site", "date", "sum_click"])
    lookup = cohort[KEY + ["due_day"]]
    matrix = daily.daily_events(source, lookup, pre_days=14, post_days=post_days)
    frame, _, _ = daily.build_daily_features(cohort, matrix, None, pre_days=14,
                                             post_days=post_days)
    frame["total_clicks"] = frame["pre_clicks"] + frame["post_clicks"]
    return frame


def run(ctx, st):
    cfg = ctx.cfg
    seed = cfg["seed"]
    repeats = int(cfg["clustering"].get("max_pairs", 800))
    master = read_master(ctx)
    cohort = binary_cohort(master)
    vle = read_vle(ctx)

    with st.step("spec_curve"):
        cache = {}
        rows, effect_rows = [], []
        for name, spec in SPECS:
            key = (spec["post_days"], spec["dedupe"])
            if key not in cache:
                cache[key] = _features(vle, cohort, spec["post_days"], spec["dedupe"])
            frame = cache[key]
            column = spec["match_on"]
            pairs = pairing.make_pairs(frame, spec["tolerance"], seed, column)
            if pairs.empty:
                continue
            balance = pairing.balance_row(frame, pairs, spec["tolerance"], column)
            eff = effects.effect_table(frame, pairs, METRIC_SUMMARY, seed, repeats)
            for _, r in eff.iterrows():
                effect_rows.append({"spec": name, "post_days": spec["post_days"],
                                    "tolerance": spec["tolerance"],
                                    "match_on": column, "dedupe": spec["dedupe"],
                                    **r.to_dict()})
            rows.append({"spec": name, "post_days": spec["post_days"],
                         "tolerance": spec["tolerance"], "match_on": column,
                         "dedupe": spec["dedupe"], "pairs": balance["pairs"],
                         "fail_coverage": balance["fail_coverage"],
                         "pre_log_click_smd": balance["pre_log_click_smd"],
                         "post_log_click_smd": balance["post_log_click_smd"]})
        table = pd.DataFrame(rows)
        p = ctx.paths.tables_dir / "t11_specs.csv"
        table.to_csv(p, index=False)
        st.artifact(p, rows=len(table), cols=table.shape[1], message="specification curve")
        st.metric("specs", {"specifications": int(len(table)),
                            "pairs_min": int(table["pairs"].min()) if len(table) else 0,
                            "pairs_max": int(table["pairs"].max()) if len(table) else 0})
        st.check("primary_spec_present", "primary" in set(table["spec"]),
                 "the pre-specified primary specification is in the curve")

        effects_table = pd.DataFrame(effect_rows)
        p = ctx.paths.tables_dir / "t11_spec_effects.csv"
        effects_table.to_csv(p, index=False)
        st.artifact(p, rows=len(effects_table),
                    cols=effects_table.shape[1] if len(effects_table) else 0,
                    message="paired effects across specifications")

    with st.step("window_sensitivity"):
        window = table[table["spec"].str.startswith("window")].copy()
        p = ctx.paths.tables_dir / "t11_window_sensitivity.csv"
        window.to_csv(p, index=False)
        st.artifact(p, rows=len(window), cols=window.shape[1] if len(window) else 0,
                    message="+21/+28 window versions")
        st.check("window_versions_labelled",
                 set(window["match_on"]) <= {"post_clicks", "total_clicks"},
                 "each window is labelled by its matching variable")

    with st.step("negative_control"):
        # Persist the spec curve's sign stability: if a direction flips across
        # specifications, the report must say so instead of quoting one number.
        eff = pd.DataFrame(effect_rows)
        if len(eff):
            summary = (eff.assign(negative=eff["fail_minus_pass"] < 0)
                       .groupby("feature")["negative"]
                       .agg(["sum", "count"]).reset_index()
                       .rename(columns={"sum": "specs_negative", "count": "specs"}))
            summary["share_negative"] = summary["specs_negative"] / summary["specs"]
            p = ctx.paths.tables_dir / "t11_direction_stability.csv"
            summary.to_csv(p, index=False)
            st.artifact(p, rows=len(summary), cols=summary.shape[1],
                        message="sign stability across specifications")
            for feature in ("post_active_days", "post_longest_inactive_run"):
                row = summary[summary["feature"].eq(feature)]
                if len(row):
                    st.metric(f"direction_{feature}",
                              {"share_negative": float(row["share_negative"].iloc[0]),
                               "specs": int(row["specs"].iloc[0])})

    with st.step("figure"):
        eff = pd.DataFrame(effect_rows)
        if len(eff):
            pivot = eff.pivot_table(index="spec", columns="feature",
                                    values="fail_minus_pass", aggfunc="first")
            fig, ax = plt.subplots(figsize=(7.4, 3.8))
            order = pivot.index.tolist()
            x = np.arange(len(order))
            width = 0.8 / max(len(METRIC_SUMMARY), 1)
            for i, feature in enumerate([f for f in METRIC_SUMMARY if f in pivot.columns]):
                ax.bar(x + (i - 0.5) * width, pivot[feature], width, label=feature)
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(x, order, rotation=20, ha="right")
            ax.set_ylabel("Fail minus Pass/Distinction")
            ax.set_title("Primary directions across robustness specifications")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fp = ctx.paths.figures_dir / "f11_spec_curve.png"
            fig.savefig(fp, dpi=170)
            plt.close(fig)
            st.artifact(fp, message="specification curve figure")
