"""Feature construction with explicit cutoff isolation.

Only studentVle.date < cutoff and studentAssessment.date_submitted < cutoff are
ever read.  Scores and is_banked are excluded from strict features.
"""

# Legacy module kept for the archived day-0-27 design (see
# results/oulad_sem_day0_27_baseline_20261108/). The TMA pipeline in
# stages/ does not import it; it reads config keys that the frozen
# config no longer carries, so run it only against the archived config.
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY, read_raw_cached
from .windows import bounds

QUIZ_TYPES = {"quiz", "externalquiz"}
TOP_ACTIVITY_TYPES = ["resource", "subpage", "oucontent", "url", "forumng",
                      "homepage", "page", "questionnaire", "oucollaborate",
                      "ouwiki", "dataplus", "glossary"]

WINDOW_BASE = [
    "clicks_total", "clicks_log1p", "active_days", "n_sites",
    "n_activity_types", "clicks_per_week", "clicks_per_active_day",
    "max_week_clicks", "weekly_std", "active_weeks", "week_slope",
    "daily_cv", "max_gap", "longest_streak", "day_phase_entropy",
    "last_7d_share", "clicks_last_48h", "days_since_last_click",
    "first_click_day", "entropy_activity", "gini_activity",
] + [f"share_{t}" for t in TOP_ACTIVITY_TYPES] + ["share_other"]
SUBMISSION_BASE = [
    "submissions", "on_time_share", "late_submissions", "mean_lead_days",
    "days_since_last_submission", "first_submission_day",
]


def _ensure_window_columns(out: pd.DataFrame, prefix: str,
                           top_types: list[str]) -> pd.DataFrame:
    names = WINDOW_BASE + [f"share_{t}" for t in top_types] + ["share_other"]
    names = list(dict.fromkeys(names)) + SUBMISSION_BASE
    for n in names:
        c = f"{prefix}_{n}"
        if c not in out.columns:
            out[c] = 0.0
    return out


def prepare_vle(ctx, include_quiz_sites: bool | None = None) -> tuple[pd.DataFrame, dict]:
    vle = pd.read_parquet(
        (ctx.paths.interim_dir / "studentVle.parquet") if ctx.mode == "simulation"
        else ctx.paths.root / "data" / "interim" / "cache" / "studentVle.parquet"
    )
    cat = read_raw_cached("vle")[["code_module", "code_presentation", "id_site",
                                  "activity_type"]]
    v = vle.merge(cat, on=["code_module", "code_presentation", "id_site"],
                  how="left", validate="many_to_one")
    info = {"rows_in": int(len(v)),
            "missing_activity_type": int(v["activity_type"].isna().sum())}
    v["activity_type"] = v["activity_type"].fillna("unknown")
    n_before = len(v)
    if include_quiz_sites is None:
        include_quiz_sites = bool(ctx.cfg["leakage"].get("include_quiz_sites", False))
    info["include_quiz_sites"] = bool(include_quiz_sites)
    if not include_quiz_sites:
        v = v[~v["activity_type"].isin(QUIZ_TYPES)]
        info["quiz_rows_excluded"] = int(n_before - len(v))
        info["quiz_row_share"] = float((n_before - len(v)) / max(n_before, 1))
    else:
        info["quiz_rows_excluded"] = 0
    info["rows_after"] = int(len(v))
    info["date_min"] = int(v["date"].min())
    info["date_max"] = int(v["date"].max())
    info["negative_rows"] = int((v["date"] < 0).sum())
    info["negative_share"] = float((v["date"] < 0).mean())
    return v, info


def prepare_assessment(ctx) -> pd.DataFrame:
    if ctx.mode == "simulation":
        p = ctx.paths.interim_dir / "studentAssessment.parquet"
    else:
        p = ctx.paths.root / "data" / "interim" / "cache" / "studentAssessment_enriched.parquet"
    return pd.read_parquet(p)


def _safe_div(a, b, fill=0.0):
    return np.where(np.asarray(b) > 0, np.asarray(a) / np.maximum(np.asarray(b), 1), fill)


def _window_features(df: pd.DataFrame, prefix: str, start: int, end: int,
                     total_days: int, top_types: list[str]) -> pd.DataFrame:
    """df has KEY, date, activity_type, id_site, sum_click."""
    key = KEY
    if len(df) == 0:
        return pd.DataFrame(columns=key)

    tot = df.groupby(key, observed=True)["sum_click"].sum().rename("clicks_total")
    out = tot.to_frame()
    out["clicks_log1p"] = np.log1p(out["clicks_total"])
    out["active_days"] = df.groupby(key, observed=True)["date"].nunique()
    out["n_sites"] = df.groupby(key, observed=True)["id_site"].nunique()
    out["n_activity_types"] = df.groupby(key, observed=True)["activity_type"].nunique()
    weeks = max(total_days / 7.0, 1e-9)
    out["clicks_per_week"] = out["clicks_total"] / weeks
    out["clicks_per_active_day"] = _safe_div(out["clicks_total"], out["active_days"])
    out["max_week_clicks"] = 0.0
    out["weekly_std"] = 0.0
    out["active_weeks"] = 0
    out["week_slope"] = 0.0
    out["daily_cv"] = 0.0
    out["max_gap"] = float(total_days)
    out["longest_streak"] = 0
    out["day_phase_entropy"] = 0.0
    out["last_7d_share"] = 0.0
    out["clicks_last_48h"] = 0.0
    out["days_since_last_click"] = float(total_days + 1)
    out["first_click_day"] = np.nan

    # daily series: regularity
    d = (df.groupby(key + ["date"], observed=True)["sum_click"].sum()
         .reset_index().sort_values(key + ["date"]))
    g = d.groupby(key, observed=True)
    daily_mean = g["sum_click"].mean()
    daily_std = g["sum_click"].std(ddof=0)
    out.loc[daily_mean.index, "daily_cv"] = (
        daily_std.reindex(daily_mean.index).fillna(0) /
        daily_mean.where(daily_mean > 0, 1)
    ).to_numpy()
    d["gap"] = g["date"].diff()
    d["newrun"] = d["gap"].isna() | (d["gap"] != 1)
    d["run"] = d.groupby(key, observed=True)["newrun"].cumsum()
    mg = d.groupby(key, observed=True)["gap"].max() - 1
    out.loc[mg.index, "max_gap"] = mg.clip(lower=0).to_numpy()
    streak = d.groupby(key + ["run"], observed=True).size().groupby(level=key).max()
    out.loc[streak.index, "longest_streak"] = streak.to_numpy()
    out.loc[g["date"].min().index, "first_click_day"] = g["date"].min().to_numpy()
    out.loc[g["date"].max().index, "days_since_last_click"] = (
        end - g["date"].max()).to_numpy()

    # weekly series: slope / acceleration / max
    d["week"] = (d["date"] - start) // 7
    w = d.groupby(key + ["week"], observed=True)["sum_click"].sum().reset_index()
    wg = w.groupby(key, observed=True)
    n = wg["week"].size()
    sx = wg["week"].sum(); sy = wg["sum_click"].sum()
    sxx = wg.apply(lambda x: (x["week"] ** 2).sum(), include_groups=False)
    sxy = wg.apply(lambda x: (x["week"] * x["sum_click"]).sum(), include_groups=False)
    denom = sxx - sx * sx / n
    slope = np.where(denom.abs() > 1e-12, (sxy - sx * sy / n) / denom, 0.0)
    out.loc[n.index, "week_slope"] = slope
    out.loc[n.index, "active_weeks"] = n.to_numpy()
    out.loc[n.index, "max_week_clicks"] = wg["sum_click"].max().to_numpy()
    out.loc[n.index, "weekly_std"] = wg["sum_click"].std(ddof=0).fillna(0).to_numpy()

    # day-phase concentration (weekday is not available in OULAD)
    dp = df.copy()
    dp["phase"] = (dp["date"] % 7)
    phase = dp.groupby(key + ["phase"], observed=True)["sum_click"].sum()
    phase = phase / phase.groupby(level=key).transform("sum")
    ent = -(phase * np.log(phase.where(phase > 0, 1))).groupby(level=key).sum() / np.log(7)
    out.loc[ent.index, "day_phase_entropy"] = ent.to_numpy()

    # last-window intensity
    last48 = df[df["date"] >= end - 1].groupby(key, observed=True)["sum_click"].sum()
    last7 = df[df["date"] >= end - 6].groupby(key, observed=True)["sum_click"].sum()
    out.loc[last48.index, "clicks_last_48h"] = last48.to_numpy()
    out["last_7d_share"] = _safe_div(
        last7.reindex(out.index).fillna(0).to_numpy(), out["clicks_total"].to_numpy())

    # activity composition
    act = (df.groupby(key + ["activity_type"], observed=True)["sum_click"].sum()
           .unstack(fill_value=0))
    total = act.sum(axis=1)
    p = act.div(total.where(total > 0, 1), axis=0)
    nz = (act > 0).sum(axis=1).clip(lower=1)
    ent = -(p * np.log(p.where(p > 0, 1))).sum(axis=1) / np.log(nz)
    gini = 1 - (p ** 2).sum(axis=1)
    out.loc[ent.index, "entropy_activity"] = ent.to_numpy()
    out.loc[gini.index, "gini_activity"] = gini.to_numpy()
    top = [t for t in top_types if t in act.columns]
    for t in top:
        out.loc[act.index, f"share_{t}"] = (act[t] / total.where(total > 0, 1)).to_numpy()
    if top:
        rest = total - act[top].sum(axis=1)
        out.loc[act.index, "share_other"] = (rest / total.where(total > 0, 1)).to_numpy()
    else:
        out["share_other"] = 0.0

    out = out.fillna({
        "first_click_day": -1.0, "share_other": 0.0,
    })
    out = out.add_prefix(prefix + "_")
    return out


def _submission_features(sa: pd.DataFrame, master_keys: pd.DataFrame, start: int,
                         end: int, prefix: str, exclude_banked: bool) -> pd.DataFrame:
    key = KEY
    out = master_keys.copy().set_index(key)
    if len(sa) == 0:
        out[f"{prefix}_submissions"] = 0
        out[f"{prefix}_on_time_share"] = 0.0
        out[f"{prefix}_late_submissions"] = 0
        out[f"{prefix}_mean_lead_days"] = 0.0
        out[f"{prefix}_days_since_last_submission"] = float(end + 1)
        out[f"{prefix}_first_submission_day"] = -1.0
        return out.reset_index()[[*KEY] + [c for c in out.reset_index().columns if c.startswith(prefix + "_")]]
    s = sa.copy()
    if exclude_banked:
        s = s[s["is_banked"] == 0]
    s = s[(s["date_submitted"] >= start) & (s["date_submitted"] <= end)]
    if len(s) == 0:
        out[f"{prefix}_submissions"] = 0
        out[f"{prefix}_on_time_share"] = 0.0
        out[f"{prefix}_late_submissions"] = 0
        out[f"{prefix}_mean_lead_days"] = 0.0
        out[f"{prefix}_days_since_last_submission"] = float(end + 1)
        out[f"{prefix}_first_submission_day"] = -1.0
        return out.reset_index()[[*KEY] + [c for c in out.reset_index().columns if c.startswith(prefix + "_")]]
    s["on_time"] = (s["date_submitted"] <= s["date"]).astype(float)
    s["late"] = (s["date_submitted"] > s["date"]).astype(float)
    s["lead"] = s["date"] - s["date_submitted"]
    g = s.groupby(key, observed=True)
    f = pd.DataFrame({
        f"{prefix}_submissions": g.size(),
        f"{prefix}_on_time_share": g["on_time"].mean(),
        f"{prefix}_late_submissions": g["late"].sum(),
        f"{prefix}_mean_lead_days": g["lead"].mean(),
        f"{prefix}_days_since_last_submission": end - g["date_submitted"].max(),
        f"{prefix}_first_submission_day": g["date_submitted"].min(),
    })
    out = out.join(f, how="left")
    out[f"{prefix}_submissions"] = out[f"{prefix}_submissions"].fillna(0).astype(int)
    out[f"{prefix}_on_time_share"] = out[f"{prefix}_on_time_share"].fillna(0.0)
    out[f"{prefix}_late_submissions"] = out[f"{prefix}_late_submissions"].fillna(0)
    out[f"{prefix}_mean_lead_days"] = out[f"{prefix}_mean_lead_days"].fillna(0.0)
    out[f"{prefix}_days_since_last_submission"] = \
        out[f"{prefix}_days_since_last_submission"].fillna(end + 1)
    out[f"{prefix}_first_submission_day"] = \
        out[f"{prefix}_first_submission_day"].fillna(-1.0)
    return out.reset_index()[[*KEY] + [c for c in out.reset_index().columns if c.startswith(prefix + "_")]]


def build_features(master: pd.DataFrame, vle: pd.DataFrame, sa: pd.DataFrame,
                   cutoff: int, config: dict, pre_start: str | None = None,
                   top_types: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per (module, presentation, student); only dates < cutoff are read."""
    pre = pre_start or config["windows"]["pre_start"]
    top_types = top_types or TOP_ACTIVITY_TYPES
    keys = master[KEY].drop_duplicates().copy()
    info = {"cutoff": int(cutoff), "pre_start": pre,
            "n_negative_date_rows": int((vle["date"] < 0).sum()),
            "negative_date_share": float((vle["date"] < 0).mean())}
    v = vle[vle["date"] < cutoff].copy()
    if pre == "exclude":
        v = v[v["date"] >= 0]
    elif pre == "clip":
        v.loc[v["date"] < 0, "date"] = 0
    elif pre == "feature":
        # pre-start clicks are a separate sensitivity feature, never in windows
        pre_rows = v[v["date"] < 0]
        v = v[v["date"] >= 0]
        if len(pre_rows):
            pf = pre_rows.groupby(KEY, observed=True).agg(
                pre_start_clicks=("sum_click", "sum"),
                pre_start_active_days=("date", "nunique"))
            keys = keys.merge(pf.reset_index(), on=KEY, how="left")
            keys["pre_start_clicks"] = keys["pre_start_clicks"].fillna(0)
            keys["pre_start_active_days"] = keys["pre_start_active_days"].fillna(0)
    else:
        raise ValueError(f"unknown pre_start policy {pre}")

    # W1 is common to every presentation.  W2 end is presentation-specific.
    w1 = (0, min(int(config["windows"]["w1_end"]), cutoff - 1))
    max_date_seen = int(v["date"].max()) if len(v) else -1
    frames = []
    sub = v[(v["date"] >= w1[0]) & (v["date"] <= w1[1])]
    max_date_seen = max(max_date_seen, int(sub["date"].max()) if len(sub) else -1)
    frames.append(_window_features(sub, "w1", w1[0], w1[1],
                                   w1[1] - w1[0] + 1, top_types).reset_index())
    frames.append(_submission_features(
        sa, keys, w1[0], w1[1], "w1",
        config["leakage"].get("exclude_banked", True)))

    pres = (master[["code_module", "code_presentation", "first_tma_date"]]
            .drop_duplicates(["code_module", "code_presentation"]).copy())
    from .windows import w2_end_value
    pres["w2_end"] = pres["first_tma_date"].map(
        lambda x: w2_end_value(x, config))
    pres["w2_end_eff"] = np.minimum(cutoff - 1, pres["w2_end"]).astype(int)
    w2_ends = []
    if cutoff > 28:
        for e in sorted(pres["w2_end_eff"].unique().tolist()):
            if e < 28:
                continue
            w2_ends.append(int(e))
            ksel = pres[pres["w2_end_eff"] == e][["code_module", "code_presentation"]]
            v_e = v.merge(ksel, on=["code_module", "code_presentation"], how="inner")
            v_e = v_e[(v_e["date"] >= 28) & (v_e["date"] <= e)]
            max_date_seen = max(max_date_seen,
                                int(v_e["date"].max()) if len(v_e) else -1)
            frames.append(_window_features(v_e, "w2", 28, e, e - 28 + 1,
                                           top_types).reset_index())
            keys_e = keys.merge(ksel, on=["code_module", "code_presentation"],
                                how="inner")
            frames.append(_submission_features(
                sa, keys_e, 28, e, "w2",
                config["leakage"].get("exclude_banked", True)))

    out = keys.copy()
    for f in frames:
        out = out.merge(f, on=KEY, how="left")
    out = _ensure_window_columns(out, "w1", top_types)
    if cutoff > 28:
        out = _ensure_window_columns(out, "w2", top_types)
    out = out.fillna(0.0)
    # Zero-click enrolments are information, not missing rows: give them
    # window-specific "never clicked" defaults instead of 0 (= clicked today).
    length = w1[1] - w1[0] + 1
    zero = out["w1_clicks_total"] <= 0
    if zero.any():
        out.loc[zero, "w1_days_since_last_click"] = float(length + 1)
        out.loc[zero, "w1_first_click_day"] = -1.0
        out.loc[zero, "w1_max_gap"] = float(length)
        out.loc[zero, "w1_longest_streak"] = 0.0
    if w2_ends:
        tmp = out.merge(pres[["code_module", "code_presentation", "w2_end_eff"]],
                        on=["code_module", "code_presentation"], how="left")
        zero = tmp["w2_clicks_total"] <= 0
        length_e = tmp["w2_end_eff"] - 28 + 1
        if zero.any():
            tmp.loc[zero, "w2_days_since_last_click"] = (length_e + 1)[zero]
            tmp.loc[zero, "w2_first_click_day"] = -1.0
            tmp.loc[zero, "w2_max_gap"] = length_e[zero]
            tmp.loc[zero, "w2_longest_streak"] = 0.0
        out = tmp.drop(columns=["w2_end_eff"])
    w2 = (28, max(w2_ends)) if w2_ends else None
    if w2 is not None:
        out["d_clicks"] = out["w2_clicks_total"] - out["w1_clicks_total"]
        out["d_clicks_ratio"] = (out["w2_clicks_total"] + 1) / (out["w1_clicks_total"] + 1)
        out["d_active_days"] = out["w2_active_days"] - out["w1_active_days"]
        out["d_entropy_activity"] = out["w2_entropy_activity"] - out["w1_entropy_activity"]
        out["d_clicks_per_week"] = out["w2_clicks_per_week"] - out["w1_clicks_per_week"]
        out["d_week_slope"] = out["w2_week_slope"] - out["w1_week_slope"]
        out["accel_clicks_per_week"] = out["w2_clicks_per_week"] - out["w1_clicks_per_week"]
        out["d_submissions"] = out["w2_submissions"] - out["w1_submissions"]
    info["max_date_seen"] = max_date_seen
    info["n_rows"] = int(len(out))
    info["n_features"] = int(out.shape[1] - len(KEY))
    info["zero_click_share_w1"] = float((out["w1_clicks_total"] == 0).mean())
    info["w1_window"] = w1
    info["w2_window"] = w2
    info["w2_ends"] = w2_ends
    # strict-feature guard
    forbidden = [c for c in out.columns if any(t in c.lower() for t in
                                               ("score", "is_banked", "final_result",
                                                "date_unregistration", "unregistration"))]
    info["forbidden_columns"] = forbidden
    return out, info


def build_score_features(master: pd.DataFrame, sa: pd.DataFrame, cutoff: int,
                         exclude_banked: bool = True) -> pd.DataFrame:
    """M4-only upper-bound score aggregates.  Not part of strict features."""
    s = sa.copy()
    if exclude_banked:
        s = s[s["is_banked"] == 0]
    s = s[(s["date_submitted"] < cutoff) & s["score"].notna()]
    g = s.groupby(KEY, observed=True)["score"]
    f = pd.DataFrame({
        "score_mean": g.mean(),
        "score_min": g.min(),
        "score_max": g.max(),
        "score_count": g.size(),
    })
    out = master[KEY].merge(f.reset_index(), on=KEY, how="left")
    for c in ["score_mean", "score_min", "score_max"]:
        out[c] = out[c].fillna(out[c].median())
    out["score_count"] = out["score_count"].fillna(0)
    return out


# ---- model-ladder feature selectors ----------------------------------------
FAMILY = {
    "volume": ["clicks_total", "clicks_log1p", "clicks_per_week",
               "clicks_per_active_day", "max_week_clicks", "weekly_std"],
    "breadth": ["active_days", "n_sites", "n_activity_types", "active_weeks"],
    "regularity": ["daily_cv", "max_gap", "longest_streak", "day_phase_entropy"],
    "composition": ["entropy_activity", "gini_activity"],
    "timing": ["last_7d_share", "clicks_last_48h", "days_since_last_click",
               "first_click_day", "week_slope"],
    "assessment": ["submissions", "on_time_share", "late_submissions",
                   "mean_lead_days", "days_since_last_submission",
                   "first_submission_day"],
    "change": ["d_clicks", "d_clicks_ratio", "d_active_days",
               "d_entropy_activity", "d_clicks_per_week", "d_week_slope",
               "accel_clicks_per_week", "d_submissions"],
}


def feature_columns(df: pd.DataFrame, level: int, cutoff: int,
                    include_pre_start: bool = False) -> list[str]:
    """M0..M3 columns available for a cutoff (W2/delta only after day 28)."""
    allowed = {
        0: ["volume"],
        1: ["volume", "breadth", "regularity", "composition"],
        2: ["volume", "breadth", "regularity", "composition", "timing"],
        3: ["volume", "breadth", "regularity", "composition", "timing",
            "assessment"],
    }[level]
    cols = []
    prefixes = ["w1"]
    if cutoff > 28:
        prefixes.append("w2")
    for pre in prefixes:
        for fam in allowed:
            for base in FAMILY[fam]:
                c = f"{pre}_{base}"
                if c in df.columns:
                    cols.append(c)
        # activity shares are part of composition
        if "composition" in allowed:
            cols += [c for c in df.columns if c.startswith(f"{pre}_share_")]
    if cutoff > 28 and level >= 1:
        cols += [c for c in df.columns if c.startswith("d_") or
                 c.startswith("accel_")]
    if include_pre_start:
        cols += [c for c in ("pre_start_clicks", "pre_start_active_days")
                 if c in df.columns]
    # de-duplicate while keeping order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
