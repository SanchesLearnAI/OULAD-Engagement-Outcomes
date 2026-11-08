"""Daily click matrices and post-deadline timing features.

Ported from ``oulad_analysis/tma_two_week.py`` (``_longest_zero_run``,
``_time_features``, ``build_behavior_features``) and the window aggregates in
``oulad_analysis/assessment_node.py`` (``build_node_features``).  Counts keep
every source row, including repeated student/site/day events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]


def _sql_path(path) -> str:
    return str(path).replace("'", "''")


def _source_sql(path) -> str:
    suffix = str(path).lower()
    if suffix.endswith(".parquet"):
        return f"read_parquet('{_sql_path(path)}')"
    return f"read_csv_auto('{_sql_path(path)}', header=true)"


def _register_source(con, name: str, source) -> str:
    """Register a DataFrame, or return the SQL that reads a CSV/parquet path."""
    if isinstance(source, pd.DataFrame):
        con.register(name, source)
        return name
    return _source_sql(source)


def daily_events(source, lookup: pd.DataFrame, *, pre_days: int = 14,
                 post_days: int = 14) -> pd.DataFrame:
    """Aggregate clicks per student and relative day inside the observation window.

    ``source`` is ``studentVle`` as a CSV or parquet path; ``lookup`` carries
    ``KEY + due_day`` for the eligible cohort.  Returns
    ``KEY + [relative_day, clicks]`` with relative day in
    ``[-(pre_days - 1), post_days]``; the pre window includes the deadline day
    itself, so 14 pre days mean ``d-13 .. d``.
    """
    import duckdb

    con = duckdb.connect()
    try:
        con.register("tma_lookup", lookup)
        src = _register_source(con, "tma_source", source)
        events = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student,
                   e.date - l.due_day AS relative_day,
                   sum(e.sum_click) AS clicks,
                   count(*) FILTER (WHERE e.sum_click IS NULL OR e.sum_click < 0) AS bad_rows
            FROM {src} e
            JOIN tma_lookup l USING (code_module, code_presentation, id_student)
            WHERE e.date - l.due_day BETWEEN {-pre_days + 1} AND {post_days}
            GROUP BY 1, 2, 3, 4
        """).df()
    finally:
        con.close()
    if events.empty:
        return pd.DataFrame(columns=KEY + ["relative_day", "clicks"])
    if events["bad_rows"].sum():
        raise ValueError("Invalid VLE clicks")
    out = events.groupby(KEY + ["relative_day"], as_index=False)["clicks"].sum()
    if not out["relative_day"].between(-pre_days + 1, post_days).all():
        raise AssertionError("Event outside observation window")
    return out


def resource_events(source, lookup: pd.DataFrame, vle: pd.DataFrame, *,
                    post_days: int = 14) -> tuple[pd.DataFrame, dict]:
    """Post-deadline clicks by resource ``activity_type``; unmatched rows kept.

    Mirrors ``build_behavior_features``: join ``vle.csv`` on the three keys,
    label misses ``__unmatched__`` and reconcile totals against the daily feed.
    """
    import duckdb

    site_key = BATCH + ["id_site"]
    meta = vle[site_key + ["activity_type"]].copy()
    if meta.duplicated(site_key).any() or meta["activity_type"].isna().any():
        raise ValueError("Invalid VLE metadata")
    con = duckdb.connect()
    try:
        con.register("tma_lookup", lookup)
        con.register("vle_metadata", meta)
        src = _register_source(con, "tma_source", source)
        events = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student,
                   coalesce(v.activity_type, '__unmatched__') AS activity_type,
                   sum(e.sum_click) AS clicks,
                   count(*) FILTER (WHERE v.activity_type IS NULL) AS unmatched_rows
            FROM {src} e
            JOIN tma_lookup l USING (code_module, code_presentation, id_student)
            LEFT JOIN vle_metadata v USING (code_module, code_presentation, id_site)
            WHERE e.date - l.due_day BETWEEN 1 AND {post_days}
            GROUP BY 1, 2, 3, 4
        """).df()
    finally:
        con.close()
    quality = {"unmatched_resource_event_rows": int(events["unmatched_rows"].sum())
               if len(events) else 0}
    return events.drop(columns=["unmatched_rows"]), quality


def click_matrix(daily: pd.DataFrame, index: pd.MultiIndex, lo: int, hi: int) -> np.ndarray:
    """Dense ``(records, hi-lo+1)`` click matrix, zero-filled on missing days."""
    daily = daily[daily["relative_day"].between(lo, hi)]
    wide = (daily.pivot(index=KEY, columns="relative_day", values="clicks")
            .reindex(index=index, columns=list(range(lo, hi + 1))).fillna(0))
    return wide.to_numpy(dtype=float)


def longest_zero_run(clicks: np.ndarray) -> np.ndarray:
    """Longest run of consecutive zero-click days inside the supplied window."""
    run = np.zeros(len(clicks), dtype=int)
    longest = run.copy()
    for day in range(clicks.shape[1]):
        run = np.where(clicks[:, day] > 0, 0, run + 1)
        longest = np.maximum(longest, run)
    return longest


def time_features(post: np.ndarray, post_days: int) -> dict[str, np.ndarray]:
    """Post-window timing summaries; identical formulas to the two-week study."""
    total = post.sum(axis=1)
    shares = np.divide(post, total[:, None], out=np.zeros_like(post, dtype=float),
                       where=total[:, None] > 0)
    entropy = -(shares * np.log(np.where(shares > 0, shares, 1))).sum(axis=1)
    cumulative = np.pad(np.cumsum(post, axis=1), ((0, 0), (1, 0)))
    if post.shape[1] >= 7:
        peak7 = (cumulative[:, 7:] - cumulative[:, :-7]).max(axis=1)
    else:
        peak7 = post.sum(axis=1)
    return {
        "post_clicks": total,
        "post_active_days": (post > 0).sum(axis=1).astype(float),
        "post_longest_inactive_run": longest_zero_run(post),
        "post_week2_share": np.divide(post[:, 7:].sum(axis=1), total,
                                      out=np.zeros(len(post)), where=total > 0),
        "post_peak7_share": np.divide(peak7, total, out=np.zeros(len(post)), where=total > 0),
        "post_entropy_norm": entropy / np.log(post_days) if post_days > 1 else entropy * 0.0,
        "post_first_active_day": np.where((post > 0).any(axis=1),
                                          (post > 0).argmax(axis=1) + 1, post_days + 1),
    }


def week_features(post: np.ndarray, post_days: int) -> dict[str, np.ndarray]:
    """Week-block activity used by the primary persistence indicators."""
    weeks = (post_days + 6) // 7
    blocks = [post[:, i * 7:min((i + 1) * 7, post_days)] for i in range(weeks)]
    active = [b.sum(axis=1) > 0 for b in blocks]
    both = active[0] & active[1] if weeks >= 2 else np.zeros(len(post), dtype=bool)
    return {
        "post_week1_active_days": (blocks[0] > 0).sum(axis=1).astype(float),
        "post_week2_active_days": (blocks[1] > 0).sum(axis=1).astype(float) if weeks >= 2
        else np.zeros(len(post)),
        "post_both_weeks": both.astype(int),
        "post_active_weeks": np.sum(np.vstack(active), axis=0).astype(int),
        "post_active_week_fraction": np.sum(np.vstack(active), axis=0) / weeks,
    }


def pre_features(pre: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-window volume features; the last three days are the burst window."""
    total = pre.sum(axis=1)
    last3 = pre[:, -3:].sum(axis=1)
    return {
        "pre_clicks": total,
        "pre_active_days": (pre > 0).sum(axis=1).astype(float),
        "last3_clicks": last3,
        "last3_share": np.divide(last3, total, out=np.zeros(len(pre)),
                                 where=total > 0),
    }


def build_daily_features(frame: pd.DataFrame, daily: pd.DataFrame,
                         resources: pd.DataFrame | None, *,
                         pre_days: int = 14, post_days: int = 14) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Attach the full TMA feature set and reconcile it against the raw events.

    ``frame`` must be the binary cohort (Fail vs Pass/Distinction) with
    ``due_day``.  Returns the enriched frame, the post matrix and quality stats.
    """
    index = pd.MultiIndex.from_frame(frame[KEY])
    pre = click_matrix(daily, index, -pre_days + 1, 0)
    post = click_matrix(daily, index, 1, post_days)

    out = frame.copy()
    for name, values in pre_features(pre).items():
        out[name] = values
    for name, values in time_features(post, post_days).items():
        out[name] = values
    for name, values in week_features(post, post_days).items():
        out[name] = values
    out["log_pre_clicks"] = np.log1p(out["pre_clicks"])
    out["log_post_clicks"] = np.log1p(out["post_clicks"])
    out["log_post_clicks_squared"] = out["log_post_clicks"] ** 2
    out["post_clicks_squared"] = out["post_clicks"] ** 2
    out["post_minus_pre_log"] = out["log_post_clicks"] - out["log_pre_clicks"]
    out["module_presentation"] = out["code_module"] + "_" + out["code_presentation"]

    quality = {
        "binary_records": int(len(out)),
        "fail_records": int(out["final_result"].eq("Fail").sum()),
        "zero_post_click_records": int(out["post_clicks"].eq(0).sum()),
        "post_days": int(post_days),
        "pre_days": int(pre_days),
    }
    if resources is not None:
        types = sorted(resources["activity_type"].unique().tolist())
        block = resources.groupby(KEY + ["activity_type"], as_index=False)["clicks"].sum()
        wide = (block.pivot(index=KEY, columns="activity_type", values="clicks")
                .reindex(index=index, columns=types).fillna(0))
        values = wide.to_numpy(dtype=float)
        if not np.array_equal(values.sum(axis=1), out["post_clicks"].to_numpy(dtype=float)):
            raise ValueError("Post resource totals disagree with daily clicks")
        denom = out["post_clicks"].to_numpy(dtype=float)[:, None]
        shares = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)
        out["post_resource_type_count"] = (values > 0).sum(axis=1)
        out["post_resource_entropy"] = -(shares * np.log(np.where(shares > 0, shares, 1))).sum(axis=1)
        for j, name in enumerate(types):
            out[f"post_resource_share__{name}"] = shares[:, j]
        quality["resource_types_observed"] = types
    return out, post, quality
