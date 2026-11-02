"""Build one row per student, module and presentation from OULAD CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
WINDOW = 28


def _entropy(values: np.ndarray) -> float:
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _longest_gap(active_dates: np.ndarray) -> int:
    if len(active_dates) == 0:
        return WINDOW
    boundaries = np.r_[-1, np.sort(active_dates), WINDOW]
    return int(np.diff(boundaries).max() - 1)


def build_features(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / f"{name}.csv" for name in ("studentInfo", "studentVle", "vle")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OULAD CSV files: {missing}")

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    for name, path in paths.items():
        escaped_path = str(path).replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{escaped_path}', header=true)")
    stats: dict[str, object] = {}
    stats["student_info_rows"] = con.execute("SELECT count(*) FROM studentInfo").fetchone()[0]
    stats["student_info_missing_core_rows"] = con.execute("""
        SELECT count(*) FROM studentInfo
        WHERE code_module IS NULL OR code_presentation IS NULL
           OR id_student IS NULL OR final_result IS NULL
    """).fetchone()[0]
    stats["student_info_duplicate_keys"] = con.execute(
        "SELECT count(*) FROM (SELECT 1 FROM studentInfo GROUP BY code_module, code_presentation, id_student HAVING count(*) > 1)"
    ).fetchone()[0]
    stats["vle_duplicate_keys"] = con.execute(
        "SELECT count(*) FROM (SELECT 1 FROM vle GROUP BY code_module, code_presentation, id_site HAVING count(*) > 1)"
    ).fetchone()[0]
    stats["vle_missing_core_rows"] = con.execute("""
        SELECT count(*) FROM vle
        WHERE code_module IS NULL OR code_presentation IS NULL
           OR id_site IS NULL OR activity_type IS NULL
    """).fetchone()[0]
    if (stats["student_info_duplicate_keys"] or stats["vle_duplicate_keys"]
            or stats["student_info_missing_core_rows"] or stats["vle_missing_core_rows"]):
        raise ValueError(f"Invalid core fields or non-unique join keys: {stats}")

    con.execute("""
        CREATE TEMP TABLE events AS
        SELECT code_module, code_presentation, id_student, id_site, date, sum_click
        FROM studentVle WHERE date BETWEEN 0 AND 27
    """)
    stats["window_rows"] = con.execute("SELECT count(*) FROM events").fetchone()[0]
    stats["window_missing_core_rows"] = con.execute("""
        SELECT count(*) FROM events
        WHERE code_module IS NULL OR code_presentation IS NULL
           OR id_student IS NULL OR id_site IS NULL OR date IS NULL
           OR sum_click IS NULL
    """).fetchone()[0]
    stats["window_distinct_rows"] = con.execute("""
        SELECT count(DISTINCT (code_module, code_presentation, id_student, id_site, date, sum_click)) FROM events
    """).fetchone()[0]
    stats["window_distinct_student_site_days"] = con.execute("""
        SELECT count(DISTINCT (code_module, code_presentation, id_student, id_site, date)) FROM events
    """).fetchone()[0]
    stats["negative_click_rows"] = con.execute("SELECT count(*) FROM events WHERE sum_click < 0 OR sum_click IS NULL").fetchone()[0]
    if stats["negative_click_rows"] or stats["window_missing_core_rows"]:
        raise ValueError("Missing core fields or negative click counts in the analysis window")
    stats["orphan_student_events"] = con.execute("""
        SELECT count(*) FROM events e ANTI JOIN studentInfo s
        USING (code_module, code_presentation, id_student)
    """).fetchone()[0]
    stats["unmatched_vle_events"] = con.execute("""
        SELECT count(*) FROM events e ANTI JOIN vle v
        USING (code_module, code_presentation, id_site)
    """).fetchone()[0]
    con.execute("""
        CREATE TEMP TABLE activity AS
        SELECT e.code_module, e.code_presentation, e.id_student, e.date,
               coalesce(v.activity_type, '__unmatched__') AS activity_type,
               sum(e.sum_click) AS clicks
        FROM events e LEFT JOIN vle v
        USING (code_module, code_presentation, id_site)
        GROUP BY 1, 2, 3, 4, 5
    """)
    base = con.execute("SELECT code_module, code_presentation, id_student, final_result FROM studentInfo").df()
    daily = con.execute("""
        SELECT code_module, code_presentation, id_student, date, sum(clicks) AS clicks
        FROM activity GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
    """).df()
    types = con.execute("""
        SELECT code_module, code_presentation, id_student, activity_type, sum(clicks) AS clicks
        FROM activity GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
    """).df()
    con.close()

    daily = daily[daily.clicks > 0].copy()
    types = types[types.clicks > 0].copy()
    if len(daily):
        grouped = daily.groupby(KEY, sort=False)
        total = grouped.clicks.sum().rename("total_clicks")
        active = grouped.date.nunique().rename("active_days")
        gap = grouped.date.apply(lambda x: _longest_gap(x.to_numpy())).rename("longest_inactive_run")
        hhi = grouped.clicks.apply(lambda x: float(np.square(x.to_numpy() / x.sum()).sum())).rename("daily_concentration")
        day_entropy = grouped.clicks.apply(lambda x: _entropy(x.to_numpy())).rename("daily_entropy")
        daily["week"] = daily.date // 7
        weeks = daily.pivot_table(index=KEY, columns="week", values="clicks", aggfunc="sum", fill_value=0)
        weeks = weeks.reindex(columns=range(4), fill_value=0)
        weeks.columns = [f"week_{i}_share" for i in range(4)]
        weeks = weeks.div(total, axis=0)
        aggregate = pd.concat([total, active, gap, hhi, day_entropy, weeks], axis=1).reset_index()
    else:
        aggregate = pd.DataFrame(columns=KEY)

    if len(types):
        type_total = types.groupby(KEY).clicks.sum()
        type_count = types.groupby(KEY).activity_type.nunique().rename("resource_type_count")
        type_entropy = types.groupby(KEY).clicks.apply(lambda x: _entropy(x.to_numpy())).rename("resource_entropy")
        shares = types.pivot_table(index=KEY, columns="activity_type", values="clicks", aggfunc="sum", fill_value=0)
        resource_names = sorted(str(c) for c in shares.columns)
        shares = shares.reindex(columns=resource_names).div(type_total, axis=0)
        shares.columns = [f"resource_share__{name}" for name in resource_names]
        aggregate = aggregate.merge(pd.concat([type_count, type_entropy, shares], axis=1).reset_index(), on=KEY, how="outer", validate="one_to_one")
    else:
        resource_names = []

    frame = base.merge(aggregate, on=KEY, how="left", validate="one_to_one")
    numeric = frame.columns.difference(KEY + ["final_result"])
    frame[numeric] = frame[numeric].fillna(0)
    frame["total_clicks"] = frame.get("total_clicks", 0)
    frame["log_clicks"] = np.log1p(frame.total_clicks)
    frame["zero_click"] = (frame.total_clicks == 0).astype(int)
    frame.loc[frame.zero_click == 1, "longest_inactive_run"] = WINDOW
    for week in range(4):
        column = f"week_{week}_share"
        if column not in frame:
            frame[column] = 0.0
    for column in ("active_days", "daily_concentration", "daily_entropy", "resource_type_count", "resource_entropy"):
        if column not in frame:
            frame[column] = 0.0
    stats["feature_rows"] = len(frame)
    stats["unique_students"] = int(frame.id_student.nunique())
    stats["zero_click_students"] = int(frame.zero_click.sum())
    stats["label_counts"] = {str(k): int(v) for k, v in frame.final_result.value_counts(dropna=False).items()}
    stats["resource_types"] = resource_names
    frame.to_parquet(output_dir / "features.parquet", index=False)
    (output_dir / "data_quality.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return frame
