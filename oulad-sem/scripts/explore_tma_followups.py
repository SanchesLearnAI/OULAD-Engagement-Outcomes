"""Fifth exploration pass: the three highest-value follow-ups (A1, A2, B1).

A1  Silence structure split by presentation and by post-click layer.
A2  Activity-type sequences with the zero-click channel removed, tested within
    pairs (sign test) so click volume cannot drive the result.
B1  Alert-policy comparison: fixed precision target vs fixed capacity.

Run from ``oulad-sem/``::

    python scripts/explore_tma_followups.py --out ../results/tma_deep_exploration_20261108
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.tma import daily, modeling, pairing  # noqa: E402
from src.tma.stats import bh  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]
CATEGORY = {
    "oucontent": "content", "resource": "content", "subpage": "content",
    "url": "content", "page": "content", "ouwiki": "content",
    "glossary": "content", "dataplus": "content", "dualpane": "content",
    "homepage": "content", "htmlactivity": "content",
    "oucollaborate": "collab", "ouelluminate": "collab", "sharedsubpage": "content",
    "quiz": "assess", "externalquiz": "assess", "questionnaire": "assess",
    "forumng": "forum",
}


def events_with_site(root: Path, cohort: pd.DataFrame, post_days: int) -> pd.DataFrame:
    """Student-day-site clicks inside the window, keeping ``id_site`` for typing."""
    import duckdb

    lookup = cohort[KEY + ["due_day"]]
    path = str(root / "data" / "raw" / "oulad" / "studentVle.csv").replace("'", "''")
    con = duckdb.connect()
    try:
        con.register("lookup", lookup)
        frame = con.execute(f"""
            SELECT e.code_module, e.code_presentation, e.id_student, e.id_site,
                   e.date - l.due_day AS relative_day,
                   sum(e.sum_click) AS clicks
            FROM read_csv_auto('{path}', header=true) e
            JOIN lookup l USING (code_module, code_presentation, id_student)
            WHERE e.date - l.due_day BETWEEN 1 AND {post_days}
            GROUP BY 1, 2, 3, 4, 5
        """).df()
    finally:
        con.close()
    return frame


def silence_indicators(post: np.ndarray) -> pd.DataFrame:
    active = post > 0
    rows = []
    for row in active:
        lengths, kinds = [], []
        current, value = 0, bool(row[0]) if len(row) else False
        for flag in row:
            flag = bool(flag)
            if flag == value:
                current += 1
            else:
                lengths.append(current)
                kinds.append(value)
                current, value = 1, flag
        if current:
            lengths.append(current)
            kinds.append(value)
        silences = [l for l, on in zip(lengths, kinds) if not on]
        returns = sum(1 for on in kinds[:-1] if not on)
        rows.append({"n_silences": len(silences),
                     "longest_silence": max(silences, default=0),
                     "mean_silence": float(np.mean(silences)) if silences else 0.0,
                     "returns_after_silence": returns,
                     "ended_silent": int(bool(len(row) and not row[-1]))})
    return pd.DataFrame(rows)


def a1_silence(frame: pd.DataFrame, indicators: pd.DataFrame,
               pairs: pd.DataFrame, seed: int, repeats: int = 500
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pair differences by presentation and by post-click quartile."""
    data = frame[KEY + ["final_result", "post_clicks"]].join(indicators)
    fi = pairs["fail_index"].to_numpy()
    pi = pairs["pass_index"].to_numpy()
    metrics = ["n_silences", "longest_silence", "mean_silence",
               "returns_after_silence", "ended_silent"]

    def paired_rows(subset: pd.DataFrame, label_key: str, label: str) -> list[dict]:
        f_idx = subset["fail_index"].to_numpy()
        p_idx = subset["pass_index"].to_numpy()
        rows = []
        rng = np.random.default_rng(seed)
        for metric in metrics:
            diff = (data.iloc[f_idx][metric].to_numpy(dtype=float)
                    - data.iloc[p_idx][metric].to_numpy(dtype=float))
            size = max(len(diff), 1)
            draws = [diff[rng.integers(size, size=size)].mean()
                     for _ in range(min(repeats, 2000))]
            rows.append({label_key: label, "metric": metric, "pairs": int(len(diff)),
                         "fail_minus_pass": float(diff.mean()),
                         "ci_low": float(np.quantile(draws, .025)),
                         "ci_high": float(np.quantile(draws, .975))})
        return rows

    by_pres = []
    for (module, presentation), idx in pairs.groupby(BATCH).indices.items():
        subset = pairs.loc[list(idx)].reset_index(drop=True)
        if len(subset) < 50:
            continue
        by_pres += paired_rows(subset, "presentation", f"{module}_{presentation}")

    # Post-click quartiles of the pair's mean post clicks (both sides are close
    # by construction, so the pair has one meaningful level).
    q = pairs.assign(pair_clicks=(pairs["fail_clicks"] + pairs["pass_clicks"]) / 2)
    q["layer"] = pd.qcut(q["pair_clicks"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
                         duplicates="drop")
    by_layer = []
    for layer, subset in q.groupby("layer", observed=True):
        subset = subset.reset_index(drop=True)
        if len(subset) < 50:
            continue
        by_layer += paired_rows(subset, "post_click_layer", str(layer))

    overall = pd.DataFrame(paired_rows(pairs.reset_index(drop=True), "post_click_layer",
                                       "ALL"))
    return pd.DataFrame(by_pres), pd.DataFrame(by_layer), overall


def a2_type_sequences(typed_events: pd.DataFrame, catalog: pd.DataFrame,
                      frame: pd.DataFrame, pairs: pd.DataFrame, post_days: int,
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Most-common daily activity category per student-day, then pair sign tests.

    Only pairs where both sides have at least one post-deadline click are used,
    so the zero-click channel that dominated E6 is removed by construction.
    """
    from scipy.stats import binomtest

    meta = catalog[BATCH + ["id_site", "activity_type"]].copy()
    meta["category"] = meta["activity_type"].map(CATEGORY).fillna("other")
    typed = typed_events.merge(meta.drop(columns=["activity_type"]),
                               on=BATCH + ["id_site"], how="left")
    typed["category"] = typed["category"].fillna("other")
    typed = typed[typed["relative_day"].between(1, post_days)]
    by_cat = (typed.groupby(KEY + ["relative_day", "category"], observed=True)["clicks"]
              .sum().reset_index())
    top = (by_cat.sort_values("clicks", ascending=False)
           .drop_duplicates(KEY + ["relative_day"]))
    letters = {"content": "C", "assess": "A", "forum": "F", "collab": "L", "other": "O"}
    top["letter"] = top["category"].map(letters).fillna("O")
    wide = (top.pivot(index=KEY, columns="relative_day", values="letter")
            .reindex(columns=list(range(1, post_days + 1))))
    symbol = wide.fillna("0")          # 0 = no activity at all that day
    index = pd.MultiIndex.from_frame(frame[KEY])
    symbol = symbol.reindex(index)
    alphabet = sorted(set("".join(symbol.fillna("0").to_numpy().ravel().tolist())))

    def has(pattern: str) -> np.ndarray:
        """Pattern occurs as a contiguous run somewhere in the 14 days."""
        n, days = symbol.shape
        hit = np.zeros(n, dtype=bool)
        width = len(pattern)
        for start in range(0, days - width + 1):
            block = symbol.iloc[:, start:start + width].to_numpy()
            match = np.ones(n, dtype=bool)
            for j, letter in enumerate(pattern):
                match &= block[:, j] == letter
            hit |= match
        return hit

    fi = pairs["fail_index"].to_numpy()
    pi = pairs["pass_index"].to_numpy()
    positive = (pairs["fail_clicks"] > 0).to_numpy()
    fi, pi = fi[positive], pi[positive]
    rows = []
    for length in (2, 3):
        for pattern in [p for p in np.array(np.meshgrid(*[alphabet] * length)).T.reshape(-1, length)
                        if "0" not in p]:
            text = "".join(pattern)
            hit = has(text)
            fail_only = int((hit[fi] & ~hit[pi]).sum())
            pass_only = int((~hit[fi] & hit[pi]).sum())
            if fail_only + pass_only < 15:
                continue
            p = float(binomtest(fail_only, fail_only + pass_only, 0.5).pvalue)
            rows.append({"pattern": text, "length": length,
                         "pairs_fail_only": fail_only,
                         "pairs_pass_only": pass_only,
                         "discordant": fail_only + pass_only,
                         "odds_fail_over_pass": fail_only / max(pass_only, 1),
                         "p_value": p})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_bh"] = bh(table["p_value"].to_numpy())
        table["reject_bh_0_05"] = table["p_bh"] < 0.05
        table = table.sort_values("odds_fail_over_pass", ascending=False).reset_index(drop=True)
    coverage = (symbol.replace("0", np.nan).notna().mean(axis=1).rename("days_with_activity")
                .reset_index())
    return table, coverage


def b1_policies(frame: pd.DataFrame, splits: dict, seed: int) -> pd.DataFrame:
    """Three alert policies on the same main test split."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y = frame["y_fail"].to_numpy(dtype=int)
    tr = np.array(splits["main"]["train"])
    va = np.array(splits["main"]["val"])
    te = np.array(splits["main"]["test"])
    cols = modeling.columns_for(frame, "M2")
    prob, thr, rule = modeling._fit(frame, cols, tr, te, "logistic", seed, va, 0.80)
    pv = None

    def at_threshold(t: float) -> dict:
        flagged = prob >= t
        tp = int((flagged & (y[te] == 1)).sum())
        fp = int((flagged & (y[te] == 0)).sum())
        fn = int((~flagged & (y[te] == 1)).sum())
        return {"threshold": float(t), "flagged": int(flagged.sum()),
                "flagged_share": float(flagged.mean()),
                "precision": float(tp / max(flagged.sum(), 1)),
                "recall": float(tp / max(int((y[te] == 1).sum()), 1)),
                "true_positives": tp, "false_positives": fp, "missed": fn}

    rows = []
    base = at_threshold(0.5)
    rows.append({"policy": "fixed_0.5_probability", **base})
    rows.append({"policy": "validation_rule_" + rule, **at_threshold(thr)})

    # Fixed capacity: top k by score, ties credited at their hit rate.
    for fraction in (0.05, 0.10, 0.20):
        k = max(int(round(len(te) * fraction)), 1)
        order = np.argsort(-prob)
        boundary = prob[order[k - 1]]
        above = prob > boundary
        ties = prob == boundary
        slots = k - int(above.sum())
        tp = y[te][above].sum() + (y[te][ties].mean() if ties.any() and slots > 0 else 0) * slots
        rows.append({"policy": f"fixed_capacity_top_{int(fraction*100)}pct",
                     "threshold": float(boundary), "flagged": k,
                     "flagged_share": fraction,
                     "precision": float(tp / k),
                     "recall": float(tp / max(int((y[te] == 1).sum()), 1)),
                     "true_positives": float(tp),
                     "false_positives": float(k - tp),
                     "missed": float(int((y[te] == 1).sum()) - tp)})

    # Fixed precision: the smallest flagged set that still keeps 0.80 precision
    # on the test data, as an oracle upper bound for comparison.
    precision, recall, thresholds = precision_recall_curve(y[te], prob)
    mask = precision[:-1] >= 0.80
    if mask.any():
        idx = int(np.argmax(np.where(mask, recall[:-1], -1)))
        rows.append({"policy": "oracle_precision_0.80",
                     "threshold": float(thresholds[idx]),
                     "flagged": int((prob >= thresholds[idx]).sum()),
                     "flagged_share": float((prob >= thresholds[idx]).mean()),
                     "precision": float(precision[idx]), "recall": float(recall[idx]),
                     "true_positives": float(precision[idx] * (prob >= thresholds[idx]).sum()),
                     "false_positives": float((1 - precision[idx]) * (prob >= thresholds[idx]).sum()),
                     "missed": float(int((y[te] == 1).sum()) - precision[idx] * (prob >= thresholds[idx]).sum())})
    table = pd.DataFrame(rows)
    table["average_precision"] = float(average_precision_score(y[te], prob))
    table["no_skill"] = float(y[te].mean())
    table["n_test"] = int(len(te))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    post_days = int(cfg["study"]["post_days"])

    master = pd.read_parquet(root / "data" / "interim" / "master.parquet")
    frame = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    splits = json.loads((root / "data" / "processed" / "splits.json").read_text())
    catalog = pd.read_csv(root / "data" / "raw" / "oulad" / "vle.csv", na_values=["?"])
    cohort = binary_cohort(master)
    lookup = cohort[KEY + ["due_day"]]
    events = daily.daily_events(root / "data" / "raw" / "oulad" / "studentVle.csv",
                               lookup, pre_days=14, post_days=post_days)
    index = pd.MultiIndex.from_frame(frame[KEY])
    post = daily.click_matrix(events, index, 1, post_days)
    pairs = pairing.make_pairs(frame, 0.05, cfg["seed"])

    indicators = silence_indicators(post)
    indicators.index = frame.index
    by_pres, by_layer, overall = a1_silence(frame, indicators, pairs, cfg["seed"])
    by_pres.to_csv(out / "a1_silence_by_presentation.csv", index=False)
    by_layer.to_csv(out / "a1_silence_by_click_layer.csv", index=False)
    overall.to_csv(out / "a1_silence_overall.csv", index=False)

    patterns, coverage = a2_type_sequences(events_with_site(root, cohort, post_days),
                                           catalog, frame, pairs, post_days, cfg["seed"])
    patterns.to_csv(out / "a2_type_patterns_pairs.csv", index=False)
    coverage.to_csv(out / "a2_type_coverage.csv", index=False)

    policy = b1_policies(frame, splits, cfg["seed"])
    policy.to_csv(out / "b1_policy_comparison.csv", index=False)

    summary = {
        "a1_presentations": int(by_pres["presentation"].nunique()) if len(by_pres) else 0,
        "a1_longest_silence_by_layer": by_layer[by_layer["metric"].eq("longest_silence")]
        [["post_click_layer", "fail_minus_pass", "pairs"]].to_dict("records")
        if len(by_layer) else [],
        "a2_patterns_tested": int(len(patterns)),
        "a2_significant_bh": int(patterns["reject_bh_0_05"].sum()) if len(patterns) else 0,
        "b1_rows": int(len(policy)),
    }
    (out / "summary_followups.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
