"""Coarsened exact matching with balance diagnostics and cluster bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_VARS = ["w1_clicks_log1p", "w1_active_days", "w1_entropy_activity"]


def _qbin(s: pd.Series, q: int) -> pd.Series:
    n = len(s)
    if n < 4:
        return pd.Series(0, index=s.index, dtype=int)
    qq = max(2, min(int(q), n // 2))
    r = s.rank(method="first")
    return pd.qcut(r, qq, labels=False, duplicates="drop").astype(int)


def _coarsen(df: pd.DataFrame, level: int, match_vars=None,
             click_bins=None, active_edges=None, entropy_bins=None) -> pd.DataFrame:
    out = df.copy()
    match_vars = list(match_vars or MATCH_VARS)
    pres = ["code_module", "code_presentation"]
    if level == 0:
        cq, edges, eq = 10, [0, 1, 4, 8, 15, 22, 29], 3
    elif level == 1:
        cq, edges, eq = 5, [0, 1, 4, 8, 15, 29], 3
    elif level == 2:
        cq, edges, eq = 4, [0, 1, 8, 29], 2
    elif level == 3:
        cq, edges, eq = 2, [0, 8, 29], 2
    else:
        cq, edges, eq = 2, [0, 29], 2
    cq = int(click_bins) if click_bins is not None else cq
    edges = list(active_edges) if active_edges is not None else edges
    eq = int(entropy_bins) if entropy_bins is not None else eq
    if "w1_clicks_log1p" in match_vars:
        out["_click_bin"] = out.groupby(pres, observed=True)["w1_clicks_log1p"].transform(
            lambda s: _qbin(s, cq))
    else:
        out["_click_bin"] = 0
    if "w1_active_days" in match_vars:
        out["_active_bin"] = pd.cut(out["w1_active_days"], bins=edges, right=False,
                                    labels=False, include_lowest=True).fillna(
            len(edges) - 1).astype(int)
    else:
        out["_active_bin"] = 0
    if "w1_entropy_activity" in match_vars:
        out["_entropy_bin"] = out.groupby(pres, observed=True)["w1_entropy_activity"].transform(
            lambda s: _qbin(s, eq))
    else:
        out["_entropy_bin"] = 0
    return out


def _wmean(x, w):
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / max(np.sum(w), 1e-12))


def _wvar(x, w):
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    m = _wmean(x, w)
    return float(np.sum(w * (x - m) ** 2) / max(np.sum(w), 1e-12))


def smd(x, w, t):
    t = np.asarray(t, dtype=bool); w = np.asarray(w, dtype=float)
    mt, mc = _wmean(x[t], w[t]), _wmean(x[~t], w[~t])
    vt, vc = _wvar(x[t], w[t]), _wvar(x[~t], w[~t])
    den = np.sqrt(max((vt + vc) / 2.0, 1e-12))
    return float((mt - mc) / den)


def _binned_smd(df, t, w, cols):
    worst = 0.0
    rows = []
    for c in cols:
        for lev in sorted(df[c].dropna().unique()):
            x = (df[c] == lev).astype(float).to_numpy()
            s = smd(x, w, t)
            rows.append({"variable": c, "level": lev, "smd_after": s})
            if abs(s) > abs(worst):
                worst = s
    return float(worst), rows


def cem(df: pd.DataFrame, config: dict, max_levels: int = 5,
        match_vars=None, click_bins=None, active_edges=None,
        entropy_bins=None):
    """Return matched frame with ATT weights, balance rows and attempt log."""
    match_vars = list(match_vars or MATCH_VARS)
    attempts = []
    chosen = None
    for level in range(max_levels):
        d = _coarsen(df, level, match_vars, click_bins, active_edges,
                     entropy_bins)
        strata_cols = ["code_module", "code_presentation", "_click_bin",
                       "_active_bin", "_entropy_bin"]
        d["stratum"] = d.groupby(strata_cols, observed=True).ngroup()
        g = d.groupby("stratum", observed=True)["y_t1"].agg(["size", "sum"])
        g["n_t"] = g["sum"].astype(int)
        g["n_c"] = g["size"].astype(int) - g["n_t"]
        keep_ids = g.index[(g["n_t"] > 0) & (g["n_c"] > 0)]
        d["kept"] = d["stratum"].isin(keep_ids)
        matched = d[d["kept"]].copy()
        n_t = int((matched["y_t1"] == 1).sum())
        n_c = int((matched["y_t1"] == 0).sum())
        # ATT weights: treated 1; control n_t/n_c inside stratum.
        cnt = matched.groupby("stratum", observed=True)["y_t1"].agg(["size", "sum"])
        cnt["n_t"] = cnt["sum"].astype(int)
        cnt["n_c"] = cnt["size"].astype(int) - cnt["n_t"]
        ratio = (cnt["n_t"] / cnt["n_c"]).rename("cw")
        matched = matched.merge(ratio, left_on="stratum", right_index=True, how="left")
        matched["att_w"] = np.where(matched["y_t1"] == 1, 1.0, matched["cw"])
        t = matched["y_t1"].to_numpy() == 1
        w = matched["att_w"].to_numpy()
        bin_worst, bin_rows = _binned_smd(
            matched, t, w, ["_click_bin", "_active_bin", "_entropy_bin"])
        raw_after = {v: smd(matched[v].to_numpy(), w, t) for v in match_vars}
        raw_before = {v: smd(df[v].to_numpy(), np.ones(len(df)),
                             df["y_t1"].to_numpy() == 1) for v in match_vars}
        attempts.append({
            "level": level, "n_matched": len(matched), "n_treated": n_t,
            "n_control": n_c, "n_strata": int(len(keep_ids)),
            "binned_max_abs_smd": abs(bin_worst),
            "raw_max_abs_smd": max(abs(v) for v in raw_after.values()),
            "accepted": bool(len(matched) >= 30 and abs(bin_worst) < 0.1),
        })
        if chosen is None:
            chosen = (d, matched, keep_ids, t, w, raw_before, raw_after,
                      bin_worst, bin_rows, level)
        if len(matched) >= 30 and abs(bin_worst) < 0.1:
            chosen = (d, matched, keep_ids, t, w, raw_before, raw_after,
                      bin_worst, bin_rows, level)
            break
    d, matched, keep_ids, t, w, raw_before, raw_after, bin_worst, bin_rows, level = chosen
    result = {
        "data": d, "matched": matched, "keep_ids": keep_ids,
        "level": level, "attempts": attempts,
        "n_matched": int(len(matched)),
        "n_unmatched": int(len(d) - len(matched)),
        "n_treated_matched": int((matched["y_t1"] == 1).sum()),
        "n_control_matched": int((matched["y_t1"] == 0).sum()),
        "n_strata": int(len(keep_ids)),
        "n_discordant_units": int(len(matched)),
        "binned_max_abs_smd": float(abs(bin_worst)),
        "raw_before": raw_before, "raw_after": raw_after,
        "bin_rows": bin_rows, "match_vars": match_vars,
        "common_support_treated": float((matched["y_t1"] == 1).sum() /
                                        max((d["y_t1"] == 1).sum(), 1)),
    }
    return result


def balance_table(res) -> pd.DataFrame:
    rows = []
    for v in MATCH_VARS:
        rows.append({
            "variable": v, "kind": "continuous",
            "smd_before": res["raw_before"][v],
            "smd_after": res["raw_after"][v],
            "abs_smd_before": abs(res["raw_before"][v]),
            "abs_smd_after": abs(res["raw_after"][v]),
        })
    for r in res["bin_rows"]:
        rows.append({"variable": r["variable"], "kind": f"bin_{r['level']}",
                     "smd_before": np.nan, "smd_after": r["smd_after"],
                     "abs_smd_before": np.nan, "abs_smd_after": abs(r["smd_after"])})
    return pd.DataFrame(rows)


def w2_differences(res, metrics, n_boot=1000, seed=42):
    d = res["matched"]
    strata = d["stratum"].to_numpy()
    uniq = np.unique(strata)
    groups = {s: np.where(strata == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        if m not in d.columns:
            continue
        x = d[m].to_numpy(dtype=float)
        t = (d["y_t1"].to_numpy() == 1)
        w = d["att_w"].to_numpy(dtype=float)
        mt, mc = _wmean(x[t], w[t]), _wmean(x[~t], w[~t])
        diffs = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([groups[s] for s in pick])
            xx, tt, ww = x[idx], t[idx], w[idx]
            if tt.sum() == 0 or (~tt).sum() == 0:
                continue
            diffs.append(_wmean(xx[tt], ww[tt]) - _wmean(xx[~tt], ww[~tt]))
        lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else
                  (np.nan, np.nan))
        rows.append({"metric": m, "mean_pass": mt, "mean_fail": mc,
                     "diff_pass_minus_fail": mt - mc,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "n_boot": len(diffs)})
    return pd.DataFrame(rows)


def e_value_rr(rr: float) -> float:
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))
