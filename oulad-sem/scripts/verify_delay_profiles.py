"""V4 — delay-profile cross-check against the published delay typology.

The deep exploration found three SFI profiles that split mostly by window length.
This script puts that finding next to the outcome mix and the click volume, so the
report can state whether OULAD supports the "high-achieving / low-achieving /
sporadic delayer" typology.

Run from ``oulad-sem/``::

    python scripts/verify_delay_profiles.py
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

KEY = ["code_module", "code_presentation", "id_student"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = args.out or (root.parent / "results" / "tma_deep_exploration_20261108")
    out = Path(out).resolve()

    profiles = pd.read_csv(out / "e8_profile_mix.csv")
    sfi = pd.read_parquet(out / "e8_sfi_records.parquet")
    features = pd.read_parquet(root / "data" / "processed" / "features.parquet")
    concordance = pd.read_csv(out / "e8_pair_concordance.csv")

    # Rebuild the profile assignment so the outcome mix can be read per profile.
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    candidates = pd.read_csv(out / "e8_profile_candidates.csv")
    best = int(candidates.sort_values(["silhouette", "mean_ari"],
                                      ascending=False).iloc[0]["k"])
    # Restrict to the binary cohort so the profile table and the outcome mix use
    # the same population; ``e8_sfi_records.parquet`` covers the full master frame.
    binary_keys = features[KEY].drop_duplicates()
    sfi = sfi.merge(binary_keys, on=KEY, how="inner", validate="one_to_one")
    data = sfi[sfi["sfi"].notna()].copy()
    data = data[data["sfi"].between(0, 1.5)].copy()
    x = StandardScaler().fit_transform(data[["sfi", "window_days"]].to_numpy(dtype=float))
    labels = KMeans(n_clusters=best, n_init=20, random_state=cfg["seed"]).fit_predict(x)
    order = pd.Series(labels).groupby(labels).apply(
        lambda s: data["sfi"].to_numpy()[s.index].mean()).sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    names = {0: "consistent_early", 1: "gradual_early", 2: "gradual_late",
             3: "consistent_late", 4: "very_late"}
    data["profile"] = np.array([mapping[l] for l in labels])
    data["profile_name"] = data["profile"].map(lambda i: names.get(i, f"p{i}"))

    merged = features[KEY + ["final_result", "post_clicks", "pre_clicks",
                             "post_active_days"]].merge(
        data[KEY + ["profile_name", "sfi", "window_days"]], on=KEY, how="inner",
        validate="one_to_one")

    cross = (merged.groupby(["profile_name", "final_result"]).size()
             .unstack(fill_value=0))
    cross["records"] = cross.sum(axis=1)
    for column in ("Pass", "Distinction", "Fail", "Withdrawn"):
        if column in cross:
            cross[f"share_{column}"] = cross[column] / cross["records"]
    cross = cross.reset_index()
    cross.to_csv(out / "v4_profile_by_outcome.csv", index=False)

    profile_stats = (merged.groupby("profile_name")
                     .agg(records=("id_student", "size"),
                          mean_sfi=("sfi", "mean"),
                          median_window_days=("window_days", "median"),
                          median_post_clicks=("post_clicks", "median"),
                          median_pre_clicks=("pre_clicks", "median"),
                          mean_post_active_days=("post_active_days", "mean"),
                          fail_share=("final_result",
                                      lambda s: float(s.eq("Fail").mean())),
                          pass_share=("final_result",
                                      lambda s: float(s.isin(["Pass", "Distinction"]).mean())))
                     .reset_index())
    profile_stats.to_csv(out / "v4_profile_stats.csv", index=False)

    # How much of the profile split is explained by window length alone?
    between = profile_stats["median_window_days"].max() - \
        profile_stats["median_window_days"].min()
    sfi_range = profile_stats["mean_sfi"].max() - profile_stats["mean_sfi"].min()
    # The decisive check: if the profiles only proxy window length, then inside a
    # window-length bucket the fail share should stop moving across profiles.
    merged["window_bucket"] = pd.qcut(merged["window_days"], 3,
                                      labels=["short", "medium", "long"],
                                      duplicates="drop")
    within = (merged.groupby(["window_bucket", "profile_name"], observed=True)
              .agg(records=("id_student", "size"),
                   fail_share=("final_result",
                               lambda s: float(s.eq("Fail").mean())))
              .reset_index())
    within = within[within["records"] >= 50]
    within.to_csv(out / "v4_profile_within_window.csv", index=False)
    within_spread = (float(within.groupby("window_bucket")["fail_share"]
                           .agg(lambda s: s.max() - s.min()).max())
                     if len(within) else None)

    clamped = merged[merged["sfi"].between(0.8, 1.0)]
    clamped_stats = (clamped.groupby("profile_name")
                     .agg(records=("id_student", "size"),
                          fail_share=("final_result",
                                      lambda s: float(s.eq("Fail").mean())))
                     .reset_index()) if len(clamped) else pd.DataFrame()
    clamped_stats.to_csv(out / "v4_profile_clamped.csv", index=False)

    payload = {
        "profiles": int(profile_stats["profile_name"].nunique()),
        "records_profiled": int(profile_stats["records"].sum()),
        "median_window_days_spread": float(between),
        "mean_sfi_spread": float(sfi_range),
        "fail_share_range": float(profile_stats["fail_share"].max()
                                  - profile_stats["fail_share"].min()),
        "same_profile_share_in_pairs": float(
            concordance.query("profile_name == 'ALL'")["same_profile_share"].iloc[0])
        if len(concordance) else None,
        "chance_agreement": float((profile_stats["records"]
                                   / profile_stats["records"].sum()).pow(2).sum()),
        "clamped_records": int(len(clamped)),
        "clamped_fail_share_range": float(
            clamped_stats["fail_share"].max() - clamped_stats["fail_share"].min())
        if len(clamped_stats) else None,
        "within_window_fail_spread": within_spread,
        "within_window_small_groups": (within[within["records"] < 500]
                                       [["window_bucket", "profile_name", "records",
                                         "fail_share"]].round(3).to_dict("records")
                                       if len(within) else None),
        "spread_driven_by_small_group": bool(
            len(within) and within["records"].min() < 500),
        "window_length_variance_note": (
            "k-means runs on standardised (SFI, window_days); window_days is "
            "strongly bimodal (median 55-64 vs 147), so the profiles are close to "
            "a split on window length alone."),
        "can_test_typology": False,
        "supports_published_typology": bool(
            within_spread is not None and within_spread >= 0.10
            and not (len(within) and within["records"].min() < 500)),
        "caveat": ("Profile labels come from k-means on (SFI, window length) with no "
                   "outcome input. ``within_window_fail_spread`` is the largest "
                   "fail-share spread between profiles inside one window-length "
                   "bucket: if it is small, the profiles mostly proxy window length "
                   "and do not reproduce the published high/low/sporadic delayer "
                   "typology."),
    }
    (out / "v4_profile_verdict.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
