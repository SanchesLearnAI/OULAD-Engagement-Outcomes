"""Third exploration pass: delay archetypes and re-engagement episodes.

E8  Submission Fraction Index (SFI): where inside the *available* window the
    submission landed, then outcome-blind delay profiles and their outcome mix.
E9  Re-engagement episodes after the deadline: how many silences a student has,
    whether they come back after each one, and how that differs by outcome.

Both are descriptive.  ``SFI`` follows the normalisation argument in the recent
delay literature (not all delay is procrastination): a submission with the same
signed distance from the deadline is a different behaviour when the window is
three days long than when it is five weeks long.

Run from ``oulad-sem/``::

    python scripts/explore_tma_delay.py --out ../results/tma_deep_exploration_20261108
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
from src.tma import daily, pairing  # noqa: E402
from src.tma.anchor import binary_cohort  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]


def submission_fraction(master: pd.DataFrame, submissions: pd.DataFrame
                        ) -> pd.DataFrame:
    """SFI = elapsed share of the registration-to-deadline window at submission.

    ``0`` means the moment the student registered, ``1`` the deadline.  Values
    outside ``[0, 1.5]`` are kept but flagged, because a submission after the
    deadline is a different category rather than a longer window.
    """
    sub = submissions[["id_assessment", "id_student", "date_submitted"]]
    sub = sub.drop_duplicates(["id_assessment", "id_student"])
    frame = master[KEY + ["id_assessment", "due_day", "date_registration",
                          "date_unregistration", "final_result", "eligible",
                          "submitted_by_cutoff"]].copy()
    # Presentations without an anchor carry a missing id_assessment; an index
    # join leaves those unmatched instead of forming a NaN key group.
    frame = frame.join(sub.set_index(["id_assessment", "id_student"]),
                       on=["id_assessment", "id_student"], how="left")
    window = frame["due_day"] - frame["date_registration"]
    elapsed = frame["date_submitted"] - frame["date_registration"]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["sfi"] = np.where(window > 0, elapsed / window, np.nan)
    frame["sfi"] = frame["sfi"].where(frame["submitted_by_cutoff"].eq(1))
    frame["window_days"] = window
    frame["sfi_flag_out_of_range"] = (~frame["sfi"].between(0, 1.5)).fillna(False)
    return frame


def delay_profiles(sfi: pd.DataFrame, seed: int = 42, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-blind k-means over (SFI, window length) with stability replay."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    data = sfi[sfi["sfi"].notna() & sfi["eligible"]].copy()
    data = data[data["sfi"].between(0, 1.5)]
    if len(data) < 100:
        return pd.DataFrame(), pd.DataFrame()
    x = StandardScaler().fit_transform(data[["sfi", "window_days"]].to_numpy(dtype=float))
    candidates = []
    rng = np.random.default_rng(seed)
    fitted = {}
    for kk in range(2, 6):
        model = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit(x)
        labels = model.labels_
        if np.bincount(labels).min() < 0.05 * len(data):
            continue
        ari = []
        for repeat in range(5):
            sub = np.sort(rng.choice(len(data), int(0.8 * len(data)), replace=False))
            rerun = KMeans(n_clusters=kk, n_init=10, random_state=seed + repeat + 1) \
                .fit_predict(x[sub])
            ari.append(adjusted_rand_score(labels[sub], rerun))
        candidates.append({"k": kk,
                           "silhouette": float(silhouette_score(
                               x, labels, sample_size=min(2000, len(data)), random_state=seed)),
                           "mean_ari": float(np.mean(ari))})
        fitted[kk] = labels
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    table = pd.DataFrame(candidates)
    best = int(table.sort_values(["silhouette", "mean_ari"], ascending=False)
               .iloc[0]["k"])
    data["profile"] = fitted[best]
    # Order profiles by mean SFI so the labels read early -> late.
    order = data.groupby("profile")["sfi"].mean().sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    data["profile"] = data["profile"].map(mapping)
    names = {0: "consistent_early", 1: "gradual_early", 2: "gradual_late",
             3: "consistent_late", 4: "very_late"}
    data["profile_name"] = data["profile"].map(lambda i: names.get(i, f"profile_{i}"))
    profile = (data.groupby(["profile", "profile_name"])
               .agg(records=("id_student", "size"),
                    mean_sfi=("sfi", "mean"),
                    median_window_days=("window_days", "median"),
                    fail_share=("final_result", lambda s: float(s.eq("Fail").mean())),
                    withdrawn_share=("final_result",
                                     lambda s: float(s.eq("Withdrawn").mean())),
                    pass_share=("final_result",
                                lambda s: float(s.isin(["Pass", "Distinction"]).mean())))
               .reset_index())
    table["selected_k"] = best
    return table, profile


def profile_pairs(profile_rows: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Same-profile share inside matched pairs, as a descriptive concordance."""
    if "profile_name" not in profile_rows:
        return pd.DataFrame()
    names = profile_rows.set_index(profile_rows.index)["profile_name"]
    fi = pairs["fail_index"].to_numpy()
    pi = pairs["pass_index"].to_numpy()
    fail_name = names.reindex(fi).to_numpy()
    pass_name = names.reindex(pi).to_numpy()
    valid = pd.notna(fail_name) & pd.notna(pass_name)
    same = (fail_name == pass_name) & valid
    rows = []
    for level in sorted(set(pass_name[valid])):
        mask = valid & ((fail_name == level) | (pass_name == level))
        rows.append({"profile_name": level, "pairs_touching_profile": int(mask.sum()),
                     "same_profile_share": float(same[mask].mean()) if mask.sum() else np.nan})
    rows.append({"profile_name": "ALL", "pairs_touching_profile": int(valid.sum()),
                 "same_profile_share": float(same[valid].mean()) if valid.any() else np.nan})
    return pd.DataFrame(rows)


def reengagement(post: np.ndarray, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run-length structure of the post-deadline activity stream.

    A *silence* is a maximal run of zero-click days that is followed by at least
    one active day, so a silence that is later broken counts as a return.
    """
    active = post > 0
    rows = []
    for row in active:
        run_lengths, run_active = [], []
        current, value = 0, bool(row[0]) if len(row) else False
        for flag in row:
            flag = bool(flag)
            if flag == value:
                current += 1
            else:
                run_lengths.append(current)
                run_active.append(value)
                current, value = 1, flag
        if current:
            run_lengths.append(current)
            run_active.append(value)
        silences = [l for l, on in zip(run_lengths, run_active) if not on]
        internal = [l for l, on in zip(run_lengths, run_active[:-1]) if not on]
        returns = sum(1 for i, on in enumerate(run_active[:-1]) if not on)
        rows.append({"n_silences": len(silences),
                     "n_internal_silences": len(internal),
                     "longest_silence": max(silences, default=0),
                     "n_returns_after_silence": returns,
                     "mean_silence": float(np.mean(silences)) if silences else 0.0,
                     "ended_silent": bool(len(row) and not row[-1])})
    stats = pd.DataFrame(rows, index=frame.index)
    enriched = frame.copy()
    for name in stats.columns:
        enriched[name] = stats[name]
    by_outcome = (enriched.groupby("final_result")
                  .agg(records=("id_student", "size"),
                       n_silences=("n_silences", "mean"),
                       n_internal_silences=("n_internal_silences", "mean"),
                       longest_silence=("longest_silence", "mean"),
                       n_returns=("n_returns_after_silence", "mean"),
                       mean_silence=("mean_silence", "mean"),
                       ended_silent=("ended_silent", "mean"))
                  .reset_index())
    return by_outcome, enriched


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
    submissions = pd.read_csv(root / "data" / "raw" / "oulad" / "studentAssessment.csv",
                              na_values=["?"])
    cohort = binary_cohort(master)
    lookup = cohort[KEY + ["due_day"]]
    events = daily.daily_events(root / "data" / "raw" / "oulad" / "studentVle.csv",
                               lookup, pre_days=14, post_days=post_days)
    post = daily.click_matrix(events, pd.MultiIndex.from_frame(frame[KEY]), 1, post_days)
    pairs = pairing.make_pairs(frame, 0.05, cfg["seed"])

    sfi = submission_fraction(master, submissions)
    sfi[KEY + ["sfi", "window_days", "sfi_flag_out_of_range"]].to_parquet(
        out / "e8_sfi_records.parquet", index=False)
    describe = (sfi[sfi["eligible"]].groupby("final_result")
                .agg(records=("id_student", "size"),
                     submissions=("sfi", "count"),
                     mean_sfi=("sfi", "mean"),
                     median_sfi=("sfi", "median"),
                     median_window_days=("window_days", "median"),
                     share_out_of_range=("sfi_flag_out_of_range", "mean"))
                .reset_index())
    describe.to_csv(out / "e8_sfi_by_outcome.csv", index=False)

    candidates, profile = delay_profiles(sfi, cfg["seed"])
    candidates.to_csv(out / "e8_profile_candidates.csv", index=False)
    profile.to_csv(out / "e8_profile_mix.csv", index=False)
    if len(profile):
        joined = frame[KEY].merge(
            sfi[KEY + ["sfi", "window_days"]], on=KEY, how="left", validate="one_to_one")
        # Rebuild the assignment so pair indices line up with `frame`.
        _, enriched = reengagement(post, frame)
        joined = joined.assign(profile_name=np.nan)
        data = sfi[sfi["sfi"].notna() & sfi["eligible"]]
        data = data[data["sfi"].between(0, 1.5)]
        if len(candidates):
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            x = StandardScaler().fit_transform(
                data[["sfi", "window_days"]].to_numpy(dtype=float))
            best = int(candidates.sort_values(["silhouette", "mean_ari"],
                                              ascending=False).iloc[0]["k"])
            labels = KMeans(n_clusters=best, n_init=20,
                            random_state=cfg["seed"]).fit_predict(x)
            order = pd.Series(labels).groupby(labels).apply(
                lambda s: data["sfi"].to_numpy()[s.index].mean()).sort_values().index.tolist()
            mapping = {old: new for new, old in enumerate(order)}
            data = data.assign(profile=np.array([mapping[l] for l in labels]))
            names = {0: "consistent_early", 1: "gradual_early", 2: "gradual_late",
                     3: "consistent_late", 4: "very_late"}
            data["profile_name"] = data["profile"].map(lambda i: names.get(i, f"p{i}"))
            joined = joined.drop(columns=["profile_name"]).merge(
                data[KEY + ["profile_name"]], on=KEY, how="left", validate="one_to_one")
        concordance = profile_pairs(joined, pairs)
        concordance.to_csv(out / "e8_pair_concordance.csv", index=False)

    by_outcome, enriched = reengagement(post, frame)
    by_outcome.to_csv(out / "e9_reengagement_by_outcome.csv", index=False)
    rows = []
    for name in ("n_silences", "n_internal_silences", "longest_silence",
                 "n_returns_after_silence", "mean_silence", "ended_silent"):
        fail = enriched.iloc[pairs["fail_index"].to_numpy()][name].to_numpy(dtype=float)
        passing = enriched.iloc[pairs["pass_index"].to_numpy()][name].to_numpy(dtype=float)
        rows.append({"feature": name, "pairs": int(len(pairs)),
                     "fail_mean": float(fail.mean()),
                     "pass_mean": float(passing.mean()),
                     "fail_minus_pass": float((fail - passing).mean())})
    pd.DataFrame(rows).to_csv(out / "e9_reengagement_pairs.csv", index=False)
    enriched[KEY + ["final_result", "n_silences", "n_internal_silences",
                    "longest_silence", "n_returns_after_silence", "mean_silence",
                    "ended_silent"]].to_parquet(
        out / "e9_reengagement_records.parquet", index=False)

    summary = {
        "sfi_records_with_submission": int(sfi["sfi"].notna().sum()),
        "sfi_mean_fail": float(describe.loc[describe["final_result"].eq("Fail"), "mean_sfi"].mean()),
        "sfi_mean_pass": float(describe.loc[describe["final_result"].isin(
            ["Pass", "Distinction"]), "mean_sfi"].mean()),
        "profiles": int(profile["profile_name"].nunique()) if len(profile) else 0,
        "same_profile_share": float(pd.read_csv(out / "e8_pair_concordance.csv")
                                    .query("profile_name == 'ALL'")["same_profile_share"].iloc[0])
        if (out / "e8_pair_concordance.csv").exists() else None,
        "returns_after_silence_fail_minus_pass": float(
            [r for r in rows if r["feature"] == "n_returns_after_silence"][0]["fail_minus_pass"]),
        "ended_silent_fail_minus_pass": float(
            [r for r in rows if r["feature"] == "ended_silent"][0]["fail_minus_pass"]),
    }
    (out / "summary_delay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
