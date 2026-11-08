"""First positive-weight TMA anchor, eligibility, and as-of-cutoff submission.

Ported from ``oulad_analysis/assessment_node.py`` (``_first_weighted_tma``,
``build_node_features``) and ``oulad_analysis/tma_submission_timing.py``
(``add_submission_distance``).  The observable definition is unchanged:
the anchor is the earliest TMA with a strictly positive weight in each
module presentation, and every submission field is masked to the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
BATCH = KEY[:2]


def first_weighted_tma(assessments: pd.DataFrame, assessment_type: str = "TMA",
                       min_weight_exclusive: float = 0.0) -> pd.DataFrame:
    """Earliest assessment of ``assessment_type`` with weight above the threshold.

    Mirrors ``_first_weighted_tma``: filter the assessment type and
    ``weight > min_weight_exclusive``, then keep the earliest date per
    (module, presentation).  GGG has no such row and drops out.
    """
    a = assessments.copy()
    a = a[a["assessment_type"].eq(assessment_type)].copy()
    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    a["weight"] = pd.to_numeric(a["weight"], errors="coerce")
    a = a[a["weight"].gt(min_weight_exclusive) & a["date"].notna()].copy()
    a = a.sort_values(BATCH + ["date", "id_assessment"])
    first = a.drop_duplicates(BATCH)
    if first.duplicated(BATCH).any() or first["date"].isna().any():
        raise ValueError("Invalid first weighted TMA definition")
    return first[BATCH + ["id_assessment", "date"]].rename(columns={"date": "due_day"})


def build_tma_cohort(info: pd.DataFrame, registration: pd.DataFrame,
                     courses: pd.DataFrame, assessments: pd.DataFrame,
                     submissions: pd.DataFrame, *, post_days: int = 14,
                     assessment_type: str = "TMA",
                     min_weight_exclusive: float = 0.0,
                     require_final_result: bool = True) -> tuple[pd.DataFrame, dict]:
    """Enrolment table with anchor, masked submission state and eligibility flag.

    Steps, in the order the protocol lists them: join registration and module
    length, attach the anchor, merge the anchor submission (so a banked attempt
    can be excluded), then evaluate eligibility up to ``due_day + post_days``.
    Withdrawn records stay in the frame and are excluded by
    :func:`binary_cohort` only when a label is actually needed.
    """
    info = info.copy()
    registration = registration.copy()
    if info.duplicated(KEY).any() or registration.duplicated(KEY).any():
        raise ValueError("studentInfo/studentRegistration key is not unique")
    frame = info.merge(registration, on=KEY, how="left", validate="one_to_one")
    frame = frame.merge(courses, on=BATCH, how="left", validate="many_to_one")
    anchor = first_weighted_tma(assessments, assessment_type, min_weight_exclusive)
    frame = frame.merge(anchor, on=BATCH, how="left", validate="many_to_one")

    sub = submissions[["id_assessment", "id_student", "date_submitted", "is_banked", "score"]]
    if sub.duplicated(["id_assessment", "id_student"]).any():
        raise ValueError("Duplicate assessment submission key")
    # Presentations without an anchor carry a missing id_assessment; an
    # index join leaves them unmatched instead of forming a NaN key group.
    frame = frame.join(sub.set_index(["id_assessment", "id_student"]),
                       on=["id_assessment", "id_student"], how="left")
    frame["banked_attempt"] = frame["is_banked"].fillna(0).eq(1).astype(int)

    cutoff = frame["due_day"] + post_days
    observed = frame["date_submitted"].where(frame["date_submitted"].le(cutoff))
    relative = observed - frame["due_day"]
    frame["cutoff_day"] = cutoff
    frame["submitted_by_cutoff"] = observed.notna().astype(int)
    frame["submitted_after_due"] = relative.gt(0).astype(int)
    frame["relative_submission_day"] = relative
    frame["days_early"] = (-relative).clip(lower=0)
    frame["days_late"] = relative.clip(lower=0)
    frame["has_score_by_cutoff"] = (frame["submitted_by_cutoff"].eq(1)
                                    & frame["score"].notna()).astype(int)

    registered = (frame["due_day"].notna() & frame["date_registration"].notna()
                  & frame["date_registration"].le(frame["due_day"]))
    alive = frame["date_unregistration"].isna() | frame["date_unregistration"].gt(cutoff)
    eligible = registered & alive & frame["banked_attempt"].eq(0)
    if require_final_result:
        eligible &= frame["final_result"].isin(["Pass", "Distinction", "Fail", "Withdrawn"])
    frame["eligible"] = eligible.fillna(False)

    # Descriptive outcome flags used by the withdrawal audit and S10.
    frame["is_withdrawn"] = frame["final_result"].eq("Withdrawn").astype(int)
    frame["y_t3"] = np.where(frame["final_result"].isin(["Pass", "Distinction"]), "Pass",
                             np.where(frame["final_result"].eq("Fail"), "Fail", "Withdrawn"))

    stats = {
        "source_rows": int(len(info)),
        "anchor_presentations": int(len(anchor)),
        "with_anchor": int(frame["due_day"].notna().sum()),
        "registered_by_deadline": int(registered.sum()),
        "alive_at_cutoff": int((registered & alive).sum()),
        "banked_excluded": int((registered & alive & frame["banked_attempt"].eq(1)).sum()),
        "eligible": int(frame["eligible"].sum()),
        "banked_in_attempt": int(frame["banked_attempt"].sum()),
        "unknown_registration": int(frame["date_registration"].isna().sum()),
        "unknown_final_result": int(frame["final_result"].isna().sum()),
        "withdrawn_total": int(frame["final_result"].eq("Withdrawn").sum()),
    }
    # The banked flag is excluded through eligibility; drop raw submission data.
    frame = frame.drop(columns=["date_submitted", "is_banked"])
    return frame.reset_index(drop=True), stats


def binary_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Eligible Fail vs Pass/Distinction records with Fail = 1."""
    out = frame[frame["eligible"] & frame["final_result"].isin(
        ["Pass", "Distinction", "Fail"])].copy()
    if out.empty:
        raise ValueError("No binary records after eligibility filtering")
    out["y_fail"] = out["final_result"].eq("Fail").astype(int)
    return out.reset_index(drop=True)


def eligibility_steps(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordered sample-flow table used by S02 and the report."""
    steps = [("all_enrolments", int(len(frame)))]
    have = frame["due_day"].notna()
    steps.append(("has_first_weighted_tma", int(have.sum())))
    registered = have & frame["date_registration"].notna() & \
        frame["date_registration"].le(frame["due_day"])
    steps.append(("registered_by_deadline", int(registered.sum())))
    cutoff = frame["cutoff_day"]
    alive = frame["date_unregistration"].isna() | frame["date_unregistration"].gt(cutoff)
    steps.append(("registered_at_cutoff", int((registered & alive).sum())))
    steps.append(("banked_excluded", int((registered & alive & frame["banked_attempt"].eq(1)).sum())))
    known = frame["final_result"].notna()
    steps.append(("known_final_result", int((registered & alive & frame["banked_attempt"].eq(0)
                                             & known).sum())))
    steps.append(("withdrawn_at_cutoff", int((frame["eligible"]
                                              & frame["final_result"].eq("Withdrawn")).sum())))
    steps.append(("binary_records", int((frame["eligible"]
                                         & frame["final_result"].isin(
                                             ["Pass", "Distinction", "Fail"])).sum())))
    steps.append(("fail_records", int((frame["eligible"]
                                       & frame["final_result"].eq("Fail")).sum())))
    return pd.DataFrame(steps, columns=["step", "records"])


def presentation_anchor_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per module presentation: anchor date, cohort size and outcome mix."""
    rows = []
    for (module, presentation), group in frame[frame["due_day"].notna()].groupby(BATCH):
        eligible = group[group["eligible"]]
        rows.append({
            "code_module": module,
            "code_presentation": presentation,
            "due_day": int(group["due_day"].iloc[0]),
            "cutoff_day": int(group["cutoff_day"].iloc[0]),
            "module_length": int(group["module_presentation_length"].iloc[0])
            if "module_presentation_length" in group and group["module_presentation_length"].notna().any() else -1,
            "enrolments": int(len(group)),
            "eligible": int(len(eligible)),
            "fail": int(eligible["final_result"].eq("Fail").sum()),
            "pass_distinction": int(eligible["final_result"].isin(["Pass", "Distinction"]).sum()),
            "withdrawn": int(eligible["final_result"].eq("Withdrawn").sum()),
        })
    return pd.DataFrame(rows).sort_values(BATCH).reset_index(drop=True)
