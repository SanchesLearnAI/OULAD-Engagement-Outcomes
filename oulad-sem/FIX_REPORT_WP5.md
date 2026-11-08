# FIX_REPORT_WP5 — post-revision diagnostics

- **Date:** 2026-11-04
- **Baseline (read-only):** `run-20261104T031722Z-054c79`
- **New clean simulation:** `sim-20261104T045540Z-16a462`
- **New full run:** `run-20261104T045841Z-4f3088`
- **Frozen config:** `config/config.yaml`, SHA-256
  `1719aea81a6c72eeb4b88626151fc6113f76dea40cf3629b25735231a1003a90`
- **Scope:** WP5 only. The W1/W2 feature definitions, splits and main model
  ladder were not changed; the strict main-model metrics are unchanged.

## 1. WP5.1 — context increment with paired CI

### Change

`stages/s08_models.py` now stores the M0c and M1c held-out predictions and
runs a paired bootstrap on the **same 4,492 test rows**:

- 1,000 row resamples with a recorded seed (`79` in these runs);
- `delta_pr_auc = PR_AUC(M1c) - PR_AUC(M0c)`;
- `delta_roc_auc = ROC_AUC(M1c) - ROC_AUC(M0c)`;
- 95% percentile intervals;
- output `t08_context_delta.csv` with the required columns:
  `comparison, delta_pr_auc, ci_low, ci_high, delta_roc_auc, roc_ci_low,
  roc_ci_high, n_test, seed`.

The stage check now requires all 1,000 replicates to be retained.

### Result

**Full run:**

- ΔPR-AUC = **0.001280**, 95% CI **[-0.004213, 0.006432]** — includes zero.
- ΔROC-AUC = **0.005463**, 95% CI **[0.000606, 0.009930]** — excludes zero,
  but the difference is very small.
- n_test = 4,492; seed = 79.

**Simulation:**

- ΔPR-AUC = 0.028601, 95% CI [-0.022473, 0.086673] — includes zero.
- ΔROC-AUC = 0.095282, 95% CI [0.012894, 0.172106].

The report now uses the direction **M1c − M0c** consistently in §4.3, Table 5
caption and §5. For the full run it says there is **no detectable behavioural
increment beyond context on PR-AUC**, while noting that the ROC-AUC difference
is small and its CI excludes zero. It does not claim a context-adjusted effect.
The Table 5 caption labels the intervals as 95% CIs and states that the PR-AUC
interval includes zero. The simulation and full text are mode-aware.

## 2. WP5.2 — B2 removed from the main updating table

### Change

- Table 7 now contains only `W1_only_M3`, `W2_only`, `All_features_c56` and
  `B1_stacking`.
- B2 is reported only in Appendix Table 16 and in `t09_b2_failed.csv`.
- The main text says B2 is excluded as a failed specification and does not
  present its numbers.
- No “3 of 4 methods” wording remains; the valid-method count is 3 of 3.
- The poster no longer reports B2 numbers.

### Result

Full B2 values (unchanged by the revision): PR-AUC **0.7591**, Δ vs W1
**-0.0692** (95% CI [-0.0831, -0.0539]), Brier **0.3041**, calibration slope
**0.0509**, `status = failed_calibration`. B2 is not evidence about sequential
updating.

## 3. WP5.3 — diagnostics and placebo checks

### Change

`stages/s11_robustness.py` now writes a `role` column in
`t11_negative_controls.csv`:

| diagnostic | role |
|---|---|
| `random_labels` | `leak_check` |
| `pre_start_only` | `positive_control` |
| `placebo_window_56_83` | `placebo_failure` |
| `num_of_prev_attempts_gt0` | `negative_control_outcome` |

It also writes `t11_placebo_exposure.csv` (`exposure, metric, value, n, note`)
for the W2−W1 change in non-assessment resource types (`homepage`, `forumng`,
`resource`, `subpage`, `oucontent`, `url`, `page`, `oucollaborate`, `ouwiki`,
`dataplus`, `ouelluminate`, `glossary`, `dualpane`, `repeatactivity`,
`htmlactivity`, `sharedsubpage`, `folder`).

Report §4.6 is renamed **“Diagnostics and placebo checks”**. Table 9 now has
`Role` and `Interpretation` columns and includes the repeater and
non-assessment-exposure diagnostics. The prose says these diagnostics are not
clean nulls and do not prove the absence of leakage.

### Result

**Full run:**

- random-label ROC-AUC **0.530** — leakage sanity check, close to chance.
- pre-start-only PR-AUC **0.782** — positive control; above no-skill, not a null.
- future-window PR-AUC **0.861** — post-outcome persistence; placebo failure,
  not a leak test.
- repeater negative-control outcome ROC-AUC **0.657**.
- non-assessment W2 change: PR-AUC **0.752**, ROC-AUC **0.593**, n=4,503.
  If predictive, this indicates generic engagement/background rather than a
  learning-strategy signal.

The simulation shows the same qualitative pattern: random-label 0.504,
pre-start 0.824, future-window 0.911, repeater 0.631, non-assessment exposure
PR-AUC 0.715 / ROC-AUC 0.563.

## 4. WP5.4 — TMA decision

TMA analyses were **not attempted**. This remains documented in Report §5 and
README:

- TMA+21/+28 is an alternative estimand that overlaps W1/W2 and conditions on
  continued enrolment;
- TMA+14 uses a presentation-dependent anchor, not a fixed course-day cutoff;
- first-TMA scores are not available at the early cutoff, so score matching is
  only a post-hoc upper bound.

They are not in the Abstract, Introduction, Conclusion, or main results.

## 5. Leftover audit fixes

### Stale static-Cox table

- Deleted `outputs/tables/t10_hazard_ratios.csv` (old static W2 Cox with future
  features) and the stale simulation copy.
- `stages/s12_package.py` excludes that filename from the ZIP.
- S12 now checks that the ZIP has no `t10_hazard_ratios.csv`.

### Non-current run validation

`src/validate.py` now distinguishes the current run from older runs:

- current run: root `STATUS.json`, root artifacts, citation scan as before;
- non-current run: run-local `runs/<id>/STATUS.final.json`,
  `runs/<id>/manifest_all.json`, and only run-scoped artifact hashes;
  shared root deliverables (README/REPORT/ZIP) are not falsely compared.

This makes the clean simulation validate after the full run has replaced the
shared root deliverables. Tested on the previous full run and the previous
simulation: both return 0 errors / 0 warnings when non-current.

### S12 packaging checks

S12 now also asserts:

- no stale `t10_hazard_ratios.csv` in the ZIP;
- `t08_context_delta.csv`, `t09_b2_failed.csv`, and
  `t11_placebo_exposure.csv` are in the ZIP;
- exactly one `Group_*_Project.zip` exists (`Group_09_Project.zip`).

## 6. Additional report fixes from the independent audit

- Simulation Table 1 and §4.1 now use the 500-enrolment sample counts
  (500 enrolments, T1=350, T3 Pass/Dist=237, Fail=113) instead of mixing
  full-data and sample counts. The poster’s cohort box uses the same sample
  counts.
- Context direction and CI wording corrected (M1c−M0c; PR CI includes zero;
  ROC CI excludes zero but is small).
- Table 9 is referenced from §4.6, so it is not orphaned.
- Matching-method wording changed to “contrasts”; the report notes that the
  magnitude differs across methods and, in the simulation, the sign is not
  stable. It is called a descriptive sensitivity, not a directional effect.
- Repeater diagnostic label aligned across Tables 9 and 15; Table 10 no longer
  calls it a mitigation.
- The row-level leakage claim is qualified: the full-data design argument is
  distinguished from this rehearsal, which has zero student overlap.
- The landmark appendix now states that coefficient-level HR/CI are in
  `t10_landmark_cox.csv`; the table itself reports test concordance only.

## 7. Before / after

| Item | Baseline full `031722` | New full `4f3088` |
|---|---:|---:|
| M1 main PR-AUC / ROC-AUC | .8267 / .7236 | .8267 / .7236 |
| M3 main PR-AUC / ROC-AUC | .8283 / .7294 | .8283 / .7294 |
| Context M1c−M0c PR-AUC | not estimated | .00128 [-.00421, .00643] |
| Context M1c−M0c ROC-AUC | not estimated | .00546 [.00061, .00993] |
| Table 7 B2 row | present | removed; Appendix Table 16 |
| B2 status | failed in text | `failed_calibration` in appendix/file |
| `t11_negative_controls.csv` role | absent | `role` column present |
| `t11_placebo_exposure.csv` | absent | present (PR .752 / ROC .593) |
| Report pages / main pages | 16 / 9 | 16 / 9 |
| ZIP static-Cox hazard table | present | absent |
| WP5 context/exposure/B2 tables in ZIP | absent | present |

The main model and split metrics are unchanged; this is a diagnostics and
reporting revision, not a model-tuning revision.

## 8. Validation

| Run | Mode | Stages | validate_run.py | Report pages |
|---|---|---|---:|---:|
| `sim-20261104T045540Z-16a462` | simulation | 13/13 pass | 0 errors / 0 warnings | 16 total / 8 main |
| `run-20261104T045841Z-4f3088` | full | 13/13 pass | 0 errors / 0 warnings | 16 total / 9 main |

- `pytest -q`: **9 passed**.
- Final full ZIP: only `Group_09_Project.zip`; contains
  `runs/run-20261104T045841Z-4f3088/STATUS.final.json`, Report, README,
  DATA_DOWNLOAD, code, tables and figures; no raw data, no Poster.pdf, no
  `t10_hazard_ratios.csv`.
- S12 checks include `zip_no_stale_hazard`, `zip_has_wp5_tables`,
  `zip_has_status_final`, `zip_poster_excluded`, and `single_submission_zip`.

## 9. Remaining caveats

- The context comparison is observational and the context model is imperfect.
  The full-run PR-AUC interval includes zero; the ROC-AUC interval excludes zero
  only by a very small margin. The point estimate is small; it is not a causal
  effect.
- B2 remains a failed calibration specification and is not used as evidence.
- Pre-start-only and future-window diagnostics are not clean nulls; the
  random-label check is close to chance.
- TMA analyses remain not attempted; reasons are above.
- The 6–10-page check follows the report’s declared main-text end (`endmain`);
  some floats may appear after the main prose, so this is a prose-level page
  count rather than a strict float-inclusive check.
- The landmark appendix table reports test concordance; coefficient-level
  HR/CI are in `t10_landmark_cox.csv` rather than the compact report table.
- The simulation is a 500-enrolment rehearsal; the full run uses all 32,593
  enrolments, but the held-out test sets and subgroup cells remain finite.
