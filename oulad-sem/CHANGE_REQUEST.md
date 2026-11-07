# Change Request — cohort config consistency (WP0.4)

- **Date:** 2026-11-04
- **Baseline run:** `run-20261103T194449Z-838d99`
- **Reason:** `config/config.yaml` declared `cohort.at_risk_at_cutoff: true`, but the
  executed S02 cohort audit kept all T1 enrolments and treated the at-risk filter as a
  sensitivity analysis. The revision guide WP0.4 requires the config, README, and Report
  to describe the same cohort.

## Config change

Only the `cohort` block changed:

```diff
-  at_risk_at_cutoff: true      # drop enrolments with date_unregistration <= cutoff
+  at_risk_at_cutoff: false     # main cohort keeps all T1
+  at_risk_sensitivity: true    # at-risk filter is a sensitivity only
```

- **Old config SHA-256:** `6423242c34b2d24afcac4d5a269e8293f6b10115d07386bc4c8ff0b9367afd14`
- **New config SHA-256:** `1719aea81a6c72eeb4b88626151fc6113f76dea40cf3629b25735231a1003a90`
- No modelling, split, window, or feature parameters changed.

## Run impact

The change requires fresh run IDs; no baseline run was modified in place.

- Simulation validation: `sim-20261104T025730Z-f02b6d` (S00–S12 pass;
  `validate_run.py` 0 errors / 0 warnings).
- Full run: `run-20261104T030052Z-b96dee` (S00–S12; see `FIX_REPORT.md`).
- S02 now logs `at_risk_at_cutoff=False, at_risk_sensitivity=True` and the check
  “main cohort keeps all T1 ... at-risk cohort is a sensitivity only”.

## Effect

The main T1 cohort is unchanged at 22,437 enrolments because the previous pipeline
already behaved this way. The change makes the frozen configuration, Report, and README
consistent with the audited behaviour. The at-risk cohort remains available as a
sensitivity analysis in S09/S11 outputs.

---

# Change Request — T1 positive class direction (P5)

- **Date:** 2026-11-08
- **Baseline run:** `run-20261104T045841Z-4f3088` (Pass/Distinction = 1)
- **Reason:** With Pass/Distinction as the positive class, 68.6% of T1 is
  positive, so the no-skill PR-AUC is 0.686 and the reported 0.827 is only a
  +0.14 lift; every model's F1 (0.714–0.785) is below the 0.807 F1 of predicting
  every enrolment as Pass, because the configured precision target 0.80 is never
  reached and the threshold silently falls back to maximum F1. The research
  question is early warning, so the at-risk outcome should be the positive class.

## Config change

Only the `cohort` block changed:

```diff
+  positive_class: fail         # T1: Fail=1, Pass/Distinction=0 (early-warning direction)
```

- **Old config SHA-256:** `1719aea81a6c72eeb4b88626151fc6113f76dea40cf3629b25735231a1003a90`
- The authorship and reporting fixes are in separate commits.
- **New config SHA-256:** `6dd89b411646871c0e3623a4fc7d22aac983cf795f15fb3fca2de7bf07ae0043`

## Code changes required by the label direction

- `src/cohort.py` adds `is_pass` (Pass/Distinction = 1, Fail = 0, Withdrawn =
  missing), which does not depend on the label direction, and derives `y_t1` /
  `y_t2` from `positive_class`.
- Matching (`src/matching.py`, `stages/s06_matching.py`, `stages/s11_robustness.py`)
  now uses `is_pass` as the treated indicator, so the CEM estimand and the sign of
  the W2 differences are unchanged.
- S08 records the threshold rule actually used plus two reference points
  (`no_skill_pr_auc`, `all_positive_f1`) in `t08_model_metrics.csv`.

## Run impact

- Simulation validation: `sim-20261107T172956Z-d470d3` (S00–S12 pass;
  `validate_run.py` 0 errors / 0 warnings).
- Full run: `run-20261107T173321Z-4deefb` (S00–S12 pass;
  `validate_run.py` 0 errors / 0 warnings).
- ROC-AUC is direction-invariant: every model's main-test ROC-AUC matches the
  baseline run to within 0.0002.
- PR-AUC falls because the no-skill value falls: T1 Fail rate is 0.314, so M0
  PR-AUC is 0.529 against a 0.323 no-skill floor.

## Effect

No modelling, split, window or feature parameter changed. Only the direction of
the label and the bookkeeping that used the label as a group indicator changed.
