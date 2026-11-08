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

---

# CR-2026-11-08-a — TMA two-week design replaces the day 0–27 design

**Date:** 2026-11-08
**Decision owner:** project user; recorded here for the frozen-config audit trail.

## Reason

The research question was fixed as the two-week TMA design
(`research/TMA_PRIMARY_PROTOCOL.md`): within the same module presentation and
among Fail / Pass-Distinction records with comparable **post-deadline** clicks,
which submission and post-deadline persistence patterns differ. The original
day-0–27 question becomes the historical baseline.

## Old design

- Anchor: presentation start (day 0). Windows W1 = 0–27, W2 = 28–55 (or
  `first_tma - 1` when earlier).
- Cohort: T1 = all enrolments with a known outcome, Withdrawn excluded from the
  binary label; no post-deadline eligibility filter.
- Matching: CEM on click/active-day/entropy bins within module presentation,
  treated group = Pass/Distinction.
- Models: M0–M5 ladder on day-0–27 features; M4 adds scores as an upper bound.

## New design

- Anchor: `d` = deadline of the first `assessment_type = TMA` with `weight > 0`
  in each presentation (19 presentations, modules AAA–FFF).
- Windows: `d-13..d` and `d+1..d+14`.
- Cohort: registered by `d`, still registered at `d+14`, not a banked attempt;
  binary label Fail = 1 vs Pass/Distinction = 0; Withdrawn reported separately
  and analysed only in the S10 risk set.
- Matching: 1:1 within presentation on `post_clicks`, tolerance 5%
  (0% and 10% sensitivity), zero clicks match zero clicks only.
- Primary metrics: the six indicators in `study.primary_metrics`, Holm-adjusted.
- Models: M2 (`M1 + days_early/days_late`) is the primary model; M4 adds the
  first-TMA score and is a retrospective upper bound.
- Splits: grouped 60/20/20, grouped 5-fold, 2013→2014 temporal, leave-one-module-out,
  plus a label-permutation control.

## Affected stages

S02 (cohort), S03 (description), S04 (features), S05 (splits unchanged in shape,
new cohort), S06 (pairing), S07 (clustering), S08 (models), S09 (cutoff curve),
S10 (risk set), S11 (robustness), S12 (report, poster, README, ZIP).
S00/S01 keep their role (acquisition and raw audit).

## Frozen config

- Previous SHA-256: `6dd89b411646871c0e3623a4fc7d22aac983cf795f15fb3fca2de7bf07ae0043`
- **New SHA-256:** see `config/config.sha256` after this commit.
- Config keys removed: `cohort.target`, `cohort.at_risk_*`, `cohort.min_w1_clicks`,
  `windows.*`, `matching.*` (CEM), `models.recall_precision_target`,
  `models.include_scores_model`.
- Config keys added: `study.*`, `cohort.min_post_clicks`,
  `models.primary_model`, `models.upper_bound_model`, `models.rf_models`,
  `models.threshold_rule`, `models.precision_target`, `models.label_permutations`.

## Run impact

All previous run ids (`run-20261104T045841Z-4f3088`,
`run-20261107T173321Z-4deefb`, `sim-20261107T172956Z-d470d3`,
`sim-20261108T083219Z-9f369f`) belong to the day-0–27 design and are archived
in `results/oulad_sem_day0_27_baseline_20261108/` where relevant. The TMA design
starts with a new run id; no number from the archived run may be compared
directly with a TMA number.

---

# CR-2026-11-08-b — Exploratory blocks C1–C4 behind a config flag

**Date:** 2026-11-08

## Reason

The third work segment of the merge plan asks for four exploratory checks that
were deferred until the frozen primary design was delivered: pre-entry
background balance of the matched pairs (C1), re-matching uncertainty and
per-presentation heterogeneity (C2), the non-submitting subgroup (C3), and an
at-risk warning set that includes later Withdrawn records (C4).

## What changes

- New `exploration:` block in `config/config.yaml` with five booleans and one
  replicate cap. Every block can be switched off without touching the primary
  design.
- `src/tma/exploration.py` holds the implementations; S11 calls them after the
  robustness curve and writes `t11_x_*` tables.
- S12 adds an appendix section that reports the four blocks.
- No primary indicator, primary model, split, window or matching parameter
  changes.

## Affected stages

S11 (new exploratory steps) and S12 (appendix text and tables). S02–S10 are
unchanged.

## Frozen config

- Previous SHA-256: `77ae257566a39c0c89266ca59cc4f896f9ce429611b3e1e4cb63129c8027f1c8`
- **New SHA-256:** see `config/config.sha256`.

## Run impact

New run ids are required because the config hash changes. The primary tables
(`t06_primary_holm.csv`, `t08_model_metrics.csv`, `t08_parity.csv`) are expected
to be identical to the previous run: the new blocks only read tables that the
earlier stages produced and append new ones.
