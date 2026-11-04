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
