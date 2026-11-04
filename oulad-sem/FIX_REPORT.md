# FIX_REPORT — OULAD revision

- **Date:** 2026-11-04
- **Baseline run (read-only):** `run-20261103T194449Z-838d99`
- **Clean simulation gate:** `sim-20261104T033345Z-59a902`
- **Final full run:** `run-20261104T031722Z-054c79`
- **Frozen config:** `config/config.yaml`, SHA-256
  `1719aea81a6c72eeb4b88626151fc6113f76dea40cf3629b25735231a1003a90`
- **Config change request:** `CHANGE_REQUEST.md`

All reported numbers below are copied from the audited run outputs; none were
entered by hand. Baseline numbers are from `run-20261103T194449Z-838d99` (or its
archived outputs in `04_archive/pre_revision_20261104/`). The revision is a
correctness and transparency pass, not a tuning pass: the strict main-model
metrics are unchanged.

## 1. WP0 — correctness fixes

### WP0.1 S10 Cox used future W2 features

**Problem.** The baseline appendix fitted a static cause-specific Cox model with
`w1_*`, `w2_*`, and `d_clicks`; a Withdrawn event in week 10 therefore used
features from weeks 28–55. The baseline concordances
(.6637 Fail / .5626 Pass / .7703 Withdrawn) were in-sample and not usable as
early-warning evidence.

**Change.** Replaced the static fit with landmark cause-specific Cox models at
`L = 28` and `L = 56` in `stages/s10_survival.py`:

- cohort restricted to `date_unregistration` missing or after `L`, and events
  after `L`;
- L=28 uses W1 features only; L=56 uses W1+W2 features available at day 56;
- fit on the training split, report held-out test concordance;
- `t10_landmark_cox.csv` contains coefficients, HR, 95% CI, p-values, event
  counts, and test concordance;
- checks `cox_no_future_features`, `cox_landmark_cohort`, `survival_available`.

**After (held-out test concordance, final full run).**

| Landmark | Cause | Test C |
|---|---|---:|
| 28 | Fail | 0.662 |
| 28 | Pass | 0.560 |
| 28 | Withdrawn | 0.572 |
| 56 | Fail | 0.660 |
| 56 | Pass | 0.582 |
| 56 | Withdrawn | 0.557 |

The report appendix now carries the landmark table and explicitly says the
hazard ratios are not interpreted causally.

### WP0.2 B2 offset was treated as a valid comparison

**Problem.** Baseline B2 had PR-AUC .7591, Brier .3041, calibration slope .0509,
but appeared as an ordinary row in the updating table; the text used “3 of 4
methods”.

**Change.** `stages/s09_sequential.py` now tries the no-constant and
constant-offset GLM fits, selects the lower-Brier fit, and writes a `status`
column (`ok` / `failed_calibration`). B2 remains `failed_calibration` in the
final run. Table 7, its caption, the §4.4 prose, and the poster all report B2
separately as a failed specification; the “3 of 4 methods” wording is gone
(the final text counts only valid methods: 3 of 3).

**Before / after B2.** PR-AUC .7591 / .7591; Brier .3041 / .3041; calibration
slope .0509 / .0509; status absent / `failed_calibration`. The attempted
correction did not rescue B2; it is now explicitly excluded from the updating
evidence.

### WP0.3 `pre_start=feature` specification was a duplicate

**Problem.** Baseline `P000_baseline` and `P009_pre_feature` were both
`0.8283198449` because `feature_columns()` never selected
`pre_start_clicks` / `pre_start_active_days`.

**Change.** `src/features.py` adds the two pre-start columns only when
`include_pre_start=True`; `stages/s11_robustness.py` passes that flag only for
`pre == "feature"`. The 45-row specification table and 37 predictive
specifications are unchanged in count.

**Before / after (full data).** P009 .8283198449 → .8288507795; P000 remains
.8283198449. The full-data change is small but the specifications are no longer
identical. In the simulation the change is larger (.874653 → .851969), showing
that the previous equality was a bug rather than a null result.

### WP0.4 Config contradicted the main cohort

**Problem.** `config.yaml` said `at_risk_at_cutoff: true`, while S02 kept all T1
enrolments and treated at-risk filtering as a sensitivity analysis. The report
Table 1 then mixed at-risk counts with a full-cohort narrative.

**Change.** `config/config.yaml` now sets
`at_risk_at_cutoff: false` and `at_risk_sensitivity: true`;
`CHANGE_REQUEST.md` records the old/new hashes and the new run IDs;
`stages/s02_master.py` logs both flags and the audit; `src/reporting.py` Table 1
now reports full T1 n = 22,437 and full T3 Pass/Dist = 15,385, Fail = 7,052,
with Withdrawn observed by cutoff. S02 checks in the final run show
`at_risk_at_cutoff=False, at_risk_sensitivity=True` and
“main cohort keeps all T1”.

## 2. WP1 — scientific additions

### WP1.1 Trajectory divergence

`stages/s06_matching.py` adds weekly matched trajectories (weeks 0–8), weighted
log1p means, normalized shares, and cluster-bootstrap CIs over matched strata
(1,000 reps). Outputs: `t12_trajectory_weekly.csv`, `t12_divergence.csv`,
`f12_trajectory_divergence.png`.

Final full run: first stable divergence (two consecutive weeks with positive
difference CI) is **week 3**; week 3 difference .139 [.093, .183]. Because week 3
lies inside W1, the report explicitly states that the matched groups were
already diverging during the baseline window, so the “same early engagement”
premise is approximate, not exact. The simulation showed no stable divergence;
that difference is reported rather than hidden.

### WP1.2 Context-only and context-adjusted models

`stages/s08_models.py` adds M-1 (demographics, module/presentation, previous
attempts, credits), M0c (M-1 + M0 features), and M1c (M-1 + M1 features), all on
the same split and held-out test. Outputs: `t08_context_metrics.csv`,
`f08_context_comparison.png`.

Final full run:

| Model | PR-AUC | ROC-AUC | Calibration slope |
|---|---:|---:|---:|
| M-1 context | 0.783 | 0.657 | 0.906 |
| M0c context+M0 | 0.853 | 0.761 | 1.000 |
| M1c context+M1 | 0.854 | 0.766 | 0.957 |

The M1c−M0c difference is **0.001 PR-AUC**; the report says this is small
relative to the context-only signal and does not claim a context-adjusted
effect. The simulation context-only model was near chance, and the report uses
conditional wording for that case.

### WP1.3 Module and subgroup metrics

`stages/s08_models.py` adds per-module and per-presentation metrics plus
gender, age, IMD, disability, and education subgroups; outputs
`t08_module_metrics.csv`, `t08_subgroup_metrics.csv`, `f08_module_heatmap.png`,
`f08_subgroup_calibration.png`. Groups with fewer than five test rows, a single
outcome class, or missing values are omitted and the table captions say so.

Final full run: module PR-AUC ranges .778–.964; the lowest is GGG (.778);
subgroup calibration slopes range .571–1.933 and recall .653–.929; the lowest
reported subgroup recall is Post Graduate Qualification (.653, n=55, low-n).
The report labels these as performance differences, not causal effects.

### WP1.4 Measurement sensitivity

`stages/s11_robustness.py` adds:

- quiz-site inclusion: M1/M3 PR-AUC changes −0.0008/−0.0005;
- exposure-definition CEM: W2 click differences 36.161 (total clicks), 25.168
  (active days), −27.621 (entropy), 37.558 (all three);
- seven matching-bin grids: differences 34.451 to 39.814;
- placebo outcome (repeater status): ROC-AUC .657.

Outputs: `t11_quiz_sensitivity.csv`, `t11_exposure_sensitivity.csv`,
`t11_matching_bins.csv`, `t11_placebo_outcomes.csv`. The report says the
matching conclusion depends on the exposure definition and reports the range
rather than selecting a favourable definition.

### WP1.5 E-value and negative-control outcome

The matched E-value is 1.555; the report says an unmeasured confounder with
risk ratio about 1.555 could explain the contrast and explicitly calls the
matched result fragile. The repeater placebo is reported as a diagnostic that
behaviour features may encode background, not as causal evidence. Negative
controls are in `t11_negative_controls.csv`; pre-start-only PR-AUC is .782 and
is described as a **strong signal, not a weak control**; the future-window
control is .861 and is described as a placebo failure, not a clean null;
random-label ROC-AUC is .530.

### WP1.6 PCA and silhouette

`stages/s07_clustering.py` adds `f07_pca_scatter.png` (PC1×PC2, 5,000-point
sample, explained-variance annotation) and `f07_silhouette.png`. The report says
these are diagnostics, not evidence that clusters are real. Final clustering:
k=2, mean bootstrap ARI .913, minimum Jaccard .810; k=3 has a higher silhouette
(.248 vs .235), which the report states.

## 3. WP2 — Report, Poster, README, ZIP

`src/reporting.py` was updated and S12 regenerated all deliverables:

- Table 1 now reports full-cohort counts and no longer labels the main cohort
  “at risk”; the caption explains that T3 Withdrawn is observed by cutoff.
- Added context, module, subgroup, trajectory, landmark, sensitivity, and
  limitations tables; added the E-value, trajectory, context, module/subgroup,
  quiz/exposure/bin/placebo, and landmark prose; added PCA/silhouette appendix
  figures.
- B2 is explicitly failed and excluded from the valid-method count.
- Pre-start-only is no longer called weak; subgroup omissions and low-n groups
  are disclosed; page-count note is generated with `\pageref{endmain}`.
- References use non-empty `\bibitem` keys; the Wu et al. qualifier remains.
- Table widths were reduced for the full-cohort Table 1, negative-control table,
  and subgroup table; the final report compiles without overfull boxes.
- Poster: context, trajectory, PCA, robustness, and reproducibility blocks were
  added; the lower half is filled; no text overflow; “associations, not causal”
  is retained.
- README: `pip install -r REQUIREMENTS.lock`, reproducibility checklist,
  `make sim/full/validate/test`, run-ID and config-hash guidance.
- ZIP: only `Group_09_Project.zip`; contains `STATUS.final.json`, code, tables,
  figures, Report, README, DATA_DOWNLOAD; excludes raw data and Poster.pdf.

Final report: **16 pages total, 9 main-text pages**; references, disclosures,
and appendix are excluded from the main count. Final poster compiles; no
overfull box reported.

## 3b. Changed files

- Config: `config/config.yaml`, `config/config.sha256`, `CHANGE_REQUEST.md`.
- Feature and matching code: `src/features.py`, `src/matching.py`.
- Stages: `stages/s02_master.py`, `stages/s06_matching.py`,
  `stages/s07_clustering.py`, `stages/s08_models.py`, `stages/s09_sequential.py`,
  `stages/s10_survival.py`, `stages/s11_robustness.py`,
  `stages/s12_package.py`.
- Report/poster/README generator: `src/reporting.py`.
- Engineering: `Makefile`, `tests/test_windows.py`, `tests/test_features.py`,
  `tests/test_splits.py`, `tests/test_logging.py`.
- Generated deliverables (S12): `Report.pdf`, `Report.tex`, `Poster.pdf`,
  `README.md`, `REPORT.md`, `DATA_DOWNLOAD.md`, `Group_09_Project.zip`.
- `src/validate.py` was inspected but did not require a change.

## 4. WP3 — engineering and reproducibility

- `tests/test_windows.py`, `test_features.py`, `test_splits.py`,
  `test_logging.py`: **9 passed**.
- `Makefile` targets `sim`, `full`, `validate`, `test`.
- Final clean simulation `sim-20261104T033345Z-59a902`: 13 stages pass,
  `validate_run.py` **0 errors / 0 warnings**.
- Final full run `run-20261104T031722Z-054c79`: 13 stages pass,
  `validate_run.py` **0 errors / 0 warnings**.
- `STATUS.final.json` is present in the run directory and the ZIP; raw data and
  Poster.pdf are not in the ZIP.

| Run | Mode | Stages | validate_run.py | Report pages |
|---|---|---|---:|---:|
| `sim-20261104T033345Z-59a902` | simulation | 13/13 pass | 0 errors / 0 warnings | 16 total / 7 main |
| `run-20261104T031722Z-054c79` | full | 13/13 pass | 0 errors / 0 warnings | 16 total / 9 main |

The simulation was validated immediately after its S12. Its root `REPORT.md`,
`README.md`, `Group_09_Project.zip`, and `STATUS.json` were then replaced by the
full-run deliverables; re-validating the simulation later therefore reports
cross-run root-file hash/status mismatches. The full run is the final root state
and remains 0/0. This is a known limitation of the single-root packaging design,
not a stage failure.

An intermediate debug simulation (`sim-20261104T024202Z-ad54e2`) and an
intermediate full run (`run-20261104T030052Z-b96dee`) were used while fixing
S12 layout issues; both are retained read-only. They are not used as final gate
evidence. The final clean simulation and final full run above are the audit
runs.

## 5. Before / after

| Item | Baseline full | Final full |
|---|---:|---:|
| M1 main PR-AUC / ROC-AUC | .8267 / .7236 | .8267 / .7236 |
| M3 main PR-AUC / ROC-AUC | .8283 / .7294 | .8283 / .7294 |
| M4 main PR-AUC (upper bound) | .8432 | .8432 |
| M1 grouped-CV ROC-AUC | .7188 | .7188 |
| M1 temporal ROC-AUC | .7069 | .7069 |
| M1 LOMO ROC-AUC | .6993 | .6993 |
| CEM matched T1 n | 20,660 | 20,660 |
| CEM common support | .900 | .900 |
| E-value | 1.555 | 1.555 |
| Cluster k / ARI / Jaccard | 2 / .913 / .810 | 2 / .913 / .810 |
| P009 pre-start spec PR-AUC | .828320 | .828851 |
| B2 PR-AUC / Brier / slope | .7591 / .3041 / .0509 | .7591 / .3041 / .0509 |
| B2 status | absent (treated as normal) | failed_calibration |
| Report pages / main pages | 11 / not stated | 16 / 9 |
| T1 full n in Table 1 | 22,429 (mislabeled at-risk) | 22,437 |
| T3 Fail in Table 1 | 7,044 (at-risk) | 7,052 |
| Trajectory divergence | not analysed | week 3 (inside W1) |
| Context-only / M0c / M1c PR-AUC | not analysed | .783 / .853 / .854 |
| Landmark test C (L=28 Fail/Pass/Withdrawn) | static in-sample .664/.563/.770 | .662/.560/.572 |
| Module PR-AUC range | not analysed | .778–.964 |
| Subgroup calibration range | not analysed | .571–1.933 |
| Pre-start-only / future-window / random control | .782 / .861 / .530 | .782 / .861 / .530 |

The main strict-model metrics are deliberately unchanged. The revision fixes
design, reporting, and secondary analyses; it does not tune the main model on
the test set.

## 6. WP4 — optional TMA analyses

Not attempted. The TMA+21/+28 pairing and TMA+14 checkpoint would define
alternative estimands that overlap the W1/W2 windows, and the first-TMA score
is not available at the early cutoff. Adding them would either duplicate the
W2 analysis or create a post-hoc, availability-dependent result that cannot be
used as an early-warning claim. The report therefore contains no TMA main-text
or appendix result. If a future revision adds them, they should be appendix-only,
with an explicit availability caveat and no effect on the main conclusion.

## 7. Remaining caveats

- Observational associations only; CEM balances measured W1 variables and
  cannot remove unmeasured confounding. The E-value of 1.555 is modest.
- W1 engagement is post-baseline; the week-3 trajectory divergence means the
  “same engagement” premise is approximate even within W1.
- Context-only models use background variables and are descriptive; the
  M1c−M0c increment of .001 PR-AUC is small and is not a causal effect.
- Pre-start-only and future-window controls retain signal; they are not clean
  leak detectors. Random-label is close to chance.
- B2 remains a failed calibration specification; the updating analysis provides
  no evidence that B2 adds value.
- T3 test n=6,508; subgroup cells can be small (low-n flagged), and no
  subgroup causal claims are made.
- 93 Withdrawn enrolments lack `date_unregistration` and are treated as
  course-end events.
- The full report is 16 pages with 9 main-text pages; the appendix is excluded
  from the 6–10 page main-text requirement.
- `Poster.pdf` is submitted separately and is intentionally not in the ZIP.
