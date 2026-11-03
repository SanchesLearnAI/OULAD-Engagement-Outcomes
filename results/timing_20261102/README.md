# Timing coverage and concentration

The source provides daily click counts; within-day click timing and study
duration cannot be recovered. The window is days 0–27. Existing daily entropy,
HHI, active days, gaps and weekly shares already capture some concentration.
This extension adds weekly evenness, active weeks, maximum calendar-week share,
maximum rolling-seven-day share and evenness conditional on active days.
Definitions and zero-activity conventions are in `../../METHODS.md`.

## Observed distribution

The binary cohort has 22,437 student-course records; 947 zero-click records
are excluded from the distributions below, leaving 21,490 records:

- 13,870 (64.5%) have clicks in all four calendar weeks.
- 11,238 (52.3%) put at least half their clicks into one calendar week.
- 14,314 (66.6%) put at least half their clicks into some consecutive seven days.
- Median active-day count is 10 of 28; median rolling-seven-day peak share is 57.2%.
- Weekly normalized entropy is at least 0.9 for 5,997 records (27.9%). This is
  a descriptive cutoff, not a definition of optimal or adequate engagement.

These groups overlap. Four active weeks do not imply even click counts.
Even equal weekly totals may result from one active day per week. Evenness
among active days must be interpreted alongside the number of active days.
`timing_summary.csv` also describes all outcomes and day-28 eligible records.

## Incremental prediction experiment

The existing complete nonlinear logistic model and its concentration-extended
variant use identical student splits and training-only threshold selection.
AP is 0.597204 before and 0.597179 after the additions. The paired student
bootstrap 95% interval for AP difference is [-0.002435, 0.002288]. F1 is
0.571353 before and 0.568937 after. Five other grouped splits give AP deltas
of -0.000067, +0.001081, +0.001186, -0.000348 and +0.000557.

There is no clear incremental predictive gain in these exploratory checks.
Keep the features for descriptive interpretation and the named ablation;
do not present them as a performance improvement. Several new summaries are
nonlinear transformations of existing features, so redundancy is plausible.
The repeated splits are not independent datasets. The temporal row in
`robustness.csv` remains the previous two-model comparison and has no
concentration-extension result; its blank delta is not a zero improvement.

Reproduce with a new output directory:

```bash
python -m oulad_analysis --data-dir data/raw --output-dir outputs/timing_new --stage all --seed 42
```

Three tests passed, including equal-total examples for all-day uniformity,
partial-window uniformity, weekly spikes, a burst across a week boundary,
and zero activity. Execution used Python 3.14.4; see the manifest for package
versions. The source checkout was dirty, so its base commit is not an immutable
identifier for the complete working revision. Student-level outputs remain in
the ignored output directory.
