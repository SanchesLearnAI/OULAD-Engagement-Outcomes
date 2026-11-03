# Revised exploratory results (2026-11-02)

The code in this working revision extends the first OULAD run with registration
eligibility, click-balance diagnostics, nonlinear controlled models, robustness
checks, and clustering diagnostics. `run_manifest.json` identifies the source
data and execution environment. The published original results remain in
`results/local_uci_20261102` for comparison. All files here are aggregate;
student-level features and predictions stay in the ignored `outputs/` folder.

## Population and data checks

The source contains 32,593 student-module-presentation records and 22,437
Pass/Distinction/Fail records. There are 4,707 zero-click records; 3,458 have
unregistered by day 27. The stricter eligibility flag retains 27,528 records.
It requires a known registration date no later than day 27 and no unregistration
through day 27. This is a descriptive sensitivity population, not a replacement
for the original binary-model cohort.

The raw 0–27 day window has 2,112,323 event rows, including 140,592 exact
repeats beyond the first occurrence. The main run preserves source rows.
The separate deduplication run removes exact repeats. Its `full_nonlinear_lr`
AP is 0.596956 versus 0.597204 in the main run; this small change does not
show that repeated rows were errors or that deduplication is preferable.

## Behavioral comparisons

The 20-bin within-presentation comparison covers 22,231 records; weighted
absolute click standardized difference is 0.300. Restricting to strata with
absolute click standardized difference at most 0.1 retains 4,918 of 22,437
binary records. Within that subset, Fail minus Pass/Distinction is -1.044
active days, +0.877 longest inactive days, and -0.0148 week-3 click share.
This reduced-coverage comparison is descriptive and does not establish a
causal effect. `similar_click_ci.csv` reports student-bootstrap intervals for
the original 20-bin comparison, not the strict subset.

## Models and clustering

On the original grouped holdout, the quadratic log-click plus course-batch
baseline AP is 0.572885 and the corresponding full-feature model AP is
0.597204. Five other grouped splits yield AP deltas from +0.0102 to +0.0341.
The 2013-to-2014 transfer check yields +0.0182 after excluding students shared
between years from its test slice. This temporal check tests unseen
presentations and uses unknown-category handling for the course-batch feature.
All these evaluations are exploratory because this dataset and the original
holdout have already informed model design.

The selected two-cluster solution still separates click volume: cluster mean
clicks are about 373 and 65. Early unregistrations number 47/1,688 and
98/812 respectively. Time-only clustering at the same k has ARI 0.723
against the full-feature labels. Use the module-composition and stratified
outcome tables before assigning behavioral labels to clusters.
