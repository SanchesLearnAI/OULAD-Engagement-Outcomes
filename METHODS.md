# Analysis contract

This code implements the first version of `research/PROJECT_PLAN.md` in the
surrounding project workspace. It uses OULAD `studentInfo.csv`, `studentVle.csv`
and `vle.csv` at the student × module × presentation grain. The public UCI
source ZIP is committed with attribution. Extracted CSVs and student-level
outputs remain outside Git.

## Cleaning and feature dictionary

- Keep every `studentInfo` row. Require unique keys in `studentInfo` and unique
  `(code_module, code_presentation, id_site)` in `vle`.
- Keep `studentVle` dates **0–27 inclusive**. Preserve all source rows when
  summing clicks, including repeated student/site/day and exact repeated rows;
  the official release contains many such rows, and deleting them changes click
  totals. Report distinct-row and distinct student/site/day counts. Reject
  missing core join fields or negative click counts. Relevant missingness is
  recorded in the quality report.
- Left join resources by module, presentation and site. Unmatched sites are
  counted and assigned activity type `__unmatched__` so clicks are retained.
- `total_clicks`: sum of clicks in the window; `log_clicks`: `log1p(total_clicks)`;
  `zero_click`: 1 if the sum is zero. All metrics below use positive-click days.
- `active_days`: number of days with positive clicks. `longest_inactive_run`:
  longest run of zero-click days among all 28 days, including the ends.
- `week_0_share` through `week_3_share`: shares in days 0–6, 7–13, 14–20,
  and 21–27. `daily_concentration`: sum of squared day click shares (HHI).
  `daily_entropy`: Shannon entropy of positive daily click shares.
- `resource_share__<activity_type>`: fraction of clicks to each resource type.
  `resource_type_count`: number of positively clicked types.
  `resource_entropy`: Shannon entropy of the type click shares.
- For zero-click samples, all shares, concentration and entropy are set to zero;
  `active_days` and `resource_type_count` are zero, and
  `longest_inactive_run` is 28. This is an explicit convention, not evidence of
  regular behavior.

`data_quality.json` contains row counts, duplicate counts, unmatched events,
zero-click count, label counts and the resource vocabulary. `features.parquet`
holds one row per original `studentInfo` record.

## Descriptive analyses

The similar-click analysis excludes Withdrawn and compares Fail with
Pass/Distinction only within a module presentation and a quantile bin of
`log_clicks`. It reports 5, 10 and 20 bin sensitivity analyses. A bin contributes
only if both outcomes occur. Differences are Fail mean minus Pass/Distinction
mean, weighted by bin size. These are associations, not matched causal effects.
High-activity failures and low-activity successes use each presentation's 75th
and 25th click percentiles respectively. Withdrawn remain in the feature table
and cluster outcome descriptions; a separate presentation-level summary reports
their count and share. They are excluded from the binary model.

Clustering uses nonzero-click students only and samples at most 2,500 records
with a fixed seed. It standardizes weekly shares, resource shares, active days,
longest inactivity, daily concentration and resource entropy. Total clicks and
final result are excluded from cluster fitting and candidate choice. K-means
and agglomerative clustering are compared at 2–6 clusters. Candidates with a
cluster smaller than 3% or five students are rejected. Silhouette is the primary
selection score; mean ARI over three 80% subsamples breaks ties. Reported
profiles and outcome proportions describe the selected sample, not a full
population clustering.

## Modeling and uncertainty

Fail is positive. Pass and Distinction are negative; Withdrawn are excluded.
One grouped 80/20 student-ID split is shared by every model. A 3-fold
`StratifiedGroupKFold` inside the training set generates out-of-fold scores for
each model's F1 threshold. Preprocessing is fitted inside each fold and refitted
on the full training set before the single held-out evaluation. No matching
selection is applied to the model test set.

The seven models are pure log-click logistic regression; log-click plus module
and presentation logistic regression; time, resource and full additions to that
controlled baseline; and baseline/full random forests. Logistic regression
uses standardization and one-hot context. Random forest uses 150 trees and
minimum leaf size 5. Weekly and resource shares omit one category in the
logistic models to avoid their exact sum constraint. The resource category
omitted is the last one alphabetically. All model definitions, thresholds and
holdout metrics are saved in `model_results.json`.

Report Fail precision, recall, F1, average precision, prevalence, and the
student-paired 95% bootstrap intervals for F1 and AP differences against the
same-algorithm controlled baseline (500 replicates, seed 42). Presentation
metrics are descriptive and omitted where a presentation's test slice has only
one class. AP is the non-interpolated sklearn definition; prevalence is the
no-skill reference. The results are observational: clicks do not measure study
time or ability.
