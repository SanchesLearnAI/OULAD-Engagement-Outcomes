# Analysis contract

This code implements the first version of `research/PROJECT_PLAN.md` in the
surrounding project workspace. It uses OULAD `studentInfo.csv`, `studentVle.csv`
and `vle.csv` at the student × module × presentation grain. The public UCI
source ZIP is committed with attribution. Extracted CSVs and student-level
outputs remain outside Git.

## Revised research question and implementation status

The target comparison is among students in the same module presentation with
similar VLE click volume at 21 or 28 days after the first positive-weight TMA
deadline: which submission, timing, and resource-use patterns differ between
eventual Pass/Distinction and Fail? These are separate observation cutoffs, with
VLE features measured from deadline day -13 through the cutoff.
`assessment_node` tests incremental prediction at those cutoffs on unmatched
student holdouts. `tma_matched` performs the within-presentation click
matching, resource-type comparisons, and resource model ablation.
`tma_score_probe` then checks whether behavior persists at comparable first-TMA
marks. The `matched` analysis below uses course days 0–27 and is an earlier
baseline.

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

### Coverage and concentration extension

The original daily HHI/entropy and four weekly shares already describe timing.
The following additional summaries separate full-window coverage from evenness
conditional on being active. Each uses only day 0–27 clicks:

| Feature | Definition | Interpretation |
|---|---|---|
| `weekly_evenness` | Entropy of four weekly shares divided by ln(4) | 1 for equal weekly totals, 0 for a single active week |
| `active_weeks` | Number of calendar weeks with positive clicks | Coverage, from 0 to 4 |
| `max_week_share` | Maximum of four weekly click shares | Fraction concentrated in one calendar week |
| `peak_7day_share` | Largest rolling seven-day click sum divided by total | Detects bursts across calendar-week boundaries |
| `active_day_evenness` | Daily entropy divided by ln(active days) for at least two active days | Evenness among active days only; must be read with active-day count |

All five are 0 for zero-click records. For a single active day,
`active_day_evenness` is 0 by convention. Zero-click records are excluded from
nonzero-click distribution summaries and counted separately. Uniform activity
on 14 days has active-day evenness 1 but does not cover all 28 days. Equal
weekly totals can also arise from one active day each week. Daily records
cannot establish within-day session duration or timing.

Most additions summarize existing features nonlinearly; rolling seven-day
concentration additionally preserves contiguous-time information absent from
calendar-week shares. `concentration_full_lr` adds the five summaries to
`full_nonlinear_lr`, uses the identical student split, and reports paired
differences against that exact model. The five repeated grouped splits also
report `delta_concentration_ap`. This is an exploratory ablation, not a new
untouched test or automatic replacement of the main model. The 2013-to-2014
check remains the previous two-model check and does not validate this extension.
Existing clustering feature weights are unchanged.

`timing_summary.csv` gives coverage and concentration counts for all records,
day-28 eligible records, and the binary population. `timing_quantiles.csv` and
`timing_distribution.png` describe nonzero-click records. The 0.9 weekly
evenness and 50% concentration cutoffs are descriptive summaries, not learned
thresholds or claims about optimal study patterns.

`studentRegistration.csv` supplies registration and unregistration dates. The
feature table retains these dates and flags students registered by day 27 who
remained registered through day 27. Dates are descriptive, not model inputs.
Missing event dates are rejected before filtering; unknown outcome labels and
non-unique registration keys are rejected. Exact event deduplication is optional
and used only in a separate sensitivity run.

`data_quality.json` contains row counts, duplicate counts, unmatched events,
zero-click count, label counts and the resource vocabulary. `features.parquet`
holds one row per original `studentInfo` record.

## Descriptive analyses

### First weighted TMA node experiment

`python -m oulad_analysis.assessment_node` is an independent experiment.
`--post-weeks` selects 2, 3, or 4 weeks after the deadline (default 2); the
revised question focuses on 3 and 4 weeks.
For
each module presentation, choose the earliest TMA with `weight > 0` from
`assessments.csv`; a zero-weight task or CMA is not substituted. This yields
19 presentations in six modules; GGG has no qualifying TMA. Join
`studentInfo`, `studentRegistration`, and the chosen record in
`studentAssessment` at their documented keys, rejecting duplicate keys.
Exclude a banked assessment, unknown registration date, registration after
the deadline, and unregistration on or before deadline + the selected
post-window. The unit is
one student × module × presentation, and the observation cutoff varies by
presentation. This selection is a retrospective cohort, not a prediction
about people who already unregistered at the cutoff.

Retain source VLE clicks from relative days -13 through 7×post-weeks,
inclusive. The pre-period is -13 through 0 (14 days) and the post-period is
1 through 14, 21, or 28 days. Sum original event rows without deduplication. Compute pre/post click
totals and active days, last-three-pre-days share, presence in each post
week, fraction of observed post weeks with any clicks, and log1p(post clicks) - log1p(pre clicks). Submission indicators use
only a submission date on or before the cutoff: submitted by cutoff and
submitted after the scheduled deadline but by cutoff. Future submission dates
are absent from the saved feature table. An early submission does not imply
that its eventual score was released, so scores are not used. The source does
not record personal deadline extensions or realtime ingestion timestamps.

Evaluate four nested logistic models on the same unmatched student-ID grouped
80/20 split: B0 adds module presentation and pre/post click volume; B1 adds
the six timing indicators; B2 adds the two submission indicators to B0; B3
adds timing to B2. All models also include log1p total clicks and its square.
Report Fail AP, prevalence, Brier score, five grouped split results, and 1,000
paired student-bootstrap intervals for B1-B0, B2-B0, and B3-B2 on the first
split. All models use fixed regularization and train-only numeric scaling;
this experiment does not optimize a threshold or evaluate a new untouched
sample. The main contrast B3-B2 asks whether node timing provides information
beyond click volume and submission status at the same cutoff. Report module
differences as descriptive slices; the post-hoc binary label excludes final
Withdrawn and therefore is not an all-student early-warning claim. Across
2/3/4-week runs, each cutoff has a different enrolled population. The
`assessment_node_compare` command also intersects all three binary cohorts,
sorts the same enrollment keys, and repeats the same student splits to isolate
longer observation from cohort turnover. This common-cohort analysis uses
four-week survival to define membership, so it is retrospective rather than
an earlier prediction. The timing feature set includes post-window active
days, first-two-week coverage, and the fraction of active post weeks at every
cutoff; the latter was added for the multiwindow experiment and changes the
two-week model slightly from its earlier archived run.

### Matched TMA comparison at days +21 and +28

`python -m oulad_analysis.tma_matched` reads the two saved node feature tables
and the original `studentVle.csv` and `vle.csv`. It retains the same binary
population as the node models: students registered through each cutoff with a
known final Pass, Distinction, or Fail outcome. It does not add future scores
or submission dates. The two cutoffs are analyzed separately; both use VLE
clicks from relative day -13 through the selected cutoff. The row key is
student × module × presentation.

Each Fail record is paired without replacement to a Pass/Distinction record
in the same module and presentation. Matching uses total VLE clicks over the
node window: exact matches first, then nearest log-click matches within a
5% positive-click ratio; exact-only and 10% are sensitivity runs. Zero clicks
match zero only. Submission, timing and resource features are never used to
choose partners. The existing greedy algorithm is deterministic but does not
guarantee the globally optimal assignment. Report pair counts, Fail coverage,
actual click gaps and log-click standardized mean difference at each cutoff.

Resource metadata joins on module, presentation and site ID. Original event
rows are summed without deduplication for days -13–0, 1–21 and 22–28; a missing
site type would be kept as `__unmatched__`. Per-student resource counts must
sum exactly to the node's all-window and post-window click totals before any
comparison. Compare type counts, entropy, and each original activity type's
share of all or post-deadline clicks. A zero post-click record has zero post
shares and entropy by convention. Resource types are not manually labeled as
deep or shallow; their availability varies by course.

Paired effects are Fail minus Pass/Distinction means. Pointwise 95% intervals
resample connected components of the student-pair graph 1,000 times and are
conditional on the fixed pairs. They do not adjust for testing many features
or account for rematching uncertainty. Presentation-level effects expose
heterogeneity. A secondary, exploratory sensitivity selects the 5%-total-click
pairs where both students have positive post clicks within a further 5% or 20%
post-click ratio. This subset is smaller and changes the population being
compared; it helps assess whether post resource diversity merely reflects
different post-click volume.

For prediction only, B4 adds resource diversity and all observed all-window
and post-window activity-type shares to B3, with the same five student-grouped
splits. B2 and B3 use the existing node definitions. The first split gets a
500-draw paired student bootstrap for B3−B2 and B4−B3 AP. This ablation tests
incremental prediction on unmatched students; it is not the paired-pattern
estimate. Row-level features and pairs remain under ignored `outputs/`.

### First weighted TMA score and rank sensitivity

`python -m oulad_analysis.tma_score_probe` joins the same first positive-weight
TMA's `studentAssessment` record by assessment ID and student ID. A mark is
included only if the submission date is on or before the observation cutoff
and the final dataset records a 0–100 score. OULAD has no date on which the
mark was released; this is a retrospective control and is never used as an
as-of-cutoff model feature. Missing marks stay missing. Rank is the average-tie
score percentile within the same assessment, computed among cutoff-eligible
scored students including eventual Withdrawn records. The rank is descriptive
and is not claimed to be an operationally available live rank.

First, quantify score coverage and compare score/rank within the original 5%
total-click pairs where both marks are recorded. Then form new, no-reuse pairs
within the same module presentation, requiring marks within five points and
total clicks within a 5% ratio. Allocate exact click matches first. The
greedy result is not globally optimal. Compare the same timing and selected
resource features with 1,000 connected-student-component bootstrap draws.
Presentation slices check direction consistency. An exploratory subset also
requires positive post-deadline clicks within a 20% ratio to test whether
resource diversity differences track post-click volume. Since missing marks
are concentrated among Fail records, results for scored pairs cannot be
extended to non-submitters. A first TMA with positive weight contributes to
the eventual course result; score–outcome associations are partly mechanical.

### Retrospective score/rank model ablation

`python -m oulad_analysis.tma_score_model` reads the enriched binary node
features from the score probe. On the same student-ID grouped 80/20 splits as
the node models, compare B3 timing, B4 timing+resources, S0
clicks+submission+score, S1 B3+score, S2 S1+within-presentation score rank,
and S3 S2+resources. Score availability is an explicit binary feature and
missing marks stay missing until training-only median imputation. The rank
used by models is **not** the score probe's full-cohort descriptive rank:
it is recalculated within each split from training marks only, then applied
to held-out students through the training presentation's empirical score
distribution. This prevents test-distribution leakage. All preprocessing is
fitted on training data. Five overlapping grouped seeds test direction;
the first split receives 1,000 paired student-bootstrap AP intervals for
score, timing-after-score, rank-after-score, and resources-after-rank.
The models are explanatory, not operational: mark-release timestamps are
absent, the TMA contributes to final result, and the split is not untouched.

The earlier four-week `matched` analysis pairs one Fail record with one
Pass/Distinction record within the same module and presentation, without reuse
within a run. Exact click matches are allocated first; remaining Fail records
are considered in seeded random order and paired to the nearest available
log-click neighbor if the ratio of larger to smaller positive click counts is
at most 1.05 (main tolerance) or 1.10 (sensitivity). Exact-only matching is
also reported. Zero clicks match zero only. The greedy algorithm is reproducible,
not claimed to maximize match count or minimize global distance. Matching uses
context, clicks and outcome group only, never the behavioral features being
compared. Report retained pair counts, coverage and actual click gaps.

Effects are pair-averaged Fail-minus-Pass/Distinction differences. Pointwise
95% intervals resample connected components of the pair graph (student IDs are
nodes) 1,000 times, retaining dependencies from repeated students and both
sides of each pair. They are conditional on the fixed matching and do not
include rematching uncertainty or multiple-testing adjustment. Module-presentation
effects are reported separately; a pooled association is not automatically
consistent across courses. This is observational, outcome-defined sampling,
not causal treatment matching. The artificial 50/50 pair balance cannot be used
to estimate population failure rates. Pair records are kept in ignored outputs.

The previous quantile-bin comparison below is supplementary sensitivity work.
The all-nonzero-student clustering is exploratory population EDA and cannot
answer the similar-engagement question by itself.

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

The models include pure log-click logistic regression; log-click plus module
and presentation logistic regression; time, resource and full additions to that
controlled baseline; and baseline/full random forests. Logistic regression
uses standardization and one-hot context. Random forest uses 150 trees and
minimum leaf size 5. Weekly and resource shares omit one category in the
logistic models to avoid their exact sum constraint. The resource category
omitted is the last one alphabetically. A joint module-presentation category
controls each course batch. A quadratic log-click baseline and its full-feature
counterpart check whether gains are due to nonlinear click effects. All model definitions, thresholds and
holdout metrics are saved in `model_results.json`.

Report Fail precision, recall, F1, average precision, prevalence, and the
student-paired 95% bootstrap intervals for F1 and AP differences against the
same-algorithm controlled baseline (500 replicates, seed 42). Presentation
metrics are descriptive and omitted where a presentation's test slice has only
one class. AP is the non-interpolated sklearn definition; prevalence is the
no-skill reference. The results are observational: clicks do not measure study
time or ability.

`robustness.csv` reports five additional grouped random splits and a
2013-to-2014 transfer check that excludes students appearing in both years from
the later test slice. These are exploratory checks, not untouched test results.
`click_balance.csv` reports residual click imbalance in each binning scheme;
`similar_click_ci.csv` bootstraps students with fixed strata.
