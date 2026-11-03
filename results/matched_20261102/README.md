# Primary analysis: comparable clicks, different outcomes

This analysis directly addresses the original proposal: within the same module
and presentation, compare day-0–27 behavior of Pass/Distinction and Fail records
with comparable total clicks. Withdrawn is excluded. Previous population-wide
high/low activity clusters do not answer this question and are supplementary EDA.

## Matching and coverage

There are 22,437 binary records, including 7,052 Fail records. Exact-first,
seeded nearest-neighbor matching uses no behavioral features and no record reuse
within each run. Positive click ratios cannot exceed the stated limit; zero
clicks match only zero. The greedy matching is reproducible, not globally optimal.

| Maximum relative click gap | Pairs | Retained records | Fail coverage | Actual mean click gap |
|---|---:|---:|---:|---:|
| 0% (exact) | 3,874 | 7,748 | 54.9% | 0% |
| 5% | 5,724 | 11,448 | 81.2% | 0.379% |
| 10% | 5,830 | 11,660 | 82.7% | 0.558% |

Each run includes 232 zero-click pairs. Balance is artificial because each pair
has one Fail and one Pass/Distinction; it must not be used to estimate failure
prevalence in the population.

## Observed differences

At the main 5% tolerance, Fail records have 8.89 active days versus 9.73 for
their passing counterparts, a difference of -0.84 days (pointwise 95% interval
[-0.96,-0.72]). Their longest inactive run is 9.53 versus 8.74 days, a difference
of +0.79 days ([0.66,0.91]). Weekly normalized entropy differs by -0.046
([-0.053,-0.039]); peak rolling-seven-day share differs by +2.44 percentage
points ([1.87,2.96]). These four directions recur in all 22 presentation slices
with at least 30 pairs; this is sign consistency, not 22 significance claims.

The exact-click comparison also finds fewer active days (-0.71), longer gaps
(+0.72), lower weekly evenness (-0.046) and higher rolling-week concentration
(+2.56 percentage points) among Fail records. Directions persist at 10% and
across seeds 42,43,44. Thus the observed temporal differences are not explained
solely by the permitted small residual click-count gaps.

Forum share is about 1.03 percentage points lower among Fail records at 5%,
but this direction holds in only 14 of 22 presentations. Resource-use claims
are more heterogeneous and should not be generalized across courses.

## Interpretation and limits

The current supported pattern is that, at comparable click volume and within
the same course batch, passing records tend to distribute activity across more
days/weeks with shorter interruptions. This describes associations, not causal
effects or a rule that classifies every student. The correlated temporal
features are not proven independent mechanisms. No discrete behavioral types
have yet been validated by this comparison.

Intervals use 1,000 bootstrap resamples of connected components of matched
student pairs, preserving dependencies when a student occurs in multiple
presentations. They are conditional on fixed pairs, pointwise and exploratory;
no rematching or multiple-testing correction is included. The matched cohort
does not represent all students. Unmatched student-holdout classification
remains a separate evaluation, as required by the proposal.

Run `python -m oulad_analysis --data-dir data/raw --output-dir <feature-output>
--stage matched --seed 42` after the feature stage. Row-level pairs remain in
ignored outputs. Two new tests cover calipers, context isolation, zero clicks,
no reuse, reproducibility, paired effects and repeated-student dependence.
