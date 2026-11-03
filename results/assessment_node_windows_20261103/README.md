# First weighted TMA: 2, 3, and 4 post-deadline weeks

This sensitivity experiment uses one shared feature and model definition at
each horizon. The pre-deadline window is always days -13 through 0. Post
windows are days 1–14, 1–21, and 1–28. Each cutoff has its own as-of-cutoff
submission indicators and requires enrollment through that cutoff. Raw future
submission dates and scores are absent from the saved features and models.

Reproduce from the same extracted OULAD archive:

```bash
python -m oulad_analysis.assessment_node --data-dir data/raw --output-dir outputs/node_2w --post-weeks 2
python -m oulad_analysis.assessment_node --data-dir data/raw --output-dir outputs/node_3w --post-weeks 3
python -m oulad_analysis.assessment_node --data-dir data/raw --output-dir outputs/node_4w --post-weeks 4
python -m oulad_analysis.assessment_node_compare --two outputs/node_2w --three outputs/node_3w --four outputs/node_4w --output outputs/node_common_cohort.csv
```

All four calls require new output paths. The ignored outputs include student
features; only aggregate CSV/JSON files are stored here. Every model uses the
same timing feature set, now including the fraction of post weeks with any
VLE activity. That addition slightly changes the archived two-week result
in `results/assessment_node_20261103`; use the two-week row here for this
comparison.

| Post window | Eligible at cutoff | Binary cohort | Seed-42 B2 AP | Seed-42 B3 AP | B3−B2 AP, 95% student bootstrap interval |
|---|---:|---:|---:|---:|---:|
| 2 weeks | 24,125 | 19,806 | 0.654318 | 0.668699 | +0.014382 [0.006896, 0.022198] |
| 3 weeks | 23,923 | 19,806 | 0.668445 | 0.681108 | +0.012663 [0.004600, 0.020360] |
| 4 weeks | 23,688 | 19,806 | 0.682308 | 0.698370 | +0.016062 [0.007585, 0.024531] |

B2 = presentation context, total and pre/post clicks, and submission status
as of the cutoff. B3 adds pre/post active days, pre-deadline last-three-day
share, coverage of the first two post weeks, fraction of active post weeks,
and post/pre log-click change. The five student-grouped split B3−B2 AP
differences are all positive: 2 weeks +0.0083 to +0.0159, 3 weeks +0.0093
to +0.0155, and 4 weeks +0.0149 to +0.0220. The splits overlap and are
exploratory; the pointwise intervals condition on the first split.

The binary cohort is identical at all three cutoffs. The common-cohort
command confirms 19,806 shared records and identical student splits, so the
AP increase with a longer window here reflects more observed time rather
than a change in this binary population. Withdrawn records are excluded from
the binary model and decline from 4,319 to 4,117 to 3,882 as the cutoff
moves. Thus these models do not measure withdrawal risk in the full
at-deadline cohort.

2013→2014 transfer excludes repeated students. On previously seen modules,
the B3−B2 AP changes are +0.006640 (2 weeks), +0.003146 (3 weeks), and
−0.003916 (4 weeks). On new module CCC they are +0.004198, +0.010300,
and +0.028164 respectively. The new-module result is one module, not a
general transfer estimate. Four-week gains on the grouped random split do
not generalize consistently to later presentations of known modules.

**Decision:** Three weeks is the most defensible main expansion if a longer
post-deadline horizon is required; retain two and four weeks as predeclared
sensitivity windows. Four weeks has the highest absolute AP because more
course behavior is observed, but it does not show a more reliable incremental
timing signal across years. None of these comparisons establishes a causal
benefit from activity or an improved early intervention. The observation
times differ, so a higher AP at four weeks is not a better method at the
same decision time.

Files `models_*weeks.csv`, `intervals_*weeks.csv`, `course_*weeks.csv`,
`year_*weeks.csv`, and `protocol_*weeks.json` give the per-horizon results;
`common_cohort_models.csv` records the fixed-cohort comparison. Source
hashes and cohort rules are in each protocol JSON. The original source
archive is documented in `data/README.md`.
