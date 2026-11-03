# Full score/rank training round at first weighted TMA +21/+28

## Scope and feature handling

This is a **retrospective model ablation**, extending the
[score/rank matched study](../tma_score_probe_20261103/README.md). The binary
cohort contains 19,806 Fail/Pass/Distinction records at either cutoff.
`first_tma_score` is the final recorded first weighted TMA mark only when its
submission date is on or before the cutoff; other marks are missing. A
score-availability indicator is included. OULAD does **not** record when
grades were released, so these models do not establish deployable warning
performance at +21 or +28 days. The mark also contributes to the final
course result, so some predictive gain is mechanically expected.

Every model uses the same student-ID grouped 80/20 split and test records
within a cutoff. Five overlapping seeds (42, 143–146) check direction. The
rank feature is recomputed *inside each split* from training marks only:
each score is mapped to the empirical within-presentation midrank percentile
of training students. Test students never contribute to their own rank
reference distribution. Missing scores and ranks are imputed from training,
with the explicit availability indicator.

| Model | Added information |
|---|---|
| B3_timing | Presentation, total/pre/post clicks, cutoff submission, six timing features |
| B4_resources | B3 + type counts, entropy, all-window and post-window resource shares |
| S0_score_volume | Presentation, total/pre/post clicks, cutoff submission + score and availability |
| S1_add_score | B3 + score and availability |
| S2_add_rank | S1 + training-only within-presentation rank |
| S3_full | S2 + all resource features from B4 |

## Same-held-out-students result (seed 42)

| Model | +21 AP | +21 Brier | +28 AP | +28 Brier |
|---|---:|---:|---:|---:|
| B3_timing | 0.68111 | 0.16108 | 0.69837 | 0.15690 |
| B4_resources | 0.67835 | 0.16206 | 0.69090 | 0.15766 |
| S0_score_volume | 0.71844 | 0.15062 | 0.73027 | 0.14783 |
| **S1_add_score** | **0.72405** | **0.14855** | **0.73819** | **0.14508** |
| S2_add_rank | 0.72356 | 0.14860 | 0.73821 | 0.14509 |
| S3_full | 0.71975 | 0.14958 | 0.72933 | 0.14588 |

The first split's paired student-bootstrap 95% AP intervals give:

| Contrast | +21 AP difference | +28 AP difference |
|---|---:|---:|
| S1−B3: add score to timing model | +0.04294 [0.03206, 0.05477] | +0.03982 [0.02933, 0.04967] |
| S1−S0: add timing after score | +0.00561 [-0.00063, 0.01185] | +0.00791 [0.00096, 0.01502] |
| S2−S1: add rank after raw mark | -0.00049 [-0.00126, 0.00005] | +0.00003 [-0.00033, 0.00036] |
| S3−S2: add full resource block | -0.00381 [-0.00846, 0.00122] | -0.00888 [-0.01901, 0.00049] |

S1−B3 is positive in all five grouped splits: +0.0332 to +0.0429 AP at +21,
and +0.0283 to +0.0398 at +28. S1−S0 is also positive in all five, but the
splits overlap and the +21 first-split interval includes zero. Adding rank
after raw score yields essentially no increment; adding every resource type
does not improve AP on this set of tests. The lean S1 model outperforms the
larger S3 model on seed 42 at both cutoffs. This is a tested comparison, not
a claim that S1 is globally optimal.

## Interpretation and reproduction

The first TMA mark adds substantial information about the final outcome,
while timing retains a smaller, positive conditional association. Raw score
is more useful than adding a within-presentation rank to this linear model.
The extra resource block has no reliable gain. These conclusions concern
retrospective, already-enrolled non-Withdrawn students. The dataset lacks
grade release dates, the score partly determines the outcome, and the test
split has been explored in earlier analyses. No new untouched validation or
new-presentation generalization is claimed; a higher absolute AP at +28 also
uses seven more days of behavior.

```bash
python -m oulad_analysis.tma_score_model --three outputs/tma_score_probe --four outputs/tma_score_probe --output-dir outputs/tma_score_model --bootstrap 1000
```

`models.csv` contains every split's AP, ROC-AUC, Brier, cohort sizes and
score coverage. `ap_intervals.csv` holds the paired first-split contrasts.
`protocol.json` records the feature-table hashes and evaluation rules.
