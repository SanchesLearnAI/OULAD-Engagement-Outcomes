# First weighted TMA score and within-assessment rank: retrospective control

## Question and definitions

Does the TMA-relative behavior difference between eventual Fail and
Pass/Distinction records persist when students have both similar VLE clicks
and similar first weighted TMA marks? This extends the
[+21/+28 matched analysis](../tma_matched_21_28_20261103/README.md).
It is a **retrospective** comparison. OULAD records submission day and final
recorded score but **not the day the mark became available**. A score from a
submission made by the cutoff cannot automatically be treated as information
available for a live warning at that cutoff.

Choose the same earliest positive-weight TMA within each module presentation.
`first_tma_score` is the eventual recorded 0–100 mark only for a submission
dated on or before the observation cutoff. A missing mark stays missing.
`first_tma_rank_pct` is its average-tie percentile within that assessment,
among all cutoff-eligible scored records, including eventual Withdrawn. This
controls for assessment difficulty and grading distributions when describing
relative standing; the reference population changes slightly with the cutoff.
Neither score nor rank is used in the earlier `assessment_node` model results;
the separate [training ablation](../tma_score_model_20261103/README.md)
examines them retrospectively.

## Coverage and association

| Cutoff | Binary records | Submitted and recorded score | Fail with score | Pass with score | Distinction with score |
|---|---:|---:|---:|---:|---:|
| +21 days | 19,806 | 17,916 (90.46%) | 4,639 / 6,092 | 10,791 / 11,105 | 2,486 / 2,609 |
| +28 days | 19,806 | 17,940 (90.58%) | 4,653 / 6,092 | 10,801 / 11,105 | 2,486 / 2,609 |

Five additional Fail records submitted by each cutoff lack a recorded score.
Marks differ by final outcome among those with a recorded first TMA score:
mean 69.02 (Fail), 77.91 (Pass), and 87.77 (Distinction) at +21 days. The
corresponding means at +28 are essentially unchanged. Among scored binary
records, the observed Fail proportion falls from 53.36% in the bottom
within-assessment rank fifth to 11.44% in the top fifth at +21 days (52.86%
to 11.46% at +28). This is expected to be partly mechanical: the first TMA
has positive assessment weight and contributes to overall course results.
It should not be read as a new independent behavioral discovery.

The existing 5%-total-click pairs contain 4,004 (+21) and 3,917 (+28) pairs
with both marks recorded. Even at similar clicks, Fail records score 9.21
points lower at +21 [95% interval -9.80, -8.60] and 9.05 lower at +28
[-9.62, -8.42]. Their within-assessment ranks are lower by about 17.2
percentile points at either cutoff. Thus controlling clicks alone did not
control early assessed performance.

## Joint click and score matching

Within the same module presentation, greedily pair scored Fail and
Pass/Distinction records without reuse. Require first-TMA marks within five
points; allocate exact total-click matches first, then nearest clicks within
a 5% positive-click ratio. Zero clicks match zero only. The score and click
calipers are applied simultaneously, so the resulting pairs differ from the
original click-only pairs. This yields 3,200 pairs at +21 (68.98% of scored
Fail records) and 3,178 at +28 (68.30%). Mean absolute mark gaps are 2.47
and 2.48 points; mean relative click gaps are 1.04% at both cutoffs. The
residual mean Fail-minus-Pass score gaps are -0.39 and -0.30 points.

| Fail minus Pass/Distinction within joint pairs | +21 days | +28 days |
|---|---:|---:|
| Post-deadline active days | -0.84 [-0.98, -0.70] | -1.12 [-1.29, -0.93] |
| Pre-deadline last-three-day click share | +4.84 pp [3.40, 6.22] | +5.19 pp [3.78, 6.57] |
| Fraction of post weeks active | -5.13 pp [-6.17, -4.10] | -4.37 pp [-5.35, -3.50] |
| Submitted after scheduled due date, by cutoff | +5.69 pp [4.14, 7.12] | +5.82 pp [4.30, 7.27] |
| Post resource-type count | -0.255 [-0.326, -0.185] | -0.267 [-0.334, -0.199] |

Post active days are lower in Fail records in all 19 presentation slices at
each cutoff; the last-three-day share is higher in 18/19. The behavior
association therefore persists even when the compared students earned
similar first-TMA marks. These are pointwise connected-student-component
bootstrap intervals conditional on fixed greedy pairs, not adjusted for
multiple comparisons or rematching uncertainty.

Total-click matching still allows different pre/post click allocation. In a
further exploratory subset requiring both positive post-click counts within
20%, 634 (+21) and 949 (+28) pairs remain. Post active-day differences stay
negative: -0.88 [-1.20, -0.55] and -0.96 [-1.27, -0.66]. The forum share
difference is no longer clear at either cutoff; post resource-type count
is inconsistent across cutoffs. This subset changes the population and must
not be interpreted as a causal adjustment.

## Interpretation and files

First-TMA marks and ranks are strongly associated with eventual results, but
they do not absorb the observed post-deadline activity pattern. The scored
subset is selective: only about 76% of Fail records, versus roughly 97% of
Pass records, have a usable mark at either cutoff. Pair-level conclusions do
not represent students with no recorded first-TMA score. Grade release timing
is absent, and the mark itself contributes to the final outcome; therefore
this is an explanatory sensitivity study, not an operational forecasting
claim. The selected first-TMA assessment weights range from 5 to 16 in the
source metadata.

```bash
python -m oulad_analysis.tma_score_probe --data-dir data/raw --three outputs/node_3w --four outputs/node_4w --matched-dir outputs/tma_matched_21_28 --output-dir outputs/tma_score_probe --repeats 1000
```

`score_coverage.csv`, `rank_quintiles.csv`, `joint_balance.csv`,
`score_effects.csv`, and `joint_presentations.csv` are aggregate outputs.
`protocol.json` records definitions, seeds and source hashes. Student-level
scored features and pair identities remain in ignored `outputs/`.
