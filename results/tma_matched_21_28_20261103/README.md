# Matched behavior at first weighted TMA +21 / +28 days

## Question and population

Within the same module presentation and at comparable total VLE clicks from
TMA due day -13 through day +21 or +28, how do eventual Fail records differ
from Pass/Distinction records? The two cutoffs are separate comparisons. The
first TMA with positive assessment weight defines the node; six modules and
19 presentations qualify. GGG has no qualifying TMA. Enrollment through the
cutoff and a non-Withdrawn final label are required. This retrospective binary
cohort contains the same 19,806 records at both cutoffs: 6,092 Fail, 11,105
Pass, and 2,609 Distinction. The eligible cohort also contains Withdrawn
records, but they are excluded from this binary comparison; see the node
window result for their counts.

The run reuses the saved 21- and 28-day node features and scans original
`studentVle.csv` once to aggregate resource clicks using `vle.csv` activity
types. All original event rows are preserved. Every student's resource counts
sum exactly to the corresponding node total and post clicks; no VLE events in
this selected population lacked a resource metadata match. The two cutoffs
have identical binary records, but matching is repeated separately because
click totals change with the cutoff.

## Matched comparison

Pairing is within module presentation, exact clicks first and then nearest
available clicks within a 5% positive-click ratio, without reusing a record.
Exact-only and 10% ratios are reported in the CSVs. The greedy matching is
reproducible but not globally optimal. Effects below are pair means (Fail
minus Pass/Distinction), with pointwise 95% connected-student-component
bootstrap intervals conditional on fixed pairs.

| Cutoff | 5% pairs | Fail coverage | Mean click gap | Log-click SMD |
|---|---:|---:|---:|---:|
| +21 days | 4,627 | 75.95% | 0.391% | -0.00052 |
| +28 days | 4,477 | 73.49% | 0.399% | -0.00070 |

For context, 21-day matched Fail and Pass/Distinction mean totals are 231.40
and 231.49 clicks; 28-day means are 284.43 and 284.65. Exact-only yields
3,062 and 2,777 pairs; 10% yields 4,727 and 4,583. The main directional
results below are similar under exact-only and 10% matching.

| Fail minus Pass/Distinction, 5% pairs | +21 days | +28 days |
|---|---:|---:|
| Clicks before/through due date | +8.34 [5.85, 10.63] | +11.15 [8.64, 13.83] |
| Clicks after due date | -8.44 [-10.74, -5.92] | -11.37 [-14.05, -8.83] |
| Submitted by cutoff | -7.13 pp [-8.08, -6.18] | -5.81 pp [-6.80, -4.83] |
| Submitted after due date but by cutoff | +6.81 pp [5.60, 8.04] | +6.90 pp [5.66, 8.15] |
| Post-deadline active days | -0.93 [-1.04, -0.80] | -1.27 [-1.41, -1.12] |
| Share of pre-deadline clicks in last 3 days | +4.69 pp [3.51, 5.98] | +4.77 pp [3.61, 6.02] |
| Fraction of post weeks active | -7.06 pp [-8.01, -6.16] | -6.51 pp [-7.37, -5.75] |

At both cutoffs, lower post active days and lower submission-by-cutoff occur
in all 19 presentation slices with at least 30 pairs. The last-three-day
share is higher in Fail records in 19/19 slices at day +21 and 17/19 at day
+28. These are outcome associations among selected pairs, not causal effects
of changing student behavior. Submission status is observable by the cutoff,
but eventual scores are not used.

## Resource findings and volume sensitivity

Fail records have fewer post-deadline resource types at the main total-click
match: -0.278 types at +21 and -0.270 at +28. Post resource entropy is lower
by 0.058 and 0.057, respectively. Forum (`forumng`) has a lower share of
post-deadline clicks by 2.09 and 1.72 percentage points, with the same
direction in 17/19 and 14/19 presentation slices. Type availability differs
by module (for example, some modules have no quiz type), and all type-share
comparisons are exploratory and compositional.

Total-click matching leaves a real pre/post allocation difference. To probe
whether resource diversity simply tracks post-click volume, a sensitivity
subset retains only main pairs with positive post clicks and post-click totals
within a further 20% ratio (867 pairs at +21; 1,211 at +28). The resource-type
count differences become +0.044 [-0.065, 0.144] and -0.055 [-0.140, 0.038];
the entropy differences also include zero. Post active-day differences persist
at -0.70 [-0.94, -0.44] and -1.03 [-1.31, -0.74]. Forum share remains lower
at -2.20 pp [-3.80, -0.65] and -1.34 pp [-2.58, -0.13]. A stricter 5%
post-click subset retains just 249/382 pairs; its forum-share intervals cross
zero, while active-day differences remain negative. These are post-selection
sensitivity checks, not independently matched primary cohorts. Conditioning
on post volume changes the estimand and can select on an outcome-associated
behavior; it should not be used to claim an independent causal effect.
If the question requires similar **post-deadline** click totals, these
selected-pair subsets are not a replacement for rematching on post clicks
within each presentation and checking pre-deadline activity balance.

## Prediction ablation

B2 contains module presentation, total/pre/post clicks, and cutoff submission
status. B3 adds six timing features. B4 adds resource-type counts, entropy,
and all observed all-window and post-window activity-type shares. All use the
same grouped student split. On seed 42 at +21, AP is 0.66845 (B2), 0.68111
(B3), 0.67835 (B4); at +28 it is 0.68231, 0.69837, 0.69090. The paired
student-bootstrap B4−B3 AP is -0.00276 [-0.00854, 0.00195] at +21 and
-0.00747 [-0.01689, 0.00243] at +28. Resource gains vary in sign across
five overlapping grouped splits. Adding this raw type-share block does not
show a reliable incremental predictive gain; timing does improve B3 over B2
within either fixed cutoff. Higher absolute AP at +28 reflects more observed
time, not a method improvement at the same decision time.
Because B3 adds pre-deadline and post-deadline timing features together, its
AP gain cannot be attributed entirely to post-deadline behavior.

## Interpretation and limits

The most defensible behavioral description is a timing shift: with similar
overall clicks, Fail records concentrate more activity before the first
weighted TMA deadline, sustain fewer active days afterward, and submit by
the cutoff less often. Resource diversity is partly explained by post-click
volume; individual type shares merit further course-level study rather than
a broad deep/shallow claim. This result does not require replacing the
conditional-matching approach, but it does not justify a causal or
all-student early-warning claim. The binary cohort excludes eventual
Withdrawn records and those unregistered by each cutoff. The earliest
positive-weight TMA has different dates across presentations, so these
cutoffs are not a common course week. Pair selection and many tested features
make pointwise intervals exploratory; no multiple-testing correction or
rematching uncertainty is included. The fixed test split was explored in
earlier runs and is not a new untouched validation sample.

## Reproduction and files

Run after the 3- and 4-week `assessment_node` commands, using empty output
directories:

```bash
python -m oulad_analysis.tma_matched --data-dir data/raw --three outputs/node_3w --four outputs/node_4w --output-dir outputs/tma_matched_21_28 --repeats 1000 --bootstrap 500
```

`balance.csv`, `effects.csv`, `presentations.csv`,
`post_volume_sensitivity.csv`, `models.csv`, and `ap_intervals.csv` contain
aggregate evidence. `protocol.json` records rules, source hashes, and seeds.
Student-level features and pair indices remain in ignored `outputs/` only.
