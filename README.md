# Same Engagement, Different Outcomes

Group 9 data mining project using the [Open University Learning Analytics Dataset (OULAD)](https://doi.org/10.24432/C5KK69).

## Research question

Within the same module presentation, among students with comparable VLE click
volume observed by 21 or 28 days after the first positive-weight tutor-marked
assignment (TMA) deadline, which submission, timing, and resource-use patterns
are associated with eventual Pass/Distinction rather than Fail? The two cutoffs
are separate analyses. The VLE observation window starts 13 days before the
deadline; Withdrawn is described separately.

The implemented TMA analysis matches Fail with Pass/Distinction records at each
cutoff within a module presentation and compares submission, timing, and
resource-use features. A separate student-grouped model test checks whether
timing and resource features add predictive information. The earlier day-0–27
matched analysis remains a historical baseline.
The Wu et al. model-method probes at
[+21 days](results/wu_method_probe_20261103/README.md) and
[+28 days](results/wu_method_probe_28d_20261103/README.md) test HGB and
sigmoid calibration on this Fail/Pass task. They adapt model settings rather
than reproducing that paper's withdrawal results.

## Project map and research status

- [Current TMA +21/+28-day research record](research/ACTIVE_TMA_21_28_RESEARCH.md): cohort, click matching, behavior differences, resource and score sensitivity, models, transfer checks, and limitations.
- [Research status](research/RESEARCH_STATUS_20261103.md): distinguishes the original day-0–27 question, the current TMA question, and the separate post-deadline non-submission candidate.
- [Code, data, and experiment map](research/CODE_DATA_EXPERIMENT_MAP.md): source tables, modules, local student-level outputs, and aggregate results.
- [Research notes and scripts](research/README.md) and [aggregate result index](results/README.md).

The original proposal remains background for a later research-question revision. The current +21/+28-day analysis is the implemented main question; the non-submission subgroup has feasibility and preliminary strata, but its controlled three-outcome analysis is still pending.

## Current analyses

**Primary TMA comparison:** see
[`results/tma_matched_21_28_20261103`](results/tma_matched_21_28_20261103/README.md).
It uses exact, 5%, and 10% total-click matching at each cutoff. A secondary
sensitivity analysis also restricts post-deadline clicks, since similar total
clicks can hide different pre/post allocation. Resource clicks are joined to
the original `vle.csv` activity types and reconciled to the node totals.

**First-TMA mark sensitivity:**
[`results/tma_score_probe_20261103`](results/tma_score_probe_20261103/README.md)
compares the same behavior among students with both similar clicks and similar
first weighted TMA scores. It reports within-assessment score ranks. OULAD has
no mark-release timestamp, so this is retrospective analysis rather than a
claim that marks were available at the observation cutoff.
The [full score/rank training round](results/tma_score_model_20261103/README.md)
compares the timing baseline with score, training-only rank, and all resource
features on the same student-grouped holdouts. Its gain is also retrospective.

**Earlier four-week baseline:** compare Fail against Pass/Distinction within the same
module presentation and closely matched day-0–27 click counts. Run `--stage
matched` after features (also included in `--stage all`). Read
`matched_balance.csv`, `matched_effects.csv`, and
`matched_presentation_effects.csv` first. The original all-nonzero-student
clustering is supplementary exploratory EDA; its high/low activity groups do
not answer the research question. Classification on unmatched student holdouts
is a separate incremental-prediction check, not a substitute for this comparison.
The earlier matched results are in
[`results/matched_20261102`](results/matched_20261102/README.md).

**TMA prediction check:** `oulad_analysis.assessment_node` evaluates days 1–21 and
1–28 after the first weighted TMA deadline; see
[`results/assessment_node_windows_20261103`](results/assessment_node_windows_20261103/README.md).

The following is the earlier four-week pipeline:

- Join OULAD's studentInfo, studentVle, and vle tables. Each observation is one student in one module presentation.
- Use only days 0–27 to build features. Compare Pass/Distinction with Fail; describe Withdrawn separately.
- Within each module presentation, compare students with similar total clicks. Derive weekly activity and resource-type shares.
- Use clustering only as supplementary exploration; do not present uncontrolled activity clusters as profiles at comparable engagement.
- Compare an engagement-only logistic regression with models that also use behavioral features. Use a random forest to check nonlinear patterns.
- Review high-activity failures and low-activity successes as descriptive exceptions.

## Evaluation

Split by student ID before modeling. Compare models on the same unmatched held-out set using recall, F1, and precision-recall AUC. Check class balance, cluster stability, sensitivity to click-range thresholds, and consistency across module presentations. This is an observational study; findings describe associations, not causal effects.

## Deliverables and dates

- Nov 7: data preparation, study population, features, and exploratory analysis.
- Nov 20: clustering, exception analysis, model comparisons, and ablations.
- Nov 27: poster submission.
- Dec 4: poster presentation.
- Dec 11: final report, code, README, and project package.

## Collaboration

Use GitHub Issues to track tasks and decisions. Keep data processing rules, matching thresholds, random seeds, and model settings documented so results can be reproduced. The public UCI dataset archive is stored in `data/oulad.zip`; do not commit extracted CSVs, private data, or row-level model predictions.

## Python environment

The project uses Python 3.11. Install the core analysis libraries from `requirements.txt`; `requirements-lock.txt` records the versions verified on the course HPC environment. The core workflow uses CPU-based data processing and classical machine learning, so no GPU framework is required for setup.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the analysis

For the first weighted TMA experiment, also extract `assessments.csv` and
`studentAssessment.csv` from the same OULAD archive. Run it in a separate new
output directory:

```bash
python -m oulad_analysis.assessment_node --data-dir /path/to/oulad --output-dir outputs/node_2w --post-weeks 2
python -m oulad_analysis.assessment_node --data-dir /path/to/oulad --output-dir outputs/node_3w --post-weeks 3
python -m oulad_analysis.assessment_node --data-dir /path/to/oulad --output-dir outputs/node_4w --post-weeks 4
python -m oulad_analysis.assessment_node_compare --two outputs/node_2w --three outputs/node_3w --four outputs/node_4w --output outputs/node_common_cohort.csv
python -m oulad_analysis.tma_matched --data-dir /path/to/oulad --three outputs/node_3w --four outputs/node_4w --output-dir outputs/tma_matched_21_28
python -m oulad_analysis.tma_score_probe --data-dir /path/to/oulad --three outputs/node_3w --four outputs/node_4w --matched-dir outputs/tma_matched_21_28 --output-dir outputs/tma_score_probe
python -m oulad_analysis.tma_score_model --three outputs/tma_score_probe --four outputs/tma_score_probe --output-dir outputs/tma_score_model
```

The command writes `node_protocol.json`, `node_models.csv`,
`node_ap_intervals.csv`, `node_course_deltas.csv` and an ignored student-level
`node_features.parquet`; `node_year_transfer.csv` checks 2013-to-2014 transfer.
`--post-weeks` accepts 2, 3, or 4; each run needs a new output directory.
The comparison command restricts all three runs to records present at every
cutoff and uses the same student split at each horizon. It is a retrospective
window comparison because membership requires remaining enrolled through the
four-week cutoff. Read same-horizon B3-B2 gains first; AP across horizons
uses different information times and should not be called a method gain.
The TMA experiment applies to
six modules and 19 presentations with a positive-weight TMA. It excludes
banked submissions and records that are not registered at the deadline or
have unregistered by the observation cutoff. Pass/Distinction versus Fail is
evaluated among the remaining non-Withdrawn records; Withdrawn counts are
retained in the protocol. See [METHODS.md](METHODS.md) for exact boundaries.

Extract the public archive at `data/oulad.zip` into a local directory. The
workflow requires `studentInfo.csv`, `studentVle.csv`, `vle.csv`, and
`studentRegistration.csv`. Run in a **new** output directory:

```bash
python -m oulad_analysis --data-dir /path/to/oulad --output-dir outputs/run_001
```

The command creates a feature table and quality report, similar-click
comparisons, cluster profiles and plots, and grouped-holdout model results.
It also records input hashes, code revision, Python and dependency versions in
`run_manifest.json`. Existing results are never overwritten. Use a separate
output directory and `--deduplicate-events` for exact-duplicate sensitivity.
The main analysis preserves the original source rows.
`eligible_by_presentation.csv` describes students still registered through day
27; it is a sensitivity population, not the default binary-model cohort.
`click_balance.csv` reports residual click differences within strata, and
`similar_click_ci.csv` gives student-bootstrap intervals for selected features.
`balanced_similar_click_comparison.csv` restricts the 20-bin comparison to
strata with absolute standardized click difference at most 0.1 and reports
how many rows remain; it is a sensitivity check, not causal matching.
Model outputs now include nonlinear-click controlled baselines and
`robustness.csv` (five grouped splits and a 2013-to-2014 check). These remain
exploratory evaluations because the original holdout has already been inspected.
Cluster diagnostics report click distributions, module composition, and the
time-only clustering comparison.
The timing extension adds weekly evenness, active weeks, maximum calendar-week
and rolling-week shares, and evenness among active days. See the feature
definitions in [METHODS.md](METHODS.md). `timing_summary.csv`,
`timing_quantiles.csv`, and `timing_distribution.png` distinguish broad temporal
coverage from concentrated activity. The experimental `concentration_full_lr`
model tests incremental value over the existing complete model on identical
student splits; older feature tables must be regenerated before running it.
Individual stages can be run with `--stage features|compare|cluster|model`;
the latter three read `features.parquet` from the output directory. See
[METHODS.md](METHODS.md) for feature definitions, inclusion rules and model
settings. A synthetic end-to-end check is available with
`python -m unittest discover -s tests -v`.

The raw data are from the [UCI OULAD release](https://doi.org/10.24432/C5KK69)
by Kuzilek, Hlosta and Zdrahal, licensed CC BY 4.0. The unmodified source
archive is committed with [source and checksum details](data/README.md).
Extracted CSVs and student-level outputs are excluded from Git. The
`results/local_uci_20261102` directory contains preliminary aggregate results.
The revised exploratory run and its checks are in
[`results/revised_20261102`](results/revised_20261102/README.md).
Timing concentration distributions and the incremental-feature experiment are
in [`results/timing_20261102`](results/timing_20261102/README.md).

For HPC Phase II, [jobs/oulad_cpu.slurm](jobs/oulad_cpu.slurm) requests four
CPU cores, 16 GB RAM and 30 minutes on `i64m512u`. Verify the remote project,
environment and raw-data paths first. The script can extract an official
OULAD ZIP on the compute node if the CSVs are absent. Set a new `RUN_DIR` when
submitting; the script refuses to overwrite an existing run directory. No GPU
is requested.
