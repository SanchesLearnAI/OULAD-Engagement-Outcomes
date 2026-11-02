# Same Engagement, Different Outcomes

Group 9 data mining project using the [Open University Learning Analytics Dataset (OULAD)](https://doi.org/10.24432/C5KK69).

## Research question

Among students with similar activity during the first four weeks of a course, which timing and resource-use patterns are associated with passing rather than failing?

## Planned approach

- Join OULAD's studentInfo, studentVle, and vle tables. Each observation is one student in one module presentation.
- Use only days 0–27 to build features. Compare Pass/Distinction with Fail; describe Withdrawn separately.
- Within each module presentation, compare students with similar total clicks. Derive weekly activity and resource-type shares.
- Cluster early learning behavior, then describe each cluster and its outcome distribution.
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

Extract the public archive at `data/oulad.zip` into a local directory. The
workflow requires `studentInfo.csv`, `studentVle.csv` and `vle.csv`. Run:

```bash
python -m oulad_analysis --data-dir /path/to/oulad --output-dir outputs/run_001
```

The command creates a feature table and quality report, similar-click
comparisons, cluster profiles and plots, and grouped-holdout model results.
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

For HPC Phase II, [jobs/oulad_cpu.slurm](jobs/oulad_cpu.slurm) requests four
CPU cores, 16 GB RAM and 30 minutes on `i64m512u`. Verify the remote project,
environment and raw-data paths first. The script can extract an official
OULAD ZIP on the compute node if the CSVs are absent. Set a new `RUN_DIR` when
submitting; the script refuses to overwrite an existing run directory. No GPU
is requested.
