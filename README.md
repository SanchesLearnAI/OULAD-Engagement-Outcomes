# Same Engagement, Different Outcomes

Group 9 data mining project on the [Open University Learning Analytics Dataset
(OULAD)](https://doi.org/10.24432/C5KK69).

The study asks whether, **within the same module presentation and among students
with comparable VLE click volume, submission timing and post-deadline
persistence are associated with eventually failing the module**. The main
design anchors each presentation on the deadline `d` of its first
positive-weight tutor-marked assignment (TMA): engagement is observed for 14
days before (`d−13…d`) and 14 days after (`d+1…d+14`) that deadline, and Fail
records are paired 1:1 with Pass/Distinction records whose **post-deadline**
clicks are within 5%. Final Withdrawn records are described separately and are
excluded from the binary comparison. All findings are observational
associations, not causal effects.

Read first: the [frozen primary protocol](research/TMA_PRIMARY_PROTOCOL.md), the
[exploratory two-week protocol](research/TMA_TWO_WEEK_PROTOCOL_20261103.md),
the [two-week aggregate results](results/tma_two_week_20261103/README.md), and the
[2026-11-04 synthesis](research/TMA_SYNTHESIS_20261104.md). The `oulad-sem/`
pipeline now executes the frozen protocol end to end; `results/` keeps the
exploratory tables that led to it.

## Repository layout

Two code bases share one dataset and deliberately stay in separate directories:

| Path | Role | Tests |
|---|---|---|
| `oulad_analysis/`, `tests/` | Research and exploration code: cohort construction, pairing, clustering, feature ablations, model comparisons. Python 3.11, `unittest`, `duckdb`. | `python -m unittest discover -s tests -v` (28 tests) |
| `oulad-sem/` | Validation and delivery pipeline: frozen config with a config hash, gated stages S00–S12, JSONL run log plus schema validation, ZIP packaging. Python 3.11, `pytest`, own `Makefile`. | `cd oulad-sem && make test` (13 tests) |

`oulad_analysis/` is where exploratory questions are developed. `oulad-sem/`
is where a frozen design is executed end to end and packaged. Once the TMA
replacement in `oulad-sem/` lands, that pipeline carries the main study; the
research code stays as the exploration and reference implementation.

- `data/oulad.zip` — unmodified public archive, checksum in [data/README.md](data/README.md).
- `results/<name>_<date>/` — reviewed aggregate tables, protocols and figures.
- `research/` — protocols, synthesis notes and the code/data/experiment map.
- `outputs/` — student-level parquet, pair indices and predictions; git-ignored.

## Setting up

Python 3.11 is required. Each code base keeps its own dependency file; do not
merge them.

```bash
# research code, from the repository root
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# pipeline, in the same environment
python -m pip install pyyaml jsonschema pytest     # plus the requirements above
```

`requirements-lock.txt` and `oulad-sem/REQUIREMENTS.lock` record the versions
verified on the course HPC environment; use them to diagnose version problems.
The pipeline additionally ships `oulad-sem/config/config.example.yaml` and the
JSON schemas under `oulad-sem/docs/contracts/`.

## Data

Extract the committed archive; nothing else needs downloading:

```bash
mkdir -p data/raw && unzip -o data/oulad.zip -d data/raw              # research code
mkdir -p oulad-sem/data/raw/oulad
unzip -o data/oulad.zip -d oulad-sem/data/raw/oulad                    # pipeline
cp data/oulad.zip oulad-sem/data/raw/oulad/                            # S00 records the ZIP hash
```

Both `data/raw/` and `oulad-sem/data/` are git-ignored. The pipeline needs the
same seven CSVs; S00 refuses to continue if any is missing and it cannot
verify the archive hash.

## Running

Research code, in a **new** output directory each time:

```bash
python -m oulad_analysis.assessment_node --data-dir data/raw --output-dir outputs/node_2w --post-weeks 2
python -m oulad_analysis.tma_two_week --data-dir data/raw --node-dir outputs/node_2w \
    --output-dir outputs/two_week --seed 42 --repeats 1000 --bootstrap 500 --cluster-pairs 800
python -m oulad_analysis.tma_click_clusters --input-dir outputs/two_week \
    --output-dir outputs/click_clusters --seed 42 --repeats 1000
python -m oulad_analysis.tma_submission_timing --data-dir data/raw --node-dir outputs/node_2w \
    --two-week-dir outputs/two_week --cluster-dir outputs/click_clusters \
    --output-dir outputs/submission_timing --seed 42 --bootstrap 500 --repeats 1000
python -m oulad_analysis.tma_submission_persistence --data-dir data/raw --node-dir outputs/node_2w \
    --two-week-dir outputs/two_week --cluster-dir outputs/click_clusters \
    --output-dir outputs/submission_persistence
```

Pipeline, from `oulad-sem/`:

```bash
make test
make sim   && make validate RUN_ID=$(cat .last_sim_run)
make full  && make validate RUN_ID=$(cat .last_full_run)
```

Every stage writes `stages/Sxx.stage.json` and `handoff/Sxx.md` under
`runs/<run_id>/`; `scripts/validate_run.py` checks the JSONL log against the
schemas in `docs/contracts/` and must report 0 errors and 0 warnings. Any change
to `config/config.yaml` requires a `CHANGE_REQUEST.md` entry, a regenerated
`config/config.sha256`, and a new run id. The pipeline executes the TMA
two-week design; the earlier day-0–27 design is archived under
`results/oulad_sem_day0_27_baseline_20261108/`.

## Results index

Main design (TMA, 14 days before and after the deadline, post-deadline click
pairing):

- [Two-week aggregate results](results/tma_two_week_20261103/README.md) — pairing, balance, timing profiles, model ablations.
- [Post-click volume clusters](results/tma_click_clusters_20261103/README.md) — exploratory, within zero/low/high post-click groups.
- [Submission timing](results/tma_submission_timing_20261103/README.md) and [submission × persistence](results/tma_submission_persistence_20261104/README.md) — exploratory; the latter's `0.7136` combination was selected after inspection and is not a pre-registered main model.
- [Relative pre/post click gap](results/tma_relative_gap_20261103/README.md), [pre-deadline click burst](results/tma_predeadline_burst_20261103/README.md), [nonlinear mark interaction](results/tma_response_attributes_20261104/README.md) — exploratory ablations; the nonlinear interaction does not hold up in the year-transfer check.

Window sensitivity (+21/+28 days, clicks matched over the whole window):

- [+21/+28-day pairs and resources](results/tma_matched_21_28_20261103/README.md), [complete record](research/ACTIVE_TMA_21_28_RESEARCH.md), [node models](results/assessment_node_windows_20261103/README.md).

Historical baseline (day 0–27, the original proposal window):

- [Day 0–27 pairs](results/matched_20261102/README.md), [clustering and model diagnostics](results/revised_20261102/README.md), [timing concentration](results/timing_20261102/README.md), [preliminary aggregate](results/local_uci_20261102/README.md).
- The earlier `oulad-sem/` day-0–27 pipeline run is archived under [results/oulad_sem_day0_27_baseline_20261108](results/oulad_sem_day0_27_baseline_20261108/README.md) with its run id and config hash.

Index only, not part of the study narrative:

- `results/tma_score_probe_20261103/`, `results/tma_score_model_20261103/`, `results/tma_score_gap_20261103/` — these use the first-TMA mark, which is not available as an online feature at the cutoff; retrospective upper bounds only.
- `results/wu_method_probe_20261103/`, `results/wu_method_probe_28d_20261103/` — method probes adapted from an unrelated paper's withdrawal task, not a reproduction.
- `results/assessment_node_20261103/` — the 2/3/4-week window ladder used to choose the observation horizon.

Full index: [results/README.md](results/README.md). Research notes:
[research/README.md](research/README.md).

## Exploration and verification (beyond the frozen design)

The frozen pipeline above is the main result. Two later bodies of work sit on top
of it and are indexed in [results/README.md](results/README.md):

- **Deep exploration** — `oulad-sem/scripts/explore_tma_*.py`, outputs in
  `results/tma_deep_exploration_20261108/`. Descriptive analyses of return
  latency, multi-deadline trajectories, activity composition, day patterns,
  data minimisation, delay profiles and silence structure, plus an alert-capacity
  curve.
- **Chinese report** — `oulad-sem/Report_zh.pdf`, generated from the same tables by
  `oulad-sem/scripts/build_report_zh.py`. The English report is authoritative.
- **Verification** — `oulad-sem/scripts/verify_*.py` with protocols in
  `research/VERIFICATION_PLAN.md` and `research/TMA_DEEP_PROTOCOL.md`. The
  pre-registered increment check found no gain (ΔAP +0.0013, interval
  [−0.0008, +0.0032]); the delay typology and sequence-model advantages did not
  replicate.

Both reuse the same cohort and test records, so only the pre-registered check
(V1) counts as a frozen test; everything else stays exploratory. The deep
exploration's E2 block ("how many TMA deadlines fall inside the window") was
withdrawn after checking the data: the smallest gap between two consecutive
positive-weight TMA deadlines is 21 days, so the window never contains a second
deadline.

## Interpretation boundaries

1. AP means average precision for **Fail = 1**, not accuracy. Compare models only within the same cohort, cutoff and test records.
2. Pairing is 1:1 by design and cannot estimate the population failure rate; greedy matching is not globally optimal, and the fixed-pair component bootstrap ignores re-matching uncertainty.
3. The binary cohort excludes records that withdrew before the observation cutoff, so it is not a real-time early-warning system for all enrolled students.
4. The first-TMA mark itself contributes to the final result; models with marks are retrospective and cannot be presented as available online features.
5. The OULAD test splits have been inspected many times across rounds, so intervals and increments are exploratory.
6. VLE `activity_type` is a platform category, not content quality; `studentVle` has day-level clicks only, so study time cannot be measured.

## Evaluation and deliverables

Split by student ID before modeling; compare models on the same held-out
records. Report average precision for Fail, with the Fail prevalence as the
no-skill reference, alongside recall/F1 at a validation-chosen threshold, Brier
score and calibration. Check class balance, pairing balance, cluster stability,
tolerance sensitivity and consistency across presentations.

- Nov 7: data preparation, study population, features, exploratory analysis.
- Nov 20: clustering, exception analysis, model comparisons, ablations.
- Nov 27: poster submission; Dec 4: poster presentation.
- Dec 11: final report, code, README and project package (`Group_09_Project.zip`).

Use GitHub issues for tasks and decisions. Record processing rules, matching
thresholds, seeds and model settings. Do not commit extracted CSVs, student-level
outputs or predictions.

## Data source

Raw data: [UCI OULAD release](https://doi.org/10.24432/C5KK69) by Kuzilek,
Hlosta and Zdrahal, CC BY 4.0. The unmodified archive and its checksum are in
[`data/README.md`](data/README.md). Timing, clustering and model definitions for
the research code are in [METHODS.md](METHODS.md).
