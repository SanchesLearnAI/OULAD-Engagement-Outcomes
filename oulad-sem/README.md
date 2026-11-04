# OULAD — Same Engagement, Different Outcomes

Group 9 course project. The unit is an enrolment (`code_module`, `code_presentation`,
`id_student`). The primary target is Pass/Distinction vs Fail, with Withdrawn excluded
(T1) and analysed separately in T3 / competing risks.

## Data

Download the public OULAD archive from UCI:

```bash
mkdir -p data/raw/oulad
curl -L --fail --retry 3 -o data/raw/oulad/oulad.zip \
  "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
shasum -a 256 data/raw/oulad/oulad.zip
unzip -o data/raw/oulad/oulad.zip -d data/raw/oulad/
```

The archive must yield seven CSVs: `courses.csv`, `assessments.csv`, `vle.csv`,
`studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`,
`studentVle.csv`. The raw archive is not included in the ZIP.

## Environment

Python 3.10+; the recorded environment is in `REQUIREMENTS.lock` and
`runs/run-20261104T045841Z-4f3088/env.json`. Core dependencies: pandas, numpy, scikit-learn,
scipy, pyyaml, jsonschema, pyarrow, matplotlib, seaborn, statsmodels.
`lifelines` is optional; cause-specific Cox models use `statsmodels`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r REQUIREMENTS.lock
```

## Reproducibility checklist

1. Record the raw archive hash and seven CSV row counts (S00 `manifest.json`,
   `DATA_DOWNLOAD.md`).
2. Verify `config/config.sha256` matches the frozen config; every event records
   `config_hash`.
3. Use the recorded `run_id`; do not edit `runs/<run_id>/`.
4. Run `python3 scripts/validate_run.py --run-id <run_id>` and require 0 errors /
   0 warnings before treating outputs as final.
5. `STATUS.final.json` is written by S12 and included in the ZIP.

## Frozen configuration

`config/config.yaml` is copied from `02_contracts/config.example.yaml`; its sha256 is in
`config/config.sha256`. Do not edit it in place. Any change requires `CHANGE_REQUEST.md`
and a new `run_id`.

## One-command workflow

```bash
make test     # pytest -q
make sim      # S00-S12 on the 500-enrolment sample
make validate RUN_ID=<sim-run-id>
make full     # S00-S12 on all 32,593 enrolments
make validate RUN_ID=<full-run-id>
```

Each stage invocation enforces the previous-stage gate:

```bash
export PYTHONHASHSEED=0
RUN_ID="sim-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
for S in S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
  python3 scripts/run_stage.py --stage "$S" --mode simulation --run-id "$RUN_ID" \
    --config config/config.yaml || break
done
python3 scripts/validate_run.py --run-id "$RUN_ID"
```

After the simulation is green, run the full data with a new `run-` id:

```bash
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
for S in S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
  python3 scripts/run_stage.py --stage "$S" --mode full --run-id "$RUN_ID" \
    --config config/config.yaml || break
done
python3 scripts/validate_run.py --run-id "$RUN_ID"
```

## Outputs

- `runs/<run_id>/run.jsonl` — append-only event log.
- `runs/<run_id>/stages/Sxx.stage.json` — stage summary.
- `runs/<run_id>/handoff/Sxx.md` — handoff note.
- `STATUS.json` — atomic run state.
- `outputs/tables/` — cohort, balance, stability, model, cutoff, competing-risk,
  specification and diagnostic tables.
- `outputs/figures/` — FIG 01-09 style figures.
- `Report.pdf`, `REPORT.md`, `README.md`, `Poster.pdf`, `Group_09_Project.zip`.
- `Group_09_Project.zip` is the only submission archive; `Poster.pdf` is submitted
  separately. `STATUS.final.json` is included in the ZIP.

Simulation outputs are written under `outputs/sim/`, `data/interim/sim/`, and
`data/processed/sim/` so they cannot overwrite full-run artifacts.

## Known limitations

Observational associations only; matching balances measured W1 variables but cannot remove
unmeasured confounding. Clicks are a weak proxy for study time. The simulation sample and
small held-out test sets make point estimates noisy. Withdrawn is not a missing label.
The full run has more power but the same design limitations.

TMA-based appendix analyses were not attempted: the TMA+21/+28 window overlaps W1/W2 and
conditions on continued enrolment, the TMA+14 anchor varies by presentation, and first-TMA
scores are not available at the early cutoff. These are alternative, post-hoc estimands, not
early-warning evidence.
