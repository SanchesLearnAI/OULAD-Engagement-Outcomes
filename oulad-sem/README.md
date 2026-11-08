# OULAD — Same Deadline, Different Persistence

Group 9 course project. The unit is an enrolment (`code_module`, `code_presentation`, `id_student`).

## Question

Within the same module presentation, among students with comparable VLE click volume in the 14 days after the first positive-weight TMA deadline, which submission-timing and post-deadline persistence patterns are associated with eventually failing rather than passing or earning a distinction? Final Withdrawn records are described separately.

## Design in one table

| Item | Value |
|---|---|
| Anchor | Deadline `d` of the first `TMA` with `weight > 0` per presentation |
| Windows | `d-13 … d` and `d+1 … d+14` |
| Cohort | registered by `d`, still enrolled at `d+14`, not banked |
| Label | Fail = 1, Pass/Distinction = 0, Withdrawn excluded |
| Pairing | 1:1 within presentation on `post_clicks`, 5% tolerance |
| Primary model | M2 (volume + submission + timing + days early/late) |

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install pandas numpy scikit-learn scipy pyyaml jsonschema \
    pyarrow matplotlib seaborn statsmodels pytest duckdb
make test
make sim   && make validate RUN_ID=$(cat .last_sim_run)
make full  && make validate RUN_ID=$(cat .last_full_run)
```

The frozen configuration is `config/config.yaml` with its SHA-256 in `config/config.sha256`. Any change needs a `CHANGE_REQUEST.md` entry and a new run id.

## This run

- run id: `run-20261108T101537Z-28e708` (mode `full`)
- config: `sha256:6772d0022b7e1d6fecd07ed46d7f0b461f770980135861ffbec6c3295e0e7926`
- eligible records: 24,125
- binary records: 19,806, Fail 6,092 (30.8%)
- presentations: 19
- primary model M2 Fail AP: 0.6611

## Data

See `DATA_DOWNLOAD.md`. Extract the seven CSVs into `data/raw/oulad/` (and keep `oulad.zip` there so S00 can record its hash). Raw CSVs, `data/interim/`, `data/processed/` and `runs/` are git-ignored.

## Outputs

`outputs/tables/t01`–`t11` hold the aggregate tables cited in the report; `outputs/figures/` holds the figures. Student-level features, pair indices and cluster labels stay in the ignored directories.
