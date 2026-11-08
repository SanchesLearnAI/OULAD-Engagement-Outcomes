# How to rebuild these outputs

The aggregate CSV/JSON files in this folder are committed. The **student-level**
files are not: they carry one row per enrolment and are git-ignored on purpose
(`results/tma_deep_exploration_20261108/*records.parquet`,
`a2_type_coverage.csv`).

## Prerequisites

The scripts read the frozen pipeline's intermediate tables, which are also
git-ignored:

```bash
cd oulad-sem
python -m venv .venv && source .venv/bin/activate
python -m pip install -r ../requirements.txt pyyaml jsonschema pytest
# data
mkdir -p data/raw/oulad && unzip -o ../data/oulad.zip -d data/raw/oulad
cp ../data/oulad.zip data/raw/oulad/
# run through S06 at least; `make full` runs everything
make full
```

`make full` produces `data/interim/master.parquet`,
`data/processed/features.parquet`, `data/processed/features_scores.parquet`,
`data/processed/splits.json` and `outputs/pairs_05.parquet`.

## Commands and the exact settings used

Run from `oulad-sem/`; the output directory must be this one.

| Script | Command | Notes |
|---|---|---|
| Deep exploration E1–E10 | `python scripts/explore_tma_deep.py --repeats 200` | `--repeats` only affects E5's bootstrap; the committed `e5_bootstrap_and_subgroups.csv` used 200 |
| Patterns E6, E7 | `python scripts/explore_tma_patterns.py --permutations 1000` | the script default is 200; the committed file's smallest non-zero `p_perm` is 0.005, so **the committed table came from 200 permutations** — rerun with 1000 for the quoted floor of 0.001 |
| Delay E8, E9 | `python scripts/explore_tma_delay.py` | no options |
| Increment E10 | `python scripts/explore_tma_incremental.py --repeats 400` | |
| Follow-ups A1, A2, B1 | `python scripts/explore_tma_followups.py` | no options |
| V1 frozen increment | `python scripts/verify_frozen_increment.py --repeats 400` | reads `e9_reengagement_records.parquet` and `e2_anchor_records.parquet`, so run the exploration first |
| V2 transfer summary | `python scripts/verify_transfer_summary.py` | reads only committed pipeline tables |
| V3 subgroup MDE | `python scripts/verify_subgroup_mde.py --repeats 400` | |
| V4 delay profiles | `python scripts/verify_delay_profiles.py` | reads four E8 outputs |
| V5 sequence baselines | `python scripts/verify_sequence_models.py --repeats 200 --epochs 40` | sklearn MLP; PyTorch was unavailable |

## Known reproducibility caveats

1. **E6's permutation p-values were rewritten once without a code change.** The
   committed table sits at the `1/200` floor; a 1000-permutation rerun will
   produce different (smaller) values. Both are valid; the docs must state which
   run they quote.
2. **E5's subgroup p-values were removed.** The earlier version computed a
   pseudo p-value that could never fall below 1.0, so its "BH p = 1.0" column was
   vacuous. Subgroup intervals now come from `verify_subgroup_mde.py`.
3. **E2 was reworked.** The original version assumed a second TMA deadline could
   fall inside the 14-day window; the smallest gap between consecutive TMAs in
   OULAD is 21 days, so the count was always one. `e2_deadline_spacing.csv` now
   records that fact and only the anchor submission facts are kept.
4. **V1's frozen feature set changed** after the E2 rework: three degenerate
   columns were replaced by two anchor columns. A de-duplicated arm is reported
   alongside because `longest_silence` is identical to `post_longest_inactive_run`.
5. **`provenance.json` hashes the script that ran**, so it changes whenever a
   script is edited. Compare it against `git log` rather than treating it as an
   immutable signature of the committed files.
