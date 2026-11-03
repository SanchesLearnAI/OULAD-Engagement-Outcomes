# First weighted TMA plus 14 days

Reproduce with:

```bash
python -m oulad_analysis.assessment_node --data-dir data/raw --output-dir outputs/node_new
```

The source is the unmodified OULAD release documented in `data/README.md`.
This folder contains aggregate outputs only. `node_features.parquet` remains
in the ignored output directory because it contains student-level records.

The first positive-weight TMA is available for 19 presentations in six
modules. Of 32,593 enrollment records, 30,059 are in those presentations;
24,125 have a known registration date no later than the deadline, remain
registered through deadline + 14 days and have no banked first TMA. Among
these, 19,806 have Pass, Distinction or Fail as their eventual result, and
4,319 have Withdrawn. The binary model excludes the latter, so it estimates
neither overall withdrawal risk nor the outcome mix among all early learners.

The observation window is relative days -13 through 14, with 14 days before
and 14 days after the deadline. The model uses only submission status as of
deadline + 14 and VLE clicks in that window. It does not use grades. The four
logistic models share the same student-grouped holdout and differ as follows:

| Model | Inputs beyond presentation context |
|---|---|
| B0 | Total clicks, squared log clicks, pre/post click totals |
| B1 | B0 plus pre/post active days, last-three-day share and post-period coverage |
| B2 | B0 plus as-of-cutoff submission indicators |
| B3 | B2 plus the B1 timing indicators |

On seed 42, the test set has 3,942 records from 3,708 students, with Fail
prevalence 31.28%. AP is B0 0.611512, B1 0.628761, B2 0.654318 and B3
0.668934. The B3-B2 difference is +0.014616, with a student-paired bootstrap
95% interval of [+0.007031, +0.022347]. Five grouped splits give B3-B2
differences of +0.00820 to +0.01619. These overlapping splits and the
previous exploration do not constitute an untouched external validation.

`node_models.csv` records all five grouped splits; `node_ap_intervals.csv`
contains three pointwise 1,000-resample differences from the first split;
`node_course_deltas.csv` describes module-level differences;
`node_year_transfer.csv` checks 2013 training against 2014 records with
repeated students removed. B3-B2 AP gain is +0.006740 for previously seen
modules and +0.003546 for the new CCC module, below the prespecified 0.01
practical reference. The new module's one-hot category is unseen in training.
`node_protocol.json` records cohort counts, the time rule, source hashes,
seed and bootstrap count. Module slices are descriptive, especially for
smaller courses. Dates are recorded in whole days and individual deadline
extensions are unavailable. See `METHODS.md` for inclusion rules.
