# Same Deadline, Different Persistence

## Question

Within the same module presentation, among students whose post-deadline clicks after the first positive-weight TMA are comparable, are submission timing and post-deadline persistence associated with eventually failing rather than passing or earning a distinction? Withdrawn remains a separate outcome.

## Cohort

- Eligible records: 24,125
- Binary records: 19,806
- Fail records: 6,092 (30.8%)
- Presentations with a positive-weight TMA: 19

## Primary indicators

| Indicator | Pairs | Fail - Pass | 95% interval | p (Holm) |
|---|---:|---:|---|---:|
| `submitted_by_cutoff` | 4768 | -0.1235 | [-0.135, -0.112] | 0.0060 |
| `submitted_after_due` | 4768 | 0.0839 | [0.072, 0.096] | 0.0060 |
| `relative_submission_day` | 3859 | 1.3809 | [1.182, 1.568] | 0.0060 |
| `post_active_days` | 4768 | -0.3683 | [-0.440, -0.295] | 0.0060 |
| `post_longest_inactive_run` | 4768 | 0.3888 | [0.303, 0.473] | 0.0060 |
| `post_week2_share` | 4768 | -0.0368 | [-0.050, -0.024] | 0.0060 |

## Model ladder (main test, mean)

| Model | Fail AP | No-skill | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| M0 | 0.6544 | 0.3078 | 0.7585 | 0.1658 |
| M1 | 0.6574 | 0.3078 | 0.7606 | 0.1651 |
| M2 | 0.6611 | 0.3078 | 0.7647 | 0.1641 |
| M3 | 0.6608 | 0.3078 | 0.7653 | 0.1640 |
| RF_M0 | 0.6541 | 0.3078 | 0.7573 | 0.1903 |
| RF_M2 | 0.6641 | 0.3078 | 0.7671 | 0.1853 |

M4 contains the first-TMA score and is a retrospective upper bound only.

## Reproducibility

See README.md. Every stage writes `run.jsonl`, a stage summary, a handoff note and an atomic `STATUS.json` update; `validate_run.py` checks them.
