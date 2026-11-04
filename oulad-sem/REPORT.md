# Same Engagement, Different Outcomes — Group 9

## Problem

Among students with similar first-four-week VLE engagement, which temporal and resource-use patterns are associated with passing rather than failing? Withdrawn students are analysed as a separate outcome.

## Data and leakage controls

- Unit: enrolment (module x presentation x student); 32,593 enrolments in full OULAD.
- Strict windows: W1 days 0-27; W2 days 28 to min(55, first TMA-1), with a fixed 28-55 fallback when the first TMA is before week 4.
- No score, banked flag, final result, or post-cutoff click in strict features.
- Grouped splits; temporal and LOMO folds are student-disjoint.

## Main results

- M0: PR-AUC 0.816, ROC-AUC 0.701, Brier 0.220.
- M1: PR-AUC 0.827, ROC-AUC 0.724, Brier 0.213.
- M2: PR-AUC 0.827, ROC-AUC 0.724, Brier 0.213.
- M3: PR-AUC 0.828, ROC-AUC 0.729, Brier 0.211.
- M4: PR-AUC 0.843, ROC-AUC 0.744, Brier 0.207.
- M5: PR-AUC 0.838, ROC-AUC 0.738, Brier 0.204.

M4 includes assessment scores and is an upper bound only.

## Reproducibility

See README.md. Every stage writes run.jsonl, a stage summary, a handoff note, and an atomic STATUS.json update.
