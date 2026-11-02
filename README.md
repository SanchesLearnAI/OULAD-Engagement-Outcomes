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

Use GitHub Issues to track tasks and decisions. Keep data processing rules, matching thresholds, random seeds, and model settings documented so results can be reproduced. Do not commit student-level raw data or personal identifiers to this repository.

## Python environment

The project uses Python 3.11. Install the core analysis libraries from `requirements.txt`; `requirements-lock.txt` records the versions verified on the course HPC environment. The core workflow uses CPU-based data processing and classical machine learning, so no GPU framework is required for setup.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
