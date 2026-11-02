# Preliminary local analysis, 2026-11-02

These aggregate results were produced from the unmodified UCI OULAD archive
in `data/oulad.zip` with `python -m oulad_analysis --stage all --seed 42`.
The feature table and row-level predictions are intentionally omitted; rerun
the pipeline to regenerate them. The outputs include data-quality counts,
similar-click summaries, clustering diagnostics, model metrics and figures.

The model test set contains 4,504 student-presentation records (4,224 distinct
students); Fail prevalence is 31.35%. The full logistic model has average
precision 0.5952 versus 0.5665 for the click-plus-context baseline. These are
exploratory results from one grouped holdout split, not a final study report.
Model selection or feature changes should be evaluated within training data
before using a fresh holdout for a final estimate.
