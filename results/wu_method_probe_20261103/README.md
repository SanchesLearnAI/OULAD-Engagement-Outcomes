# Wu et al. model-method probe on the TMA outcome task

This is an **adaptation**, not a numerical reproduction of Wu, Duan and Luo
(2026), *PLOS ONE*, DOI [10.1371/journal.pone.0352867](https://doi.org/10.1371/journal.pone.0352867).
Their published task is withdrawal risk from days 0–27, with course-presentation
holdout, broad early-window features, calibrated HGB, fairness and intervention
policy analyses. Our task here is Fail versus Pass/Distinction among students
still registered three weeks after the first positive-weight TMA; the
observation window and outcome therefore differ. The authors' archived code
and HGB/calibration settings were inspected at
[Zenodo 20388848](https://doi.org/10.5281/zenodo.20388848). This project
implements the relevant model protocol independently; the authors' full
pipeline was not run for this comparison.

Run with the ignored three-week feature table:

```bash
python -m oulad_analysis.wu_method_probe --features outputs/node_3w/node_features.parquet --output-dir outputs/wu_probe_new
```

The comparison uses exactly the same input columns and test rows within each
split. `project_lr` is the existing unweighted logistic regression;
`paper_weighted_lr` uses the paper's balanced liblinear logistic settings;
`paper_hgb` uses 200 iterations, learning rate 0.05, max depth 6;
`paper_hgb_sigmoid` fits the HGB on 80% of the training partition and fits a
sigmoid calibrator on the remaining stratified 20%. Every imputer, scaler and
encoder is fitted on training data only. Calibration uses no test labels.

| Holdout | Test n | Model | Fail AP | Brier | Top-10% PPV |
|---|---:|---|---:|---:|---:|
| Student ID | 3,942 | Project LR | 0.681108 | 0.161083 | 0.850633 |
| Student ID | 3,942 | Paper HGB | 0.685534 | 0.159854 | 0.865823 |
| Student ID | 3,942 | Paper HGB + sigmoid | 0.683733 | 0.159918 | 0.863291 |
| Course presentation | 3,253 | Project LR | 0.550296 | 0.187043 | 0.723926 |
| Course presentation | 3,253 | Paper HGB | 0.556716 | 0.170576 | 0.714724 |
| Course presentation | 3,253 | Paper HGB + sigmoid | 0.562778 | 0.171688 | 0.742331 |
| Course presentation, repeated test students removed | 3,146 | Project LR | 0.550120 | 0.186790 | 0.723810 |
| Course presentation, repeated test students removed | 3,146 | Paper HGB + sigmoid | 0.563844 | 0.170829 | 0.739683 |

On the student holdout, HGB−project LR AP is +0.004426 with a paired
student-bootstrap 95% interval of [−0.002720, +0.011265]. Calibrated
HGB−project LR is +0.002625 [−0.005625, +0.009977]. Calibration did not
improve AP or Brier versus the uncalibrated HGB on this split. The original
paper uses calibration for probability reliability and decision policies,
not as a guarantee of better AP.

The paper-style course-presentation holdout has 107 student IDs also present
in training because people can appear in multiple presentations. Excluding
those test records leaves 3,146 rows and changes calibrated HGB−LR AP to
+0.013724, but its presentation-bootstrap interval is [−0.005167,+0.016022].
Only a few presentations are held out, so that interval is a coarse
sensitivity check. All intervals are exploratory: this dataset and previous
holdouts have already been examined repeatedly.

**Decision:** HGB is a reasonable secondary model, but this experiment does
not establish a reliable gain over the simpler logistic regression for the
current Fail/Pass question. The paper's strongest contribution is its
withdrawal-focused evaluation of calibration, fairness and alert policies;
none of those findings can be transferred numerically to this outcome.
`model_comparison.csv`, `paired_ap_intervals.csv` and `protocol.json` contain
the aggregate evidence. Student-level predictions are not published here.
