# Wu-style model-method probe at first weighted TMA +28 days

This is an adaptation of the HGB and sigmoid-calibration settings from
[Wu, Duan and Luo (2026)](https://doi.org/10.1371/journal.pone.0352867),
with the same model features and split definitions as the earlier
[+21-day probe](../wu_method_probe_20261103/README.md). It is **not** a
reproduction of the paper's withdrawal task. Here the target is eventual Fail
versus Pass/Distinction among students retained through TMA due day +28. The
source feature table's SHA-256 is recorded in `protocol.json`.

| Split | Test records | Project LR AP / Brier | HGB AP / Brier | HGB + sigmoid AP / Brier |
|---|---:|---:|---:|---:|
| Student holdout | 3,942 | 0.69837 / 0.15690 | 0.70283 / 0.15582 | 0.70305 / 0.15562 |
| Presentation holdout | 3,253 | 0.54108 / 0.21265 | 0.54307 / 0.19023 | 0.54284 / 0.19033 |
| Presentation holdout, repeated students removed | 3,146 | 0.54061 / 0.21300 | 0.54325 / 0.18971 | 0.54303 / 0.18998 |

Student-holdout HGB−LR AP is +0.00446, with a paired student-bootstrap 95%
interval [-0.00334, 0.01154]. HGB + sigmoid−LR is +0.00468
[-0.00319, 0.01164]; sigmoid−uncalibrated HGB is +0.00022
[-0.00452, 0.00464]. Presentation-holdout intervals also include zero for AP
differences. The Brier loss is lower for HGB on this one presentation split,
but sigmoid calibration does not materially improve HGB's Brier score there.
Presentation-group intervals are especially unstable with few held-out
groups. This probe does not establish a reliable AP gain from replacing the
current model and does not answer the matched behavioral question.

```bash
python -m oulad_analysis.wu_method_probe --features outputs/node_4w/node_features.parquet --output-dir outputs/wu_probe_28d
```

`model_comparison.csv` contains AP, ROC-AUC, Brier and top-10% metrics;
`paired_ap_intervals.csv` contains paired AP differences. The model uses no
resource-type features; the resource ablation belongs to the matched TMA
experiment result.
