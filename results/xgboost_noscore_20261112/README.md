# 不含成绩的 XGBoost 对照（探索性）

本实验补上原 `xgboost_score_20261111` 缺的那一半：**同一个树模型，只使用截止 `d+14` 时可观测的特征**。这样才与冻结的 M2 逻辑回归构成可比对照，也不涉及成绩公布日未知的问题。

## 结果

五次学生分组 80/20 测试（与 `xgboost_score_20261111` 同一切分、同一批 19,806 条记录）：

| 随机种子 | 测试记录 | M2 逻辑回归 AP | XGBoost AP | ΔAP | 逻辑回归 ROC-AUC | XGBoost ROC-AUC | 逻辑回归 Brier | XGBoost Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3,942 | 0.671537 | 0.680393 | **+0.008856** | 0.7790 | 0.7857 | 0.16227 | 0.15957 |
| 143 | 3,967 | 0.679165 | 0.690917 | +0.011752 | 0.7846 | 0.7947 | 0.16252 | 0.15930 |
| 144 | 3,958 | 0.657217 | 0.661492 | +0.004275 | 0.7729 | 0.7781 | 0.16338 | 0.16174 |
| 145 | 3,948 | 0.665703 | 0.673261 | +0.007558 | 0.7724 | 0.7833 | 0.16158 | 0.15856 |
| 146 | 3,951 | 0.683455 | 0.687734 | +0.004279 | 0.7888 | 0.7888 | 0.15806 | 0.15725 |
| **平均** | | **0.671415** | **0.678759** | **+0.007344** | **0.7796** | **0.7861** | **0.16156** | **0.15929** |

五个种子**全部为正**。种子 42 的学生分组 bootstrap（400 次）给出 ΔAP 95% 区间 **[+0.00135, +0.01541]**，见 [interval_seed42.csv](interval_seed42.csv)。五组测试集彼此重叠，`5/5` 不能当作五次独立检验。

## 与含成绩版本的关系

| 版本 | 特征数 | 平均基线 AP | 平均 XGBoost AP | ΔAP | 可部署 |
|---|---:|---:|---:|---:|---|
| 含成绩（`xgboost_score_20261111`） | 19 | 0.713642 | 0.719919 | +0.006277 | 否 |
| **不含成绩（本实验）** | **16** | **0.671415** | **0.678759** | **+0.007344** | **是** |

两个版本的树模型增益几乎一样（+0.0063 vs +0.0073），说明**增益来自模型本身，而不是成绩特征**。含成绩版本更高的绝对 AP（0.7136 → 0.7199）主要来自成绩本身，与本实验的增量不是同一个量。

## 特征

数值特征 16 个，与冻结 M2 完全一致：

```
log_pre_clicks, pre_active_days, last3_share,
log_post_clicks, log_post_clicks_squared,
submitted_by_cutoff, submitted_after_due,
post_active_days, post_both_weeks, post_longest_inactive_run,
post_week2_share, post_peak7_share, post_entropy_norm, post_first_active_day,
days_early, days_late
```

加 19 个课程呈现 one-hot，共 35 个设计矩阵列。**没有 `first_tma_score`、`score_recorded_for_submission`，也没有 `unsubmitted_x_inactive_run` 这个交互项**——后者是看到结果之后构造的，不属于可观测特征集。

## gain 重要性（五种子平均，前八）

| 特征 | 平均 gain | 最小 | 最大 |
|---|---:|---:|---:|
| `post_both_weeks` | 0.2577 | 0.1336 | 0.4517 |
| `submitted_by_cutoff` | 0.1733 | 0.0945 | 0.2396 |
| `post_active_days` | 0.1179 | 0.0928 | 0.1541 |
| `days_early` | 0.0432 | 0.0369 | 0.0551 |
| `submitted_after_due` | 0.0305 | 0.0224 | 0.0428 |
| `module_presentation=BBB_2014J` | 0.0264 | 0.0185 | 0.0325 |
| `post_week2_share` | 0.0264 | 0.0173 | 0.0431 |
| `post_entropy_norm` | 0.0248 | 0.0159 | 0.0336 |

前三位是“两周都活跃”“截止前已提交”“活跃天数”，全部属于提交行为与持续性，与配对分析的主指标方向一致。**但重要性不表示关联方向，也不构成因果证据。**

## 复现

```bash
# 1. 本地（仓库根目录）：生成 NPZ，需要主流程已产出 features.parquet
python research/xgboost_noscore_20261112/prepare_input.py --output outputs/xgb_noscore.npz

# 2. GPU 主机：训练与导出
PYTHONPATH=./.pylibs CUDA_VISIBLE_DEVICES=0 \
  python3 run_xgboost.py --input xgb_noscore.npz --output-dir xgb_noscore_run
```

`prepare_input.py` 会用冻结 M2 的特征表在同一批切分上重算基线 AP，并把结果写进 NPZ，因此第 2 步不依赖本地环境。脚本也支持 `--device cpu`，在没有 GPU 的机器上可以跑通，但参考结果来自 CUDA。

本次运行环境：Python 3.8.10、XGBoost 2.1.4、scikit-learn 1.3.2、NumPy 1.24.4、RTX 4090 24 GB、驱动 575.57.08；见 [provenance.json](provenance.json)。**这不是 `xgboost_score_20261111` 的那台机器**（那台是 Windows + RTX 3050 + XGBoost 3.4.1），因此两个实验的绝对数值不能逐位对齐，只能比较增量大小。

## 解释边界

1. **回顾性**：基线 M2 本身是多轮探索后选定的，这批测试记录也已被反复使用。+0.0073 是探索性增益。
2. **与含成绩版本不可直接相减**：0.7136 与 0.6714 的区别主要是成绩，不是模型。
3. **不能据此改动冻结主模型**：本实验是旁路对照，`oulad-sem/` 的 `primary_model: M2` 未变，V1 冻结检验的基线也未变。
4. **五组切分重叠**：`5/5` 为正说明方向一致，不是五次独立显著。
5. **bootstrap 只做了种子 42**：受 400 次重复限制，p 值下限为 1/400 = 0.0025，区间下界 +0.00135 已接近该分辨率。
