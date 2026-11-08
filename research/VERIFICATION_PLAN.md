# 深度探索的验证计划（B3 / C1 / D1 / A3 / B2）

- 版本：v1，2026-11-08
- 分支：`verify/exploration-checkpoints`
- 上游：`results/tma_deep_exploration_20261108/ROADMAP.md`（E1–E10、A1/A2/B1 已完成）

## 0. 为什么要"验证"而不是"再探索"

上一轮的十项分析全部是探索性的：同一份 OULAD、同一批配对、同一组测试记录被反复使用，p 值和区间都偏乐观。本轮要做的是**把其中可以冻结的部分冻结起来，再检验一次**，并明确哪些结论即使冻结也仍然是描述性的。

具体分三类：

1. **冻结后重跑**（B3）：先写协议、锁定指标集与切分，再跑一次，结果记为"冻结后的检验"。
2. **汇总已有结果**（C1、D1、A3）：不产生新数字，只把散落的表整理成可审计的形式，并补上缺失的量（如最小可检测效应）。
3. **同口径对照**（B2）：用与主模型完全相同的切分、标签、测试记录，跑一个序列模型，回答"文献里的高准确率是不是口径差异"。

## 1. 冻结协议（B3，先于任何新运行）

新建 `research/TMA_DEEP_PROTOCOL.md`，写入：

| 项 | 冻结值 |
|---|---|
| 队列与配对 | 沿用 `research/TMA_PRIMARY_PROTOCOL.md`：19,806 条、4,768 对、5% 容差 |
| 基线模型 | M2（16 特征），主切分 60/20/20，种子 42 |
| 候选增量特征 | E2/E8/E9 的 9 项：沉默段数、内部沉默段、最长沉默、回归次数、平均沉默、以沉默结尾、漏交次数、累计逾期天数、窗口内截止日数 |
| 主要指标 | Fail AP，附 no-skill 基线 |
| 比较方式 | 学生聚类 bootstrap，400 次，双侧 95% 区间 |
| 判定 | 区间不含 0 且 ΔAP ≥ 0.005 记为"有增量"；否则记为"无增量" |
| 次要指标 | ROC-AUC、Brier、Top-5% 精确率 |
| 检验对象 | **只检验一次**；结果无论正负都写进报告附录 |

写完之后单独提交，提交时间即冻结时间。

## 2. 执行顺序

### V1 B3 冻结增量检验

```bash
cd oulad-sem
python scripts/verify_frozen_increment.py --repeats 400
```

产出 `results/tma_deep_exploration_20261108/v1_frozen_increment.csv`，以及 `v1_frozen_verdict.json`（含 `has_increment: true/false` 与判定依据）。

验收：
- 与 E10 的点估计一致（E10 是 +0.0009）；重复次数不同会带来小数点后 3 位的波动。
- 判定必须与本轮协议一致，不能在看到结果后改阈值。

### V2 C1 外推汇总

```bash
python scripts/verify_transfer_summary.py
```

把 `t08_model_metrics.csv` 里的 `temporal` 与 6 个 `lomo_*` 折汇总成一张表，附"已见课程 vs 新课程"对照，并标明 CCC 在 2013 没有开课。

验收：行数 = 1（temporal）+ 6（LOMO）= 7；每行有 AP、ROC-AUC、Brier、n_test、Fail 占比。

### V3 D1 子群审计与最小可检测效应

```bash
python scripts/verify_subgroup_mde.py --repeats 400
```

对性别、IMD 三分位、学历、残障四个属性，给出：
- 各子群的 AUC 与样本量；
- 行为模型 vs 全特征模型的 AUC 差；
- **最小可检测效应（MDE）**：在给定样本量与方差下，双侧 α=0.05、power=0.8 时能检出的最小 AUC 差；
- BH 校正后的 p 值。

验收：如果所有子群差都小于 MDE，就明确写"本数据不足以判断子群差异是否存在"，而不是"没有差异"。

### V4 A3 延迟亚型对照

```bash
python scripts/verify_delay_profiles.py
```

把 E8 的三剖面 × 结局做交叉表，并给出：
- 每个剖面内的点击量分布（判断剖面是否只是窗口长度的代理）；
- 与文献三分型（高成就延迟者 / 低成就延迟者 / 偶发延迟者）的对应关系；
- 若无法对应，明确写"本数据不支持该分型"。

验收：剖面数与 E8 一致（k=3）；剖面内点击量中位数的差异要报出来。

### V5 B2 序列模型同口径对照

```bash
python scripts/verify_sequence_models.py --epochs 8
```

前提：安装 PyTorch（`pip install torch`，约 100 MB）。若安装失败，改用 scikit-learn 的 `MLPClassifier` 作为替代，并在结果里注明"不是真正的序列模型"。

做法：
- 输入：14 天 × 日点击（对数化），后段窗口；
- 模型：GRU（或 MLP 替代）与小型 Transformer encoder；
- **完全相同的**主切分、Fail 标签、测试记录；
- 对比 M2 的 AP、ROC-AUC、Brier。

验收：三个模型在同一测试集上比较；如果序列模型不优于 M2，结论写"在本口径下无增益"，并解释与文献 90%+ 准确率的差异来源（标签含 Withdrawn、逐天累计口径、随机切分等）。

## 3. 版本管理

- 分支 `verify/exploration-checkpoints`，每个子任务一个提交：`verify(V1): ...`。
- 协议文件在 V1 **之前**单独提交，作为冻结证据。
- 所有脚本写出 `provenance.json`（输入哈希、脚本哈希、配置哈希），结果目录附 `VERIFICATION.md`。
- 不修改 `oulad-sem/` 的冻结主线产物；验证脚本只读中间文件。
- 全部通过后，把 `verify/exploration-checkpoints` 合回 `feat/oulad-sem-tma`（`--no-ff`），并在 `release/tma-merged/SEAL.md` 追加一节验证记录。

## 4. 停止点

- 冻结协议写完后发现需要改指标集 → 停下询问，不能自己改。
- 序列模型在 8 个 epoch 内不收敛或显存/内存不足 → 退回 MLP 替代并注明。
- V1 的判定与协议阈值冲突（例如区间含 0 但点估计 ≥ 0.005）→ 以协议为准，如实记录。
- 任何一步需要新增数据或外部数据 → 停下询问。
