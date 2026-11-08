# 深度指标的冻结检验协议

冻结时间：2026-11-08。本协议在 `verify_frozen_increment.py` 首次运行**之前**写成，提交时间即冻结时间。

上游：`research/TMA_PRIMARY_PROTOCOL.md`（主问题与六项主要指标）、`research/VERIFICATION_PLAN.md`（本轮验证计划）。

## 1. 为什么需要这一份

`results/tma_deep_exploration_20261108/` 里 E1–E10 与 A1–B1 全部是探索性分析：同一份 OULAD、同一批配对、同一组测试记录被反复使用。其中 E10 报告"深度指标 ΔAP +0.0009 [−0.0011, +0.0032]"，但那是看到结果之后才选出来的指标组合。本协议把这一组合固定下来，再跑一次，结果无论正负都进报告。

## 2. 冻结内容

| 项 | 冻结值 |
|---|---|
| 队列 | TMA 两周二分类队列，19,806 条记录，Fail 6,092 |
| 配对 | 沿用主设计：同批次，后段点击 5% 容差，1:1，零只配零；4,768 对 |
| 切分 | 主切分：学生分组 60/20/20，种子 42（`data/processed/splits.json`） |
| 基线模型 | M2 = 前段点击/活跃日/近截止 3 天份额 + 后段点击（对数与平方）+ 提交状态 + 后段时间特征 + 提交距离，共 16 个特征 |
| 候选增量特征 | 8 项：`n_silences`、`n_internal_silences`、`longest_silence`、`n_returns_after_silence`、`mean_silence`、`ended_silent`、`anchor_days_late`、`anchor_submitted` |
| 分类器 | 逻辑回归（中位数填充 + 标准化 + 独热 `module_presentation`），不做超参搜索 |
| 阈值 | 验证集最大 F1，若验证集能达到 0.80 精确率则取满足条件的最大召回 |
| 主要指标 | Fail 平均精确率（AP） |
| 次要指标 | ROC-AUC、Brier、Top-5% 精确率（并列按并列块命中率折算） |
| 比较方式 | 学生聚类 bootstrap，400 次，同一批重抽样索引同时用于两个模型（配对 bootstrap） |
| 判定 | **区间不含 0 且 ΔAP ≥ 0.005** 记为"有增量"；否则记为"无增量" |
| 运行次数 | 一次。如因技术原因失败可重跑，但需在结果里记录两次的哈希 |

## 3. 预注册的预期

写协议时的预期：**无增量**。理由是 E10 的点估计 +0.0009 远小于 0.005，且深度指标与已有的后段时间特征高度共线。

这条预期写在这里是为了防止事后改写判据。如果结果与预期相反，报告里必须同时给出"探索阶段出现过一次跨零结果"这一段。

## 4. 结果存放

- `results/tma_deep_exploration_20261108/v1_frozen_increment.csv`：两行模型指标 + 一行差值。
- `results/tma_deep_exploration_20261108/v1_frozen_verdict.json`：`has_increment`、判定依据、协议哈希、脚本哈希、输入哈希。

## 5. 不做什么

- 不在这份协议的冻结范围内增加或删除特征。
- 不因为结果不理想而改切分、改阈值规则或改 bootstrap 次数。
- 不与 80/20 复现表（`t08_parity.csv`）比较——那是另一套切分，用于核对移植是否一致，不是增量检验。


## 6. 修订记录

**2026-11-08 晚，首次运行之后**：E2 的"窗口内多截止日"前提被证伪——19 个批次里相邻两个正权重 TMA 的最小间隔是 21 天，大于 14 天的后窗，所以 `deadlines_in_window` 恒为 1，`missed_deadlines` 与 `missed_all` 在数值上完全相同。原先的 9 项里有 3 项因此退化，改用锚点提交的两个量（`anchor_days_late`、`anchor_submitted`）替代，总数变为 8。

这是冻结之后的一次修订，必须在报告里如实写明：它不是"看到了结果再改特征"，而是"发现特征在数据上不可能有变化"。修订前后两次的结论一致（都判无增量），区间都跨零。

另有一个去重补充臂：`longest_silence` 与主分析的 `post_longest_inactive_run` 完全相同，`anchor_submitted` 与 `submitted_by_cutoff` 相同。剔除这两列后的 4 特征臂一并报告，避免把"重新计一遍同样的量"当成增量。
