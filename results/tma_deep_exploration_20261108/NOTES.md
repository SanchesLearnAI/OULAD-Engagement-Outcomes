# 深度探索说明（给后续会话）

本文件说明 `results/tma_deep_exploration_20261108/` 是怎么来的、每条结论有多强、以及下一步该动哪个文件。

## 起点

- 冻结主线：`oulad-sem/`，run `run-20261108T101537Z-28e708`，配置 SHA-256 `6772d002...`。
- 交接文件：`~/Desktop/HANDOFF_20261108.md` 第 9 节列了四个优先方向（固定题目、冻结主要结果、做外部验证、交付前核对）。前三项已在前两段完成或部分完成，本轮的探索是对"冻结之后还能挖什么"的回答。
- 参考论文：`/Users/zhangyuming/Desktop/Project/论文2.pdf`（Wu, Jia & Pham 2026, Frontiers in Education）。它的三条贡献——泄漏控制协议、数据最小化代价、跨年与跨课程外推——直接对应本轮的 E5（数据最小化 + 子群）、E7（配对 vs 未匹配）、C1（外推汇总）。

## 脚本与产出的对应

| 脚本 | 产出 | 核心问题 |
|---|---|---|
| `oulad-sem/scripts/explore_tma_deep.py` | E1–E5 | 截止后是否回来、窗口内多截止日、资源构成、日模式、数据最小化 |
| `oulad-sem/scripts/explore_tma_patterns.py` | E6–E7 | 带间隙的模式 + 置换检验、配对内 sign test、容量曲线、分层校准 |
| `oulad-sem/scripts/explore_tma_delay.py` | E8–E9 | SFI 延迟剖面、沉默—回归结构 |
| `oulad-sem/scripts/explore_tma_incremental.py` | E10 | 深度指标的增量价值 |
| `oulad-sem/scripts/explore_tma_followups.py` | A1、A2、B1 | 沉默分层、类型序列、名额策略 |

全部脚本都从 `oulad-sem/` 目录运行，读取 S02–S06 产出的中间文件；不修改冻结主线的任何产物。

## 结论强度分级

| 结论 | 强度 | 依据 |
|---|---|---|
| Fail 的沉默"更久"而非"更多" | 强（描述性） | 四层点击量内方向一致，17 个批次无反向；E9 与 A1 互相印证 |
| 内容类活动的规律序列更像 Pass | 中（描述性） | 剔除零点击后只剩 5/35 显著，方向一致但支撑度低 |
| 配对把点击量差异压平 | 强（设计事实） | 未匹配 −76.96 → 配对 −0.05 |
| 深度指标无预测增益 | 中（探索性） | ΔAP +0.0009 [−0.0011, +0.0032]，需 B3 正式检验 |
| 背景特征有增量但远小于行为 | 中（与文献一致） | ΔAP +0.028 [+0.018, +0.039]；与 Wu et al. 2026 的 0.024 接近 |
| 子群差异不显著 | 弱（样本不足） | BH 后 p=1.0，需要 D1 的 MDE |
| SFI 剖面能区分延迟类型 | 弱（近乎无效） | 三剖面主要由窗口长度分裂，配对内一致率 0.516 vs 随机 0.479 |

## 与文献的关系

- **Wu et al. 2026（论文2）**：本项目的 E5 复刻了它的 3×3 消融思路（背景/行为/全部），但用的是 TMA 两周队列与 Fail 标签，因此绝对数值不可比；可比的是**相对结论**——行为信息远强于背景信息，去掉背景不自动改善子群差距。
- **延迟文献（CHI 2025 "Not ALL Delay is Procrastination"、LAK 类研究）**：他们主张按延迟亚型分类；本数据只分出三个以窗口长度为主的剖面，说明 OULAD 的截止驱动结构太强，不足以复现亚型划分。这条应作为"负结果"写。
- **序列模型文献（Transformer encoder 2023、GRU 2025）**：报 90% 以上准确率，但标签（含 Withdrawn）与切分口径都不同。若要回应，必须用 B2 的同口径对照，而不是比较绝对数字。

## 下一步的具体入口

| 要做的事 | 改哪个文件 | 预估时间 |
|---|---|---|
| B3 冻结增量检验 | 新建 `research/TMA_DEEP_PROTOCOL.md`，复用 `explore_tma_incremental.py` | 1 小时（含写协议） |
| C1 外推汇总 | 读 `oulad-sem/outputs/tables/t08_model_metrics.csv`，写一个 `c1_transfer_summary.csv` | 30 分钟 |
| D1 子群 MDE | 扩 `explore_tma_deep.py` 的 `data_minimisation` 子群段 | 1 小时 |
| A3 延迟亚型对照 | 扩 `explore_tma_delay.py` 的剖面 × 结果交叉表 | 30 分钟 |
| B2 序列模型 | 新脚本，输入 `data/processed/features.parquet` 的日点击矩阵 | 半天 |

## 复现注意

1. 先跑 S02–S06（`make full` 或至少到 S06），否则 `data/processed/features.parquet` 与 `outputs/pairs_05.parquet` 不存在。
2. E6 的置换检验用 1000 次重复；降到 200 会得到同样的定性结论，但 p 值下限变成 0.005。
3. A2 依赖 `studentVle.csv` 的 `id_site` 列，`daily_events` 默认只保留汇总列，所以 `explore_tma_followups.py` 里另写了 `events_with_site`。
4. 所有脚本都写出 `summary*.json` 与 `provenance.json`，改代码后这些文件会指示输入哈希是否变化。
