# 证据摘要（2026-11-04）

> 来源：两轮 deep-research 断点复跑。完整结果见同目录两个 `.result.json`。
> 用途：给写 report / poster 的人做引用分级。**不要把它当成同行评审综述**；每条都要能回到原文。

## 1. 核验统计

| 轮次 | 源 | 候选 claim | 确认 | 驳回 | 未核验 |
|---|---:|---:|---:|---:|---:|
| OULAD 基线 / 数据陷阱 / 分阶段 | 18 | 89 | 9 | 8 | 8 |
| Engagement / 匹配 / 聚类 / 指标 | 25 | 121 | 23 | 2 | 0 |
| 合计 | 43 | 210 | **32** | **10** | **8** |

未核验的 8 条集中在 Withdrawn 细节、低活跃/零点击过滤、负日期的官方表述、部分 LSTM 指标；不能当事实用。

## 2. 可引用，但必须带限定

| 结论 | 数字 | 限定 |
|---|---|---|
| OULAD day-28 预测 | HistGB 22 特征 ROC-AUC **0.7747**（95% CI 0.7624–0.7870）；平衡准确率 0.6974；at-risk 召回 0.6116；13 行为特征 **0.7504**；去分数 **0.7381**；纯行为 **0.7027** | 目标 **Fail-or-Withdrawn**；行为特征含早期提交数与早期均分；不是纯 clickstream、Pass/Distinction vs Fail。Wu et al. 2026, Front. Educ. 11:1917092 |
| LEAP 周级曲线 | week1 **0.7151** → week2 0.7404 → week3 0.7851 → week4 **0.8055** → week8 **0.8602**；2→3 周 +0.0447 | 预印本 arXiv:2605.25794；目标 Pass/Distinction vs Fail/Withdrawn；不是本项目设定 |
| 时间泄漏 | t=7 严格 **0.7151** vs 含未来信息 **0.9669**（GBDT 0.7147→0.9704），约 **0.25 AUC** | 预印本；用于 sanity check，不是基线 |
| 学生重叠 | 行级划分会让 **852 人 / 15.7%** 测试记录重叠；该研究 day-28 仍注册样本 **27,553** 条 | Wu et al. 2026 |
| 时间/模块外推 | 2013→2014 AUC **0.7470**（0.7393–0.7549）；LOMO **0.6426–0.7881**（median 0.6990） | all-features 模型（含评估表现），不是纯 clickstream |
| 日期语义 | `studentVle.date` 相对开课日，范围 **−25..269**；**688,988 行（6.47%）**为负 | 22 个 presentation 均有负值 |
| time-on-task 估计 | 15 种策略使同一模型 **R² 相差 0.15**（0.07–0.23），无金标准 | Kovanović et al. 2015, LAK |
| self-report vs trace | 反应风格解释问卷方差 **43%**，trace 最高 **4%**；process 型 trace 控制 mastery 后可转负 | Tempelaar et al. 2020, PLOS ONE |
| 课程情境混杂 | 课程特异模型 vs 合并模型在预测力与显著变量上不同（**n=4,134**） | Gašević et al. 2016 |
| 聚类稳定性报告 | 被评述的 EM 研究只报 BIC + MANOVA，无 bootstrap/ARI/silhouette | 这是“实务常缺”的证据，不是阈值定值 |

## 3. 不得引用（被驳回或未核验）

- `0.58–0.75` 的“全课程 0.9+ 崩塌”（0–3）。
- week-4 dropout 的 `0.626 / 0.663 / 0.636`（1–2）。
- 全课程 LSTM 周级 `69.69% → 95.23%`（0–2）。
- “assessment 相关点击是已证实泄漏通道”（0–3）。一般性时间泄漏成立，但这条具体通道不成立。
- time-on-task 系数反号区间（1–2）、“每 15 分钟只有 6 分钟在任务上”（0–3）。
- `5,040`、`35/53` 等未核验数字。
- PSM/CEM/PR-AUC/校准的具体阈值：本轮没有已核验 claim 覆盖。

## 4. 覆盖缺口

- **匹配**：PSM/CEM 的平衡诊断（SMD、Love plot）、共同支撑、caliper、匹配后偏差校正、敏感性分析，本轮没有直接证据。继续按 Rosenbaum & Rubin、Austin、Stuart、Iacus/King/Porro 的方法学共识执行，但写成“方法共识”。
- **不平衡指标**：PR-AUC、recall/F1、校准的报告规范，本轮没有直接证据。按 Saito & Rehmsmeier、Van Calster 的方法学共识执行。
- **Withdrawn**：编码与 `date_unregistration` 的细节未核验，且“退课后点击截断”捆绑断言被 0–3 驳回。执行前必须用原始 CSV 审计。

## 5. 引用规则

1. 先写目标、特征、预测点，再写数字。例：“在 Fail-or-Withdrawn、含早期分数的 22 特征设定下，day-28 AUC 为 0.7747（Wu et al. 2026）。”
2. 不把预印本写成“已发表”；LEAP 标 `arXiv preprint`。
3. 不把“方法共识”写成“本轮核验”。
4. 被驳回的数字不进正文，也不进 poster。
5. 任何引用在写进 report 前回原文核对表格/页码；`Course_Project_Guide.pdf` 明确禁止伪造引用。

## 6. 来源文件

- `deep-research-oulad.result.json`：OULAD 基线、数据陷阱、分阶段预测。
- `deep-research-methodology.result.json`：engagement 效度、SRL/轨迹、课程情境、聚类稳定性。
- 方案中的完整证据表：`01_plan/oulad-deep-mining-plan.html` 第 01 节。
- 执行引用分级：`01_plan/EXECUTION_SIMULATION.md` 第 9.4 节。
