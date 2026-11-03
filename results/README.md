# 汇总实验结果索引

这里保存可审阅的汇总表、协议和图。当前 +21/+28 天主问题的定义见[仓库 README](../README.md)和[方法文档](../METHODS.md)。学生级 `parquet` 与配对/预测明细放在 `../outputs/`，由 Git 忽略。每轮结果应先读其 README，再比较同一人群、同一观察终点下的模型差值。全部模型结果基于已反复探索的 OULAD 数据。

| 研究阶段 | 结果目录 | 阅读重点 |
|---|---|---|
| **当前主问题：TMA +21/+28 天配对** | [tma_matched_21_28_20261103](tma_matched_21_28_20261103/README.md) | 截止前后活动、提交状态与资源，附截止后点击量敏感性 |
| **当前主问题：节点模型/窗口对照** | [assessment_node_windows_20261103](assessment_node_windows_20261103/README.md) | +14/+21/+28 天统一特征模型；+21/+28 是当前主问题的两截点 |
| 首次 TMA 分数联合匹配 | [tma_score_probe_20261103](tma_score_probe_20261103/README.md) | 相似点击量和成绩下的配对；分数发布时间未知 |
| 分数/排名模型消融 | [tma_score_model_20261103](tma_score_model_20261103/README.md) | 回顾性模型；不能当成实时预警性能 |
| HGB/校准方法探针 | [wu_method_probe_20261103](wu_method_probe_20261103/README.md)、[+28 天](wu_method_probe_28d_20261103/README.md) | 改编论文模型设置；不是其退课任务的数值复现 |
| 原第 0–27 天初跑 | [local_uci_20261102](local_uci_20261102/README.md) | 最早探索基线 |
| 数据修订与稳健性 | [revised_20261102](revised_20261102/README.md) | 重复事件、注册资格、模型与聚类诊断 |
| 时间覆盖扩展 | [timing_20261102](timing_20261102/README.md) | 新时间汇总无清楚的额外 AP 增益 |
| 原题核心配对 | [matched_20261102](matched_20261102/README.md) | 同批次、相近 0–27 天点击量下的行为差异 |
| 首次计分 TMA +14 天初版 | [assessment_node_20261103](assessment_node_20261103/README.md) | 两周特征后有微调；同一窗口比较以统一特征版为准 |

本地工作区的 `research/` 另有 E1–E3 实验与“截止时未提交”子群探索。后者是独立候选研究问题，尚不能用本目录的全体 TMA 二分类结果替代。
