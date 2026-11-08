# 汇总实验结果索引

最新综合入口：[模型与相近点击量行为的整合结论（2026-11-04）](../research/TMA_SYNTHESIS_20261104.md)。

这里保存可审阅的汇总表、协议和图。主问题是首次计分 TMA 截止前后各 14 天、按截止后点击量配对的两周设计，定义见[仓库 README](../README.md)、[两周冻结协议](../research/TMA_TWO_WEEK_PROTOCOL_20261103.md)和[方法文档](../METHODS.md)；+21/+28 天是窗口敏感性，第 0–27 天是历史基线。学生级 `parquet` 与配对/预测明细放在 `../outputs/`，由 Git 忽略。每轮结果应先读其 README，再比较同一人群、同一观察终点下的模型差值。全部模型结果基于已反复探索的 OULAD 数据。

## 主问题：首次计分 TMA 前后各 14 天（按截止后点击配对）

| 实验 | 结果目录 | 阅读重点 |
|---|---|---|
| 截止后同量点击配对、时间行为、结果盲聚类与模型消融 | [tma_two_week_20261103](tma_two_week_20261103/README.md) | 主配对只平衡截止后点击；前段差异显著，时间行为关联较一致，预测增量很小 |
| **点击量分簇后簇内比较 P/F** | [tma_click_clusters_20261103](tma_click_clusters_20261103/README.md) | 先按后段点击量分零/低/高簇，再在同批次、同量点击配对中比较行为；另核查前段参与 |
| 前后点击相对差距模型消融 | [tma_relative_gap_20261103](tma_relative_gap_20261103/README.md) | 在原 M0/M1 上加入对称相对差距；同测试集 AP 增量接近零 |
| 首次 TMA 提交距截止日 | [tma_submission_timing_20261103](tma_submission_timing_20261103/README.md) | TMA 类型核对、提前/逾期天数的预测增量、同成绩与后段点击配对；个人延期未知 |
| TMA 截止前 1／3 天点击爆发 | [tma_predeadline_burst_20261103](tma_predeadline_burst_20261103/README.md) | 严格截止前短窗口、逾期交互、同成绩及点击配对与簇内比较；描述性集中明显，模型增量很小 |
| 成绩×参与变化非线性属性 | [tma_response_attributes_20261104](tma_response_attributes_20261104/README.md) | 高分下降、低分增长、活跃日和第二周组合；平均 AP 小幅提高，跨年检查变差 |
| 提交时间×后续持续性组合 | [tma_submission_persistence_20261104](tma_submission_persistence_20261104/README.md) | 簇内后段点击、前段、成绩及提交状态逐层匹配；未提交×长空档有模型增量，低簇按时且持续活跃关联较清楚 |

## 窗口敏感性：截止后 +21/+28 天（按前后合计点击配对）

| 实验 | 结果目录 | 阅读重点 |
|---|---|---|
| TMA +21/+28 天配对 | [tma_matched_21_28_20261103](tma_matched_21_28_20261103/README.md) | 截止前后活动、提交状态与资源，附截止后点击量敏感性；主配对按前后合计点击匹配 |
| 节点模型/窗口对照 | [assessment_node_windows_20261103](assessment_node_windows_20261103/README.md) | +14/+21/+28 天统一特征模型；此处只取 +21/+28 两个截点作窗口敏感性 |

## 原 proposal：开课第 0–27 天历史基线

| 实验 | 结果目录 | 阅读重点 |
|---|---|---|
| 原第 0–27 天初跑 | [local_uci_20261102](local_uci_20261102/README.md) | 最早探索基线 |
| 数据修订与稳健性 | [revised_20261102](revised_20261102/README.md) | 重复事件、注册资格、点击量模型基线与聚类诊断；聚类未控制点击量 |
| 时间覆盖扩展 | [timing_20261102](timing_20261102/README.md) | 新时间汇总无清楚的额外 AP 增益 |
| 原题核心配对 | [matched_20261102](matched_20261102/README.md) | 同批次、相近 0–27 天点击量下的行为差异 |


## 仅索引，不进研究叙事

| 内容 | 结果目录 | 原因 |
|---|---|---|
| 首次 TMA 分数联合匹配 | [tma_score_probe_20261103](tma_score_probe_20261103/README.md) | 使用首次 TMA 成绩；截止时不可见，最多作回顾性上界 |
| 分数/排名模型消融 | [tma_score_model_20261103](tma_score_model_20261103/README.md) | 同上；成绩公布日未知 |
| 成绩×点击变化 | [tma_score_gap_20261103](tma_score_gap_20261103/README.md) | 同上 |
| HGB/校准方法探针 | [wu_method_probe_20261103](wu_method_probe_20261103/README.md)、[+28 天](wu_method_probe_28d_20261103/README.md) | 改编另一篇论文的模型设置，不是其退课任务的数值复现 |

## 其他相关实验

| 实验 | 结果目录 | 阅读重点 |
|---|---|---|
| 首次计分 TMA +14 天初版 | [assessment_node_20261103](assessment_node_20261103/README.md) | 两周特征后有微调；同一窗口比较以统一特征版为准 |

`research/` 另有 E1–E3 实验与“截止时未提交”子群探索。后者是独立候选研究问题，尚不能用本目录的全体 TMA 二分类结果替代。主问题与窗口敏感性使用不同配对变量（截止后点击 vs 前后合计点击），第 0–27 天使用不同时间锚点；三类结果不直接比较模型 AP 或配对差值。
