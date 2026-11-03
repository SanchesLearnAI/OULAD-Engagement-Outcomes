# 研究材料索引

以 [项目总入口](../README.md)、[当前 +21/+28 天主问题](ACTIVE_TMA_21_28_RESEARCH.md)和 [进展与发现](RESEARCH_STATUS_20261103.md) 为起点。本目录的脚本从仓库根目录定位数据和输出；请在仓库根目录运行文中命令。

## 两个问题分别从哪里读

| 问题 | 观察窗口与研究重点 | 设计和结果入口 |
|---|---|---|
| **原 proposal：历史四周基线** | 开课第 0–27 天；同课程、同开课批次、总点击相近时，比较最终通过与失败者的四周行为 | [原实施计划](PROJECT_PLAN.md)；[四周配对结果](../results/matched_20261102/README.md)；[模型与聚类诊断](../results/revised_20261102/README.md) |
| **当前主问题：首次计分 TMA 截止后行为** | 以每批次首次计分 TMA 截止日为锚点，分别观察至 +21、+28 天；关注截止后的提交、活跃时间及资源使用 | [主问题完整记录](ACTIVE_TMA_21_28_RESEARCH.md)；[+21/+28 天配对结果](../results/tma_matched_21_28_20261103/README.md)；[节点模型](../results/assessment_node_windows_20261103/README.md) |
| **两周改写候选：首次计分 TMA 前后各 14 天** | 前段用于起点核查，主配对按截止后 14 天点击量；比较后段时间行为与结果 | [事前协议](TMA_TWO_WEEK_PROTOCOL_20261103.md)；[完整结果](../results/tma_two_week_20261103/README.md)；[点击量分簇补充](../results/tma_click_clusters_20261103/README.md) |

各设计的人群范围、时间锚点和特征窗口可能不同，模型指标与配对差值不能直接混排。此前 +21/+28 天主配对平衡的是截止前后**合计**点击；两周改写按截止后点击匹配。其[点击量分簇补充](../results/tma_click_clusters_20261103/README.md)已在零/低/高点击组内比较行为，但前段参与仍不平衡。原四周的聚类是另一项补充探索。各轮汇总文件见[结果索引](../results/README.md)。

## 研究设计与背景

| 文件 | 用途与状态 |
|---|---|
| [ACTIVE_TMA_21_28_RESEARCH.md](ACTIVE_TMA_21_28_RESEARCH.md) | **当前主问题**的样本、配对、模型、敏感性和限制；其他结果的阅读起点 |
| [TMA_TWO_WEEK_PROTOCOL_20261103.md](TMA_TWO_WEEK_PROTOCOL_20261103.md) | 两周改写候选的冻结设计、原 proposal 能力对应、运行和判读约束 |
| [CODE_DATA_EXPERIMENT_MAP.md](CODE_DATA_EXPERIMENT_MAP.md) | 六张原始表、代码模块、学生级/汇总产物和各轮实验清单 |
| [RESEARCH_STATUS_20261103.md](RESEARCH_STATUS_20261103.md) | 原 proposal、当前主线与独立候选题的进展关系 |
| [PROJECT_BRIEF_AND_LITERATURE.md](PROJECT_BRIEF_AND_LITERATURE.md) | 11 月 2 日文献及早期项目快照；其代码/数据缺失描述已过时 |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 原第 0–27 天问题的实施计划；保留为历史协议 |
| [WU2026_COMPARISON_AND_NEXT_STEP.md](WU2026_COMPARISON_AND_NEXT_STEP.md) | 与一篇前四周预警论文比较，讨论原问题边界；与仓库内的另一项 Wu 方法探针不是同一论文/任务 |
| [EXPERIMENT_PLAN_20261103.md](EXPERIMENT_PLAN_20261103.md) | E0–E3 顺序实验的事前计划；部分“尚未运行”措辞应结合下一行结果阅读 |
| [EXPERIMENT_RESULTS_20261103.md](EXPERIMENT_RESULTS_20261103.md) | E1–E3 已运行的模型消融与限制 |
| [RESEARCH_REFRAMING_20261103.md](RESEARCH_REFRAMING_20261103.md) | 考虑改写研究问题的背景、候选题与可行性；文内推荐是当时建议，尚未决定替代当前 +21/+28 天主线 |

## 脚本与对应输出

| 脚本 | 汇总输出 | 主要作用 |
|---|---|---|
| `audit_current.py`、`audit_boundaries.py` | `audit_20261102/` | 早期数据/边界检查 |
| `assessment_node_probe.py` | `assessment_node_probe/` | 混合首次考核类型的早期探针；不要直接当成统一准时率 |
| `experiment_design_audit.py` | `experiment_design_20261103/` | 数据规模、课程及资源目录核查 |
| `run_experiments_20261103.py` | `experiment_design_20261103/` | E1/E2/E3 汇总模型消融；重跑会覆盖该目录的汇总文件 |
| `reframe_feasibility.py` | `reframe_20261103/` | 新候选题的人群、TMA 日期与结局覆盖核查 |
| `explore_reengagement.py` | `reframe_20261103/` | 截止日尚未提交子群的初步活动分层 |

`reframe_20261103/EXPLORATORY_FINDINGS.md` 是该子群探索的文字解释。`experiment_design_20261103/` 的模型表应结合 `EXPERIMENT_RESULTS_20261103.md` 读取。这里的 CSV/JSON 是汇总证据；学生级特征留在仓库的 Git 忽略目录中。

`audit_current.py`、`assessment_node_probe.py`、`experiment_design_audit.py` 和 `run_experiments_20261103.py` 还读取历史运行产生的 `outputs/` 特征表；新克隆仓库需先按对应实验命令生成这些学生级中间文件。`reframe_feasibility.py` 则直接从已提交的 OULAD ZIP 读取聚合所需的原始表。
