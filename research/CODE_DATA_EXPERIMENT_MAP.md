# 文件、数据流和实验目录清单

更新：2026-11-03。本页回答“哪个文件负责什么、实验从哪份数据来、结果写在哪里”。项目方向与数值结论见 [当前 +21/+28 天主问题](ACTIVE_TMA_21_28_RESEARCH.md) 和 [总体进展](RESEARCH_STATUS_20261103.md)。本页位于 Git 仓库的 `research/` 目录，`../` 是仓库根目录。

## 1. 源数据与保存边界

| 原始表 | 当前本地解压位置 | 在项目中的用途 |
|---|---|---|
| `studentInfo.csv` | `../data/raw/` | 学生×课程×批次的背景和最终课程结果 |
| `studentRegistration.csv` | 同上 | 注册/退课日期与截点资格 |
| `studentVle.csv` | 同上，约 454 MB | 学生×资源×日期的点击事件；原始重复行在主版本保留 |
| `vle.csv` | 同上 | 资源 `id_site` 的活动类型，用课程＋批次＋站点连接 |
| `assessments.csv` | 同上 | TMA 类型、权重和统一截止日 |
| `studentAssessment.csv` | 同上 | 作业提交日期、承接旧成绩、最终记录的作业分数 |

源 [ZIP 和许可说明](../data/README.md) 在 `data/oulad.zip`。部分研究审计还直接读 ZIP 中的 `courses.csv`。解压的 `data/raw/` 与学生级 `outputs/` 被仓库 `.gitignore` 排除；`results/` 留存汇总 CSV/JSON 和图。原始 ZIP 哈希和每轮协议中的输入哈希用于溯源。既有多轮实验在未提交的工作树上运行；本次 Git 归档固定当前代码，却不能追溯成当时每轮运行的不可变代码快照。原 proposal PDF 留在本地工作区作为参考，不随研究代码公开。

## 2. 可运行代码按任务分组

| 文件/入口（均在 `../oulad_analysis/`） | 读取/产生 | 对应研究 |
|---|---|---|
| `__main__.py` | `python -m oulad_analysis`；编排建表、原四周配对/分层/聚类/建模，写运行 manifest | 第 0–27 天历史基线 |
| `features.py` | 原始点击、资源与学生信息 → `features.parquet`、`data_quality.json` | 原四周的数据准备 |
| `matched.py` | `features.parquet` → 配对平衡、行为差值、批次方向及种子敏感性 | 原四周的核心相近点击比较 |
| `compare.py`、`cluster.py`、`timing.py` | 点击分层、探索聚类、时间覆盖图表 | 原四周的补充分析，不代表当前 +21/+28 天主配对 |
| `model.py` | 未经匹配的学生分组测试、随机森林/逻辑回归、区间与稳健性 | 原四周预测检查 |
| `assessment_node.py` | 学生信息、注册、考核、学生提交、VLE 点击五张 CSV → `node_features.parquet`、四个 B 模型与协议；`--post-weeks 2/3/4` | 首次计分 TMA 节点；当前主问题的上游 |
| `assessment_node_compare.py` | 三个节点输出 → 共同人群的窗口比较 | 区分观察时间延长与队列变化 |
| `tma_matched.py` | +21/+28 天节点特征＋原始 VLE/资源表 → 配对、资源敏感性、B4 模型 | **当前主问题的直接实现** |
| `tma_two_week.py` | +14 天节点特征＋原始 VLE/资源表 → 后段点击配对、前段平衡、后段时间/资源、结果盲聚类与 M0–M2 模型 | **TMA 前后各两周改写候选**；[协议](TMA_TWO_WEEK_PROTOCOL_20261103.md) |
| `tma_click_clusters.py` | 两周学生级特征和 5% 后段点击配对 → 结果盲点击量簇、簇内 P/F 行为比较、前段平衡敏感性 | [两周分簇补充](../results/tma_click_clusters_20261103/README.md) |
| `tma_relative_gap_model.py` | 两周学生级特征 → 前后点击对称相对差距；复现原 M0/M1 并比较加入该特征后的同切分 AP/Brier | [相对差距消融](../results/tma_relative_gap_20261103/README.md) |
| `tma_score_gap.py` | 两周节点＋学生级特征＋首次 TMA 提交/成绩 → 成绩与点击差距交互模型、成绩分层、成绩及后段点击联合配对 | [+14 天成绩交互补充](../results/tma_score_gap_20261103/README.md)；成绩公布时间未知 |
| `tma_submission_timing.py` | 两周节点＋学生级特征＋首次 TMA 提交日 → 提前/逾期天数消融、提交时间分层、同成绩与后段点击联合配对及点击量簇内检查 | [+14 天提交距截止日补充](../results/tma_submission_timing_20261103/README.md)；个人延期日期未知 |
| `tma_predeadline_burst.py` | 两周节点＋原始逐日 VLE 点击 → 截止前 1／3 天速率变化、逾期交互、事前与 +14 天模型消融、配对及点击量簇内检查 | [截止前短时点击爆发](../results/tma_predeadline_burst_20261103/README.md)；VLE 点击无法归属特定 TMA |
| `tma_response_attributes.py` | 两周特征＋成绩／提交日期 → 非线性主效应、高分下降／低分增长、活跃日及第二周交互；同切分与跨年检查 | [成绩×参与组合](../results/tma_response_attributes_20261104/README.md)；[综合结论](TMA_SYNTHESIS_20261104.md) |
| `tma_score_probe.py` | 节点特征＋首个 TMA 的提交/分数＋配对结果 → 分数覆盖、联合配对 | 当前主问题的成绩敏感性 |
| `tma_score_model.py` | 上一行的学生级分数特征 → S0–S3 回顾性模型 | 检查分数、排名和时间特征 |
| `wu_method_probe.py` | 节点特征 → 逻辑回归/HGB/校准方法比较 | 方法探针；改编另一论文设置，不是退课研究复现 |

合成数据测试在仓库 `tests/`，自动工作流在 `.github/workflows/`。仓库 `jobs/oulad_cpu.slurm` 运行的是原四周 `--stage all` 流程；它仅准备 `studentInfo`、`studentVle`、`vle`，**不构成当前 TMA 实验的一键复算脚本**。Python 3.11 的环境说明见仓库 `requirements.txt` 和 `requirements-lock.txt`。

## 3. 当前 +21/+28 天主问题的数据流

```text
data/oulad.zip → data/raw/ 六张 CSV
   ↓ assessment_node.py --post-weeks 3 / 4
outputs/<两个独立节点运行目录>/node_features.parquet
   ├─ assessment_node_compare.py ← 另加 2 周节点结果
   ↓ tma_matched.py ← studentVle.csv + vle.csv
outputs/<配对运行目录>/features_21d.parquet、features_28d.parquet、配对明细
   ↓ tma_score_probe.py ← assessments.csv + studentAssessment.csv
outputs/<分数探针目录>/scored_features_21d.parquet、scored_features_28d.parquet
   └─ tma_score_model.py

节点特征 ──→ wu_method_probe.py（+21 与 +28 分别运行）
各运行目录的汇总 CSV/JSON → results/ 对应日期目录
```

两周改写候选另走：`assessment_node.py --post-weeks 2` → `outputs/<2 周节点目录>/node_features.parquet` → `tma_two_week.py`（再读 `studentVle.csv`、`vle.csv`）→ `outputs/<两周运行目录>/` 学生级特征、配对、簇标签与汇总 → [公开汇总](../results/tma_two_week_20261103/README.md)。它按**截止后**点击匹配，不能和 +21/+28 天按前后合计点击匹配的差值混为同一估计量。

`tma_click_clusters.py` 接续两周运行目录的 `features.parquet` 和 `post_pairs_05.parquet`；学生级点击量簇标签仍留在 `outputs/`，[公开汇总](../results/tma_click_clusters_20261103/README.md)仅含簇规模、配对行为差值及图。该补充为事后探索，不改动两周主分析的冻结协议。

`tma_relative_gap_model.py` 同样读取两周 `features.parquet`，以 `(后段−前段)/(后段+前段)` 比较原 M0/M1 与加相对差距后的模型，并核对原模型指标。[汇总结果](../results/tma_relative_gap_20261103/README.md)只含 AP/Brier、bootstrap 区间和图，不保存学生级预测。

`tma_score_gap.py` 在上述特征上接入首次 TMA 分数，只使用 `d+14` 前已提交且有最终记录分数者。它在同切分比较分数、相对差距及两者交互，并按同份 TMA 的成绩百分位做描述、按成绩与后段点击联合配对；[汇总结果](../results/tma_score_gap_20261103/README.md)为回顾性证据，行级配对留在 `outputs/`。

`tma_submission_timing.py` 接续同一两周特征，读取首次 TMA 的提交日并与计划截止日相减；只使用 `d+14` 前的提交。它复核 M1/S0 基线，再比较提前与逾期天数的增量，并在成绩和后段点击接近的配对中检查提交日差异。[汇总结果](../results/tma_submission_timing_20261103/README.md)含模型、分层和配对表；行级配对留在 `outputs/`。

`tma_predeadline_burst.py` 从原始 `studentVle.csv` 重建截止前逐日点击，严格以 `d−1` 为爆发窗口终点；比较最后 1／3 天与更早日均点击，复核既有提交时间模型，并分别测试纯截止前点击模型、+14 天模型及逾期交互。[汇总结果](../results/tma_predeadline_burst_20261103/README.md)含逐日曲线、模型、配对和簇内差值；学生级新特征留在 `outputs/`。

运行时在仓库根目录执行 [README 的命令](../README.md)，每次使用新的 `outputs/` 路径。`results/` 是已完成实验的汇总留档，不可直接替代所需的学生级上游 `parquet`。`tma_matched.py` 的 `--three`、`--four` 指向节点运行目录，`tma_score_probe.py` 还需要 `--matched-dir`。详细参数和哈希见各自协议文件。

## 4. 实验轮次与证据位置

| 轮次 | 人群/窗口 | 留存目录 | 状态 |
|---|---|---|---|
| 最早本地初跑 | 全部适用记录、开课第 0–27 天 | [local_uci_20261102](../results/local_uci_20261102/README.md) | 历史探索 |
| 修订与质量检查 | 同上；含原始行/去重敏感性及模型稳健性 | [revised_20261102](../results/revised_20261102/README.md) | 历史探索 |
| 时间集中度扩展 | 同上 | [timing_20261102](../results/timing_20261102/README.md) | 时间特征无清楚额外模型增益 |
| 原四周 5% 点击配对 | 同批次、Fail 对 Pass/Distinction | [matched_20261102](../results/matched_20261102/README.md) | 原 proposal 的核心回顾性比较 |
| 首次计分 TMA 节点模型 | `d−13…d+14/21/28` | [assessment_node_windows_20261103](../results/assessment_node_windows_20261103/README.md)；[早期 +14 版](../results/assessment_node_20261103/README.md) | 三截点模型，+14 早版和统一特征版稍有差异 |
| **TMA +21/+28 天配对** | 同批次、各截点分别匹配总点击 | [tma_matched_21_28_20261103](../results/tma_matched_21_28_20261103/README.md) | **当前研究问题的核心结果** |
| **TMA 前后各两周改写候选** | 同批次、后段 14 天点击匹配；前段用于起点核查 | [tma_two_week_20261103](../results/tma_two_week_20261103/README.md) | 已完成配对、聚类和同截点模型；前段仍不平衡，模型增益很小 |
| 两周点击量簇内比较 | 先按后段点击量分簇，再在同批次 5% 配对内比较 P/F | [tma_click_clusters_20261103](../results/tma_click_clusters_20261103/README.md) | 已完成；两正点击簇的持续性方向相似，前段相近子集有差异 |
| 两周点击相对差距模型 | 原 M0/M1 加入前后点击对称相对差距 | [tma_relative_gap_20261103](../results/tma_relative_gap_20261103/README.md) | 已完成；同切分 AP 增量接近零 |
| 两周首次 TMA 成绩交互 | 评分记录的分层与成绩×前后点击差距；同分数和后段点击配对 | [tma_score_gap_20261103](../results/tma_score_gap_20261103/README.md) | 已完成；成绩有回顾性预测增益，交互项无可靠 AP 增益 |
| 两周首次 TMA 提交时间 | 计划截止日与实际提交日间隔；同成绩和后段点击配对 | [tma_submission_timing_20261103](../results/tma_submission_timing_20261103/README.md) | 已完成；小幅模型增益，失败者在配对中平均较晚提交；延期信息缺失 |
| TMA 截止前短时爆发及逾期 | 截止前 1／3 天点击速率变化；只用截止前信息及 +14 天两种模型；逾期交互和匹配 | [tma_predeadline_burst_20261103](../results/tma_predeadline_burst_20261103/README.md) | 已完成；截止前相对点击集中，预测增量很小；+14 天模型无明显增益 |
| TMA 成绩联合匹配 | 同截点、点击＋首份 TMA 分数 | [tma_score_probe_20261103](../results/tma_score_probe_20261103/README.md) | 仅已记录分数者；回顾性敏感性 |
| 成绩/排名模型 | 同二分类人群 | [tma_score_model_20261103](../results/tma_score_model_20261103/README.md) | 回顾性模型，不代表实时可用分数 |
| HGB/校准探针 | +21/+28 天各自的节点特征 | [wu_method_probe_20261103](../results/wu_method_probe_20261103/README.md)、[+28 天](../results/wu_method_probe_28d_20261103/README.md) | 方法对照，没有确立可靠 AP 增益 |
| E0–E3 课程情境消融 | 原四周、固定第 27 天或 TMA +14 天，按实验而异 | [计划](EXPERIMENT_PLAN_20261103.md)、[结果](EXPERIMENT_RESULTS_20261103.md)、`experiment_design_20261103/` | 已运行 E1–E3，E4 未运行；不能与 +21/+28 天结果混算 |
| “截止时未提交”候选题 | 截止日无提交、到 `d+14` 仍注册 | [可行性](RESEARCH_REFRAMING_20261103.md)、[初步发现](reframe_20261103/EXPLORATORY_FINDINGS.md)、`reframe_20261103/` | 有样本审计和粗分层；核心控制分析未完成 |

本目录的审计脚本及输出见 [研究材料索引](README.md)。`assessment_node_probe/` 是混合最早非期末考核的早期探针，不能和“首次权重非零 TMA”主节点互换。早期文献笔记中“本地未发现代码/数据”的描述仅反映 11 月 2 日当时快照。
