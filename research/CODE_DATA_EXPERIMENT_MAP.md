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
| TMA 成绩联合匹配 | 同截点、点击＋首份 TMA 分数 | [tma_score_probe_20261103](../results/tma_score_probe_20261103/README.md) | 仅已记录分数者；回顾性敏感性 |
| 成绩/排名模型 | 同二分类人群 | [tma_score_model_20261103](../results/tma_score_model_20261103/README.md) | 回顾性模型，不代表实时可用分数 |
| HGB/校准探针 | +21/+28 天各自的节点特征 | [wu_method_probe_20261103](../results/wu_method_probe_20261103/README.md)、[+28 天](../results/wu_method_probe_28d_20261103/README.md) | 方法对照，没有确立可靠 AP 增益 |
| E0–E3 课程情境消融 | 原四周、固定第 27 天或 TMA +14 天，按实验而异 | [计划](EXPERIMENT_PLAN_20261103.md)、[结果](EXPERIMENT_RESULTS_20261103.md)、`experiment_design_20261103/` | 已运行 E1–E3，E4 未运行；不能与 +21/+28 天结果混算 |
| “截止时未提交”候选题 | 截止日无提交、到 `d+14` 仍注册 | [可行性](RESEARCH_REFRAMING_20261103.md)、[初步发现](reframe_20261103/EXPLORATORY_FINDINGS.md)、`reframe_20261103/` | 有样本审计和粗分层；核心控制分析未完成 |

本目录的审计脚本及输出见 [研究材料索引](README.md)。`assessment_node_probe/` 是混合最早非期末考核的早期探针，不能和“首次权重非零 TMA”主节点互换。早期文献笔记中“本地未发现代码/数据”的描述仅反映 11 月 2 日当时快照。
