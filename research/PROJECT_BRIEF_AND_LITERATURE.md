# OULAD 项目理解与论文代码调研

核对日期：2026-11-02。用途：正式分析前确认研究范围，选择可参考的方法与代码。

## 1 项目现状与研究问题

本地 `Group9_Proposal.pdf` 与 `OULAD_DataMining/OULAD_Project_Proposal.pdf` 的 SHA-256 相同，是同一份两页 proposal。另有 `PROJECT_LOCATIONS.md` 记录 GitHub 与 HPC 位置。本地尚未发现原始 CSV、notebook 或分析脚本。

[项目公开仓库](https://github.com/SanchesLearnAI/OULAD-Engagement-Outcomes) 当前主分支包含 README、项目位置说明、requirements.txt 和 requirements-lock.txt；未见分析代码。README 与 proposal 一致。位置说明记录了 HPC Python 3.11 环境，但本次未登录 HPC，不能据此判断远端数据和实验是否已准备完成。

研究问题：在同一 module presentation 内，前四周活动总量相似的学生，其时间分布和资源使用方式，与最终通过或不通过有什么关联？

- 观测单位：`id_student × code_module × code_presentation`。
- 特征窗口：严格限制在第 0–27 天。
- 主要标签：Pass/Distinction 对比 Fail；Withdrawn 单独描述。
- 数据：studentInfo、studentVle、vle。
- 核心方法：相近点击量比较、无监督行为聚类、仅活动量的 logistic regression、加入行为特征的 logistic regression、random forest。
- 评估：按学生 ID 分组划分；同一未经匹配筛选的测试集；precision、recall、F1、PR 指标；聚类稳定性及跨 presentation 一致性。
- 解释边界：观察性关联，点击不等于学习时长或努力，不推断因果。

依 proposal：11 月 7 日完成数据准备和探索，11 月 20 日完成主要实验，11 月 27 日交 poster，12 月 4 日展示，12 月 11 日交最终包。此处为 proposal 时间表，尚未独立核对课程公告。

## 2 文献筛选原则

优先同时满足：确实使用公开 OULAD；问题与时间规律、资源组合或早期预测有关；方法和结果能查到原文；代码由论文明确关联或由作者提供。把论文报告的结果与本项目可以达到的结果分开，把完整实验代码与通用算法工具箱分开。以下代码仅做静态核对，未声称完成运行复现。

## 3 优先参考的六篇文献

### 3.1 数据定义基础

Kuzilek, Hlosta & Zdrahal (2017), **Open University Learning Analytics dataset**, Scientific Data 4, 170171。

[论文](https://doi.org/10.1038/sdata.2017.171)；[UCI 数据集与下载](https://doi.org/10.24432/C5KK69)。

数据来自 2013–2014 年，涉及 7 个 module、22 个 module presentation；studentInfo 有 32,593 条记录，studentVle 有 10,655,280 条按日汇总的交互记录。分析时应另算唯一学生数，不能把 studentInfo 行数直接当作独立学生数。该文是数据描述论文，没有可作为分类目标的 accuracy/F1。

对本项目：确定字段含义、表间连接、缺失值和日粒度限制。proposal 中 2015 年 UCI 数据引用与 2017 年论文引用是不同资源，可以同时保留。OULAD 不能提供同一天内的精确点击顺序，也不能直接测量学习时长。

### 3.2 时间行为聚类的核心参考

Peach, Yaliraki, Lefevre & Barahona (2019), **Data-driven unsupervised clustering of online learner behaviour**, npj Science of Learning 4, 14。

[正式论文](https://doi.org/10.1038/s41539-019-0054-0)；[可读全文](https://arxiv.org/html/1902.04047v2)；[Markov Stability 代码](https://github.com/michaelschaub/PartitionStability)；[论文列出的 DTW 工具](https://github.com/pierre-rouanet/dtw)。

方法：时间序列 → DTW 相似度 → RMST 稀疏图 → Markov Stability 多尺度聚类，再用成绩解释各群体。

OULAD 实验使用 100 名学习者、约 250 天记录，获得稳定的三个群体：两个主要群体及一个离群群体；作者报告 7 名低表现学习者中 6 名位于集中学习群体。正文另一项 Imperial College 实验也出现了 6/7，引用时需定位到 Fig. 6B 的 OULAD 实验。

对本项目：借鉴“先按行为聚类，再解释成绩”及稳定性检查。这里没有控制相同点击量，也不是前 28 天预测。代码提供算法组件，不能当作论文全部实验的一键复现仓库；PartitionStability 使用 MATLAB/C++，README 声明 GPL v3 或更新版本。

### 3.3 活动量分群及其局限

Araka, Oboko, Maina & Gitonga (2022), **Using Educational Data Mining Techniques to Identify Profiles in Self-Regulated Learning: An Empirical Evaluation**, IRRODL 23(1), 131–162。

[论文及全文](https://www.irrodl.org/index.php/irrodl/article/view/5401)；DOI：10.19173/irrodl.v22i4.5401。尚未找到可确认的论文作者 GitHub 实验仓库。

使用 AAA 的 2013J、2014J，共 735 条学生记录，比较 K-means、EM 和凝聚层次聚类。作者最终选择层次聚类的四群方案；Table 5 的四群 silhouette 为 0.7095，三群为 0.7111，四群选择还依赖其他指标，不能称其为 silhouette 最优。分群与最终结果的卡方检验 p 约为 8.99×10^-22。

对本项目：可参考聚类评价和 outcome 分布展示。但其分群主要对应点击量高低；我们的研究应进一步分离总量与时间/资源结构。不能直接将点击量群体命名为真实自我调节能力，也不能把该 p 值解释为因果或预测性能。

### 3.4 优先借鉴的 Python 模型评估框架

Velychko, Chepurna & Fedorenko (2026), **Comparing machine learning models for academic success prediction in online learning**, CTE Workshop Proceedings，2026-10-17 Early Access。

[论文](https://doi.org/10.55056/cte.1576)；[论文明确链接的 GitHub](https://github.com/ssemerikov/oulad)；[核心脚本](https://github.com/ssemerikov/oulad/blob/main/oulad.py)。

比较 LR、决策树、RF、Gradient Boosting；采用学生分组测试集及重复分组交叉验证。标签为 Pass/Distinction 对比 Fail/Withdrawn，包含人口、活动和作业特征。论文报告 GB 在 30/60/90 天窗口的 holdout macro-F1 分别为 0.706、0.751、0.775。代码 README 给出的 W30 LR/RF/GB 分别为 0.6961/0.7060/0.7063。

对本项目：借鉴 Pipeline、分组划分、同一测试学生、特征消融及结果导出。必须改成 0–27 天、剔除 Withdrawn，并重写时间/资源特征。论文主动指出测试集选阈值、特征可用时间等限制。公开根目录未见明确 LICENSE，参考设计与直接复制代码须区分。

### 3.5 可解释早期预警与复用风险

Shaikhanova et al. (2025), **Interpretable Predictive Modeling for Educational Equity: A Workload-Aware Decision Support System for Early Identification of At-Risk Students**, Big Data and Cognitive Computing 9(11), 297。

[论文](https://doi.org/10.3390/bdcc9110297)；[论文关联 GitHub](https://github.com/KuznetsovKarazin/oulad-early-warning)。

使用 OULAD 的活动、作业和人口特征，研究第 14 天预警。论文报告 LightGBM ROC-AUC 0.789、AP 0.722；LR 为 0.783、0.713。标记 15% 学生时，precision 约 84%、recall 约 35%。这些数字是论文报告值，不是本次运行结果。

对本项目：适合参考可解释 LR、PR 曲线、预警覆盖率与误报的权衡。仓库有 MIT LICENSE，但静态检查发现首次作业成绩未被统一截断到预测日期、分组及缺失值处理也需修改，不能直接认可其为严格的第 14 天无泄漏基线，详见下一节。

### 3.6 数据预处理的实用参考

Howard (2025), **ouladFormat R Package: Preparing the Open University Learning Analytics Dataset for Analysis**, arXiv:2501.08366，本文按预印本引用。

[论文和示例](https://arxiv.org/html/2501.08366v1)；[作者 GitHub](https://github.com/ehoward1/ouladFormat)；[CRAN](https://CRAN.R-project.org/package=ouladFormat)。

提供表合并、日/周活动聚合、活动类别聚合和不同点击变换。主要价值是数据准备；其 DDD-2013J 示例报告 decision tree accuracy 0.5262、RF 0.5011，示例的标签/样本筛选不同于我们的二分类，不作为性能基准。

对本项目：优先参考 `dataset_VLE_time.R`、`dataset_VLE_activity.R`、`combined_dataset.R`。核对源码后，week_begin=1、week_end=4 对应 date 0–27，与 proposal 相符。DESCRIPTION 标注 GPL >=2。我们现有环境是 Python，建议借鉴规则并验证聚合，而非为整个项目更换语言；零点击样本与多次选课保留规则需自行明确。

## 4 GitHub 代码静态核对与复用建议

### 4.1 优先参考 ssemerikov/oulad 的评估结构

`oulad.py` 中确实存在 `StratifiedGroupKFold`，以及把填补、标准化和编码放进 Pipeline 的实现。应优先借鉴这一结构。

以下是本项目必须改变的部分：

- `build_dataset()` 只写 `date <= cutoff`，会包含负日期；本项目需同时限制 `date >= 0` 和 `date <= 27`。
- 原标签把 Fail 和 Withdrawn 合并，不能直接用于本项目。
- 原始 VLE 特征主要是总点击、活跃天数，没有本项目需要的资源占比与周分布。
- `bootstrap_ci()` 重采样的是记录；同一学生可能有多个观测，本项目应按学生重采样，模型差值采用同一批样本的配对比较。
- `risk_threshold_table()` 在测试结果上探索阈值。我们应在训练内部的验证数据上定阈值，再一次性评价测试集。

上述判断来自[核心源码](https://github.com/ssemerikov/oulad/blob/main/oulad.py)及[作者 README](https://github.com/ssemerikov/oulad/blob/main/README.md)。

### 4.2 KuznetsovKarazin 仓库只选择性参考

发现以下会影响复用的具体实现：

1. [loader.py](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/src/data/loader.py) 只筛选 `date <= 14`，包含开课前日期；跨 chunk 直接把 `active_days` 相加，对跨块重复日期可能重复计数。
2. [merger.py](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/src/data/merger.py) 合并全量首次作业成绩；[assessment.py](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/src/features/assessment.py) 仅为“是否完成”设 `<=14` 条件，标准化成绩仍来自原始首次成绩。[logistic.py](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/src/models/logistic.py) 和 [boosting.py](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/src/models/boosting.py) 都将该成绩选入模型。若首次提交晚于预测日，沿用该流程就会使用未来信息。
3. [训练脚本](https://github.com/KuznetsovKarazin/oulad-early-warning/blob/main/scripts/03_train_models.py) 按 `code_presentation.unique()` 的现有顺序切分，未显式排序年月，也没有排除跨组重复学生。
4. 两个模型的 `predict_proba()` 都用当前预测批次的中位数填补缺失，未复用训练期填补参数。
5. merger 的注释说移除第二周前退课者，但实际条件删除所有 Withdrawn。实际人群必须以执行代码为准。

这些是对所查主分支的静态判断，不能据此确定论文历史实验究竟运行了哪个版本。建议借鉴展示与解释方式，自建符合 proposal 的特征和评估 Pipeline。

### 4.3 聚类与预处理代码的定位

| 代码 | 适合复用的部分 | 主要限制 |
|---|---|---|
| ssemerikov/oulad | 学生分组、Pipeline、模型对照、输出组织 | 标签和窗口不同；许可证未明确；阈值与 bootstrap 要改 |
| KuznetsovKarazin/oulad-early-warning | 活动特征思路、PR 与误报展示 | 已发现时间可用性和处理流程问题，不能整仓直接采用 |
| ehoward1/ouladFormat | 周聚合、活动类型聚合、表合并规则 | R；GPL >=2；样本保留规则需核对 |
| michaelschaub/PartitionStability | 多尺度图聚类与稳定性 | MATLAB/C++；GPL v3+；只是算法组件 |

## 5 根据现有证据建议的项目路线

以下是针对本项目的设计建议，并非已取得的实验结果。优先完成 proposal 中的 LR、RF、可解释特征与聚类，不需要为了参考论文而增加深度模型。

### 5.1 先建立可核对的样本和特征表

以 studentInfo 的学生-课程-开课批次为主表；先筛选 studentVle 的 0–27 天，再与 vle 按课程、开课批次、资源 ID 连接，聚合后左连接回主表。检查连接前后行数、资源表键的唯一性、未匹配资源数、标签数及唯一学生数。

没有窗口内活动的学生不要因 inner join 消失。零点击样本单列标记；其资源占比和时间熵原本无定义，应明确处理，不把它们误解为“稳定学习”。Withdrawn 独立汇总。主任务是在最终非退课人群中的关联分析，其结论不能自然外推为面向全部在读学生的退课预警。

候选特征：

| 方面 | 第一版特征 | 对研究问题的作用 |
|---|---|---|
| 活动总量 | total_clicks、log1p(total_clicks) | 建立点击量基线与可比学生范围 |
| 时间覆盖 | active_days、首个/最后活跃日、最长连续无活动天数 | 区分持续参与与短期集中活动 |
| 时间分布 | 四周点击占比、按日点击熵、后两周占比 | 在相近总量下比较活动如何分配 |
| 资源组合 | activity_type 点击占比、资源类型数、资源熵 | 比较学生访问什么资源 |
| 课程背景 | module/presentation | 控制课程结构与开课批次差异 |

周窗口固定为 0–6、7–13、14–20、21–27。资源占比和周占比各自和为 1，解释 LR 时应选参考类别或处理共线性。以普通日聚合构造跨日序列，不推断同日点击先后。

### 5.2 将活动量可比性做成明确分析条件

在每个 module presentation 内，根据 `log1p(total_clicks)` 分层或设近邻距离阈值。阈值在看测试结果前确定；报告各层 Pass/Fail 数、保留比例与点击量平衡，增加阈值敏感性分析。零点击样本独立比较，避免相对差异除以零。

匹配主要用于描述“总量接近，行为不同”的对比。用于模型优劣判断的测试集仍按 proposal 保持未经匹配筛选；不能用人为平衡后的测试集宣称真实总体 precision 或 AP。该分析控制了活动量与课程背景，仍不能消除所有背景差异，不构成因果识别。

### 5.3 用成对消融回答行为结构有没有增量

建议在同一学生分组划分上运行：

| 模型 | 特征集合 | 要回答的问题 |
|---|---|---|
| M0 | 总点击量 | 活动量本身能解释多少 |
| M0c | 总点击量 + 同一组课程背景变量 | 扣除课程背景后的合理对照 |
| M1 | M0c + 时间结构 | 时间分布是否增加信息 |
| M2 | M0c + 资源组合 | 资源使用是否增加信息 |
| M3 | M0c + 时间结构 + 资源组合 | 两类行为信息是否互补 |
| RF 对照 | 与 M0c、M3 分别相同的特征 | 增量是否只是 LR 对总量的拟合过于简单 |

M0 保留 proposal 的 engagement-only baseline。M0c–M3 使用一致背景变量，避免把加入课程背景的收益错算成行为收益；也可比较 log 点击、分段函数等更灵活的量基线。课程留出检验要重新考虑未知 presentation 的编码。

建议预先固定 Fail 为正类，报告 Fail precision/recall/F1、macro-F1、Average Precision、类别比例及无技巧 AP 参考值。若使用梯形积分的 PR-AUC，另行说明算法，不与 Average Precision 混称。

先按学生划分，再在训练内部做分组交叉验证、预处理拟合和参数/阈值选择；所有模型用同一测试人群。按学生配对 bootstrap 估计指标增量的区间。补充课程或 presentation 留出检验时，确保学生仍不跨训练测试，不能只更换切分列就声称消除了学生重叠。

### 5.4 聚类作为行为描述

第一版在非零活动样本上使用时间占比、资源占比及有限的规律性指标；将总量作为聚类后的描述变量，而非让它主导距离。跨课程合并时检查簇是否只是课程资源差异；必要时在 presentation 内分析。

先比较 K-means 和凝聚层次聚类。簇数依据内部结构、规模、可解释性及重采样稳定性选择，不依据哪个簇数产生最显著的最终成绩差异。可用 silhouette、重复初始化/重采样一致性、簇中心行为曲线作检查。聚类不稳定也是有效发现。

只有第一版已稳健完成，才考虑 DTW 或 Markov Stability。对前 28 天序列，过强的时间扭曲可能抹去“在第几周发生”这一研究信号，因此即使采用 DTW，也应与保留日历位置的表示比较。

高活动失败、低活动成功按各 presentation 内部的活动量分位数定义；分析群体分布，不从少数案例推断因果。也不把论坛点击解释成发帖质量或真实求助次数。

## 6 阅读与实施优先级

1. 先读数据描述论文，固定标签、粒度、键和窗口。
2. 读 Peach et al. 的 OULAD 实验及 Araka et al. 的分群结果，写清楚本项目相对“活动量分群”的研究问题。
3. 参考 ssemerikov/oulad 的评估结构，并参照 ouladFormat 验证聚合规则。
4. 将 Shaikhanova et al. 作为解释与 PR 展示参考；其 GitHub 的现有实现不作为即用基线。
5. 得到严格同协议的结果后，再决定是否值得加入 DTW；暂不以 GPU 模型作为主线。

正式分析前还需确认：HPC 上原始数据及校验值、课程教师反馈、最终评分标准，以及 Fail 正类约定。这些未在当前本地文件中找到；不影响本次资料调研完成。

## 7 证据范围与补充文献

本次已检查本地全部三个原始文件、两页 proposal、项目公开仓库及上述参考仓库的相关文件。未下载全量 OULAD、未运行外部训练代码、未提交 GitHub 修改、未登录 HPC。文献报告的指标均保留其任务范围，不能与本项目的前 28 天、排除 Withdrawn、仅 VLE 行为任务直接排名。

补充背景文献：Nguyen, Huptych & Rienties (2018), **Linking Students’ Timing of Engagement to Learning Design and Academic Performance**，[作者机构全文](https://oro.open.ac.uk/53939/1/p141-nguyen.pdf)，DOI：10.1145/3170358.3170398。该研究讨论提前学习与追赶学习，但使用 2015–2016 年的 387 名学生记录，不是公开 OULAD 的 2013–2014 年数据，故未计入同数据集核心清单。

本次检索没有确认一个同时严格满足“0–27 天、Pass/Distinction 对 Fail、同 presentation 相近点击量比较、按学生分组评估、完整开源实现”的现成研究。这里表示本轮未找到完全匹配项，不表示这个问题从未被研究；正式报告不应据此宣称首创。
