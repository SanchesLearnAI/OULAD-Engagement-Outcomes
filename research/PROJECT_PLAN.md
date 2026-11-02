# OULAD 项目初步规划

目标：在前四周活动总量相近的学生中，分析时间分布和资源使用方式与最终成绩的关系。按以下四部分推进，暂不设时间安排。

## 1 研究范围

- 数据：OULAD；每个样本为“学生 × 课程 × 开课批次”。
- 窗口：仅使用第 0–27 天的行为。
- 标签：Pass/Distinction 对比 Fail，Fail 为正类；Withdrawn 单独描述。
- 问题：活动量相近时，哪些行为与成绩有关？加入这些行为后，模型是否比仅用活动量更有效？

**参考方法：** 按 OULAD 数据描述论文 [1] 确定字段、表关系和日粒度限制。窗口和标签采用本项目 proposal 的设定。

## 2 数据准备与特征构建

以 `studentInfo` 为主表，筛选 `studentVle` 的第 0–27 天，再连接 `vle` 获取资源类型。检查重复、缺失、连接前后样本数，保留零点击学生并单独标记。

| 特征组 | 第一版特征 | 方法来源 |
|---|---|---|
| 活动总量 | 总点击数、对数点击数 | 参考 [4][5] 的活动特征 |
| 时间结构 | 活跃天数、四周点击占比、最长无活动间隔、活动集中程度 | 参考 [2] 的时间行为分析思路、[3] 的日/周聚合；具体指标由本项目定义 |
| 资源组合 | 各类资源点击占比、资源类型数、资源使用多样性 | 参考 [3] 的活动类型聚合；占比与多样性由本项目定义 |

四周固定为 0–6、7–13、14–20、21–27 天。零点击学生的占比、熵等指标需明确处理规则。

**产出：** 特征表、数据字典、清洗规则及样本统计。可参考 ouladFormat [3] 的聚合规则，在 Python 中实现。

## 3 相近活动量比较与行为聚类

- **相近活动量比较：** 在同一课程、同一开课批次内，按总点击量分层或设相近范围，比较通过与失败学生的时间和资源特征；检查不同范围下结论是否一致。
- **行为聚类：** 比较 K-means 与凝聚层次聚类，主要使用时间、资源占比和规律性特征，聚类完成后再查看各群体的成绩分布。
- **结果检查：** 用轮廓系数、重复运行和重采样检查聚类质量与稳定性；描述高活动失败、低活动成功群体。

**参考方法：** [2] 提供“按时间行为聚类，再解释成绩”的思路；[6] 提供聚类算法和内部评价指标的比较方法。相近点击量分层是本项目的设计，不作为上述论文已验证的方法。簇数依据行为结构和稳定性选择，最终成绩不参与聚类或簇数选择。

**产出：** 行为群体画像、活动曲线、资源组合图和成绩分布。DTW 时间序列方法 [2] 可作为后续扩展。

## 4 模型对照与评估

先用逻辑回归比较以下特征组合，课程背景变量保持一致：

| 模型 | 特征 |
|---|---|
| 基线 | 活动总量 |
| 时间模型 | 活动总量＋时间结构 |
| 资源模型 | 活动总量＋资源组合 |
| 完整模型 | 活动总量＋时间结构＋资源组合 |

另保留纯点击量基线；用随机森林分别比较基线与完整特征，检查非线性关系。

**参考方法：** 借鉴 [4] 的学生分组划分、预处理 Pipeline、模型对照和特征消融；借鉴 [5] 的可解释逻辑回归和 PR 指标展示。按本项目需求，置信区间采用学生层面的配对 bootstrap。

- 同一学生不能同时进入训练与测试集；预处理、调参及阈值选择均在训练内部完成。
- 所有模型使用同一未经匹配筛选的测试集。
- 报告 Fail 的 precision、recall、F1，以及 Average Precision、类别比例和相对基线的提升区间。
- 检查不同开课批次的结果是否一致；提升很小或不稳定也如实报告。

**产出：** 模型对照表、特征消融图和行为解释。结论限于关联，点击不能直接代表学习时长或学习能力。

## 参考论文与代码

| 编号 | 论文 | 可参考代码 |
|---|---|---|
| [1] | Kuzilek et al., 2017：[Open University Learning Analytics dataset](https://doi.org/10.1038/sdata.2017.171) | 数据定义参考 |
| [2] | Peach et al., 2019：[Data-driven unsupervised clustering of online learner behaviour](https://doi.org/10.1038/s41539-019-0054-0) | [PartitionStability](https://github.com/michaelschaub/PartitionStability)，算法组件 |
| [3] | Howard, 2025：[ouladFormat](https://arxiv.org/abs/2501.08366)，预印本 | [ouladFormat](https://github.com/ehoward1/ouladFormat)，R 包 |
| [4] | Velychko et al., 2026：[Comparing machine learning models for academic success prediction in online learning](https://doi.org/10.55056/cte.1576) | [oulad](https://github.com/ssemerikov/oulad)，Python |
| [5] | Shaikhanova et al., 2025：[Interpretable Predictive Modeling for Educational Equity](https://doi.org/10.3390/bdcc9110297) | [oulad-early-warning](https://github.com/KuznetsovKarazin/oulad-early-warning)，Python |
| [6] | Araka et al., 2022：[Using Educational Data Mining Techniques to Identify Profiles in Self-Regulated Learning](https://doi.org/10.19173/irrodl.v22i4.5401) | 未确认作者代码仓库 |

参考代码需调整窗口、标签和评估流程。[5] 的公开实现存在时间截断等问题，仅选择性借鉴；代码核对详情见 [已有调研笔记](PROJECT_BRIEF_AND_LITERATURE.md)。
