# TwoLine 合并差异记录

2026-11-21 · 合并前 `fe6d90b` · 合并后 `4a9c65e`

Oliver（`Oliver-ming`）的 PR #6 已合并。新增内容主要是补充实验和交付文件，研究主结论保持一致。[PR #6](https://github.com/SanchesLearnAI/OULAD-Engagement-Outcomes/pull/6) · [合并前后完整差异](https://github.com/SanchesLearnAI/OULAD-Engagement-Outcomes/compare/fe6d90b...4a9c65e)

## 1. 合并前后的研究结果

| 项目 | 合并前 | 合并后 |
|---|---|---|
| 行为主分析 | 19,806 条课程记录，6,092 条 Fail，主分析 4,768 对 | 六项主要指标及主配对表保持一致。Fail 一侧活跃天数少 **0.368 天**、最长无点击空档长 **0.389 天**、第二周点击占比低 **3.68 个百分点** |
| 更严格的行为敏感性 | 同时匹配成绩和前段点击后 410 对，三项区间均跨零 | 保留原结果和结论边界 |
| 冻结 M2 单次主测试 | 无分数逻辑回归 Fail AP **0.661077** | 同一主切分，AP **0.661077** |
| 含分数模型对照 | 五次学生分组测试中，逻辑回归 **0.713642**，XGBoost **0.719919** | 保留原实验和指标 |
| 无分数 XGBoost 对照 | 尚无对应 XGBoost 实验 | 同五次学生分组测试、相同 16 项数值特征，逻辑回归 **0.671415**，XGBoost **0.678759**，平均 **+0.007344 AP** |
| M2 补充五折验证 | 五折平均 AP **0.67015** | 五折平均 AP **0.66988**，每折学生组成有差异，平均值相差约 **0.00027** |

无分数 XGBoost 的五次 80/20 切分用于同条件模型对照。冻结 M2 的 0.661077 来自单独的 60/20/20 主切分。补充五折验证又是另一套分组。三套结果各自有对应的用途。[行为证据报告](BEHAVIOR_REPORT.md) · [建模报告](MODELING_REPORT.md) · [无分数 XGBoost 评估](../results/xgboost_noscore_20261112/README.md)

## 2. Oliver 的 PR 增加了什么

- 恢复截止前 1 天、3 天的点击集中度及其与迟交的交互，恢复前后点击量的相对差 `(后段−前段)/(后段+前段)`。这些指标进入探索分析。相对点击差加入 M2 后，单次主测试 AP 约增加 **0.00007**。
- 新增 C5 点击量分簇配对和 C6 前后点击变化消融，补齐双线叙事的证据层级。
- 新增中文报告 PDF，更新英文报告、海报和项目 ZIP。两个独立的研究 Markdown 报告仍是各条线的入口。 [变更说明](../CHANGE_REQUEST.md) · [C6 指标表](../oulad-sem/outputs/tables/t11_x_relative_gap.csv)

## 3. 合并前补修了什么

**分簇提交日统计。** 原 PR 输出的三个簇都缺少配对差值和区间，因为把有缺失提交日的配对也带入计算。修复后只使用双方均有可用提交日的配对。表中差值为「最终未通过 − 最终通过或优秀」，正值表示平均较晚提交。

| 截止后点击量簇 | 有效配对 | 提交日差值 | 95% 区间 |
|---|---:|---:|---:|
| 零点击 | 222 | +0.49 天 | [−0.32, +1.28] |
| 低点击（1–42 次） | 1,755 | +1.18 天 | [+0.90, +1.46] |
| 高点击（43 次及以上） | 1,879 | +1.68 天 | [+1.41, +1.93] |

[修正后的 C5 配对表](../oulad-sem/outputs/tables/t11_x_volume_cluster_effects.csv)

**五折分组复现。** 旧交付使用 scikit-learn 1.8.0，Oliver 重跑使用 1.9.1。两版对同样大小的学生组排序不同。现在项目显式固定为 Oliver 发布版的规则，重跑的逐折样本数、未通过比例和 AP 与该版一致。补充五折的测试学生不同，合并前后的 AP 差值不能用来衡量模型改进。Windows 检出也为冻结配置保留 LF 换行，使配置哈希跨平台一致。 [五折切分表](../oulad-sem/outputs/tables/t05_split_summary.csv) · [修复提交](https://github.com/SanchesLearnAI/OULAD-Engagement-Outcomes/commit/dc58af9ccf1fee2d43103b94efd498eed3812822)

## 4. 验证和交付

全量重跑 `run-20261121T083050Z-2803af` 的 S00–S12 校验为 **0 错误、0 警告**。修复提交的四项 GitHub 检查均通过，本地根目录 28 项单元测试通过。重新生成的中英文报告和海报均可读取，ZIP 中的分簇表及关键代码与本次生成结果一致。

[英文报告](../oulad-sem/Report.pdf) · [中文报告](../oulad-sem/Report_zh.pdf) · [项目 ZIP](../oulad-sem/Group_09_Project.zip) · [GitHub 检查](https://github.com/SanchesLearnAI/OULAD-Engagement-Outcomes/pull/6/checks)
