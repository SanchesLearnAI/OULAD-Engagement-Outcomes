# OULAD 项目修订指导（Best Version）

> **交付对象**：执行修订的 agent。本文档是操作契约，不是建议清单。
> **基线 run**：`run-20261103T194449Z-838d99`（S00–S12 全 pass，`validate_run.py` 0 errors / 0 warnings）
> **目标**：先修正确性问题，再把主结论做强、做透明，最后重出 Report / Poster / README / ZIP。
> **原则**：旧 run 只读；所有代码/config 改动用新 `run_id`；日志 append-only；不伪造数字；不写因果；引用按 `EXECUTION_SIMULATION.md` 9.4 分级。

---

## 0. 先读什么

按顺序：

1. `START_HERE.md`
2. `01_plan/EXECUTION_SIMULATION.md`（重点 S07–S12、9.4）
3. 本文件（`01_plan/REVISION_GUIDE.md`）
4. `src/reporting.py`（Report/Poster/README 的生成器，不是手改 `Report.tex`）
5. `stages/s07_clustering.py`、`s08_models.py`、`s09_sequential.py`、`s10_survival.py`、`s11_robustness.py`、`s12_package.py`
6. `src/features.py`（`build_features` / `feature_columns` / `prepare_vle`）
7. `outputs/tables/t06_balance.csv`、`t08_model_metrics.csv`、`t09_incremental.csv`、`t10_hazard_ratios.csv`、`t11_specs.csv`、`t11_negative_controls.csv`
8. `runs/run-20261103T194449Z-838d99/run.jsonl`、`stages/*.stage.json`

**重要**：`Report.tex`、`Poster.tex`、`README.md` 是 S12 生成的。要改文字/表格，改 `src/reporting.py`（`build_latex` / `build_poster` / `build_readme`），然后重跑 S12；只手改生成的 `.tex` 会在下一次 S12 被覆盖。

---

## 1. 修订范围

### 必须修（WP0）

- S10 cause-specific Cox 用未来 W2 特征，Appendix Table 9 的 Withdrawn W2 hazard ratios 无效。
- B2 offset 校准斜率 0.051、Brier 0.304，却被当成正常对比。
- `pre_start=feature` 规格是假的：`P009_pre_feature` 与 `P000_baseline` 完全相同。
- `config.yaml` 写 `at_risk_at_cutoff: true`，但主队列实际未过滤。

### 应该补（WP1）

- 轨迹分化分析（trajectory divergence）。
- Context-only baseline + context-adjusted ladder。
- Per-module / subgroup 校准与公平性。
- Quiz 纳入敏感性、不同 engagement 操作化、匹配分箱敏感性。
- E-value 与 negative-control outcome。
- PCA 散点图 + silhouette。

### 展示/工程（WP2/WP3）

- Report 结构、limitations 表、页数说明、Table 7 split、93 个缺失退课日期。
- Poster 排版、PCA/轨迹图。
- README `pip install -r`、复现清单、单元测试、一键运行。
- ZIP 只保留 `Group_09_Project.zip`，补 `STATUS.final.json`。

### 可选（WP4）

- TMA+21/+28、TMA+14、首次 TMA 分数联合匹配：只做 appendix / upper-bound，明确不可作 early-warning 主结果。若样本或时间可用性不支持，直接不做并写清楚。

### 后续诊断（WP5）

- Context 增量给 paired bootstrap CI（M1c−M0c），判断是否可与 0 区分。
- B2 从 Table 7 主表降级到 appendix/footnote，不再算“有效方法”。
- Negative controls 改成 diagnostics：pre-start-only 是 positive control，future-window 是 placebo failure，补一个 negative-control exposure；不要声称它们是 clean null。
- TMA 仍按 WP4，默认不做；若做只进 appendix。

---

## 2. WP0 — 正确性修复

### WP0.1 S10 Cox 的时序泄漏

**问题**

`stages/s10_survival.py:164-184` 读取 `f56`，把 `w1_*`、`w2_*`、`d_clicks` 作为静态协变量放进 `PHReg`：

```python
f56 = read_features(ctx, 56)
cox_cols = ["w1_clicks_log1p", "w1_active_days", "w1_entropy_activity",
            "w2_clicks_log1p", "w2_active_days", "w2_entropy_activity",
            "d_clicks"]
d = sdf.merge(f56[KEY + cox_cols], on=KEY, how="left")
res = PHReg(time, X, status=status).fit()
```

Withdrawn 的 `event_week` 是退课周；第 10 周退课的学生，模型用了第 28–55 周的 W2 特征。`Report.tex` Appendix Table 9 的 Withdrawn W2 HRs 因此无效。Cox 还是全样本 fit、in-sample concordance。

**推荐实现（按优先级）**

**A. 删掉 Cox，保留三分类 + 累积发生率（最稳）**

- 在 `stages/s10_survival.py` 删除 `cox` 子步骤，或加 config `survival.cox: false` 默认关闭。
- 删除 `Report.tex` Appendix Table 9 及正文 §4.5 对 Cox 的引用。
- 保留 `t10_class_metrics.csv`、`t10_cumulative_incidence.csv`、`f10_cumulative_incidence.png`。
- 报告里写：“Withdrawn is modeled as a separate class; cumulative incidence is descriptive; no hazard-ratio claims.”

**B. Landmark Cox（如果要做，做这个）**

对 landmark `L ∈ {28, 56}`：

1. 队列：`date_unregistration.isna() | date_unregistration > L`；只保留在 L 时仍在册的人。
2. 事件：只看 L 之后发生的 cause-specific event；L 之前的事件作为已发生、不应进入队列。
3. 特征：
   - `L=28`：只用 W1 特征（0–27），不含任何 W2/delta/score。
   - `L=56`：W1 + W2（28–55），所有特征在 L 时可得。
4. 时间：`time = max(1, event_week - L/7)`；censoring 在课程结束。
5. 拟合/评估：在 train split 上 fit，在 test split 上算 concordance；报告 HR + CI + test concordance。
6. 输出：`t10_landmark_cox.csv`；报告用新表替换 Table 9，标题写 “Landmark cause-specific Cox at day 28/56; features available at landmark; test concordance”。

**C. 时变 Cox（成本最高）**

用 `t10_long_table.parquet` 的 `weekly_clicks` 作为 time-varying covariate，事件时间用 `event_week`；不要用静态 W2。只有在 B 仍不能满足时再做。

**验收**

- `L=28` 的 Cox 特征列中没有任何 `w2_*` / `d_*` / `score`。
- 没有事件早于 landmark 的样本进入该 landmark 队列。
- 报告不再出现“static W2 HR for Withdrawn”。
- 若选 A，Appendix Table 9 删除；若选 B，新表有 test concordance。

**日志**

- `st.metric("cox_landmark", {...})` 记录 L、n、events、features、test concordance。
- `st.check("cox_no_future_features", ...)` 断言 L=28 特征无 W2。
- `st.check("cox_landmark_cohort", ...)` 断言语料中没有 L 前事件。

---

### WP0.2 B2 offset 标记为 failed specification

**问题**

`stages/s09_sequential.py:196-210` 的 B2 结果：PR-AUC 0.759、Brier 0.304、calibration slope 0.051。报告 Table 6 把它和 W2_only/B1 并列，正文只说“3 of 4 methods have CI above zero”。

**要求**

1. 先尝试一次修正：`sm.GLM(y, Xw2_scaled, family=Binomial(), offset=logit(p_w1)).fit()`，不加 `add_constant`，或检查是否因截距 + offset 导致不可识别。
2. 如果修正后 calibration slope < 0.5 或 Brier > 0.25，标记 failed：
   - `t09_incremental.csv` 增加 `status` 列（`ok` / `failed_calibration`）。
   - 报告 Table 6 加脚注：“B2 is a failed specification: calibration slope 0.05, Brier 0.30; not evidence about sequential updating.”
   - 正文不要再用“3 of 4 methods”这种把 B2 当合法方法的表述。
3. 如果修正成功，更新 Table 6 和正文；仍要报告 calibration。

**验收**

- 报告对 B2 有明确“failed / misspecified”字样，或 B2 从主表移除。
- 不再出现“B2 offset adds nothing”这种把失效模型当负结果的句子。

---

### WP0.3 修好 `pre_start=feature` 规格

**问题**

`stages/s11_robustness.py:54-56` 构建了 `pre_start=feature` 的特征表；`src/features.py:252-262` 生成 `pre_start_clicks` / `pre_start_active_days`；但 `src/features.py:394-425` 的 `feature_columns()` 只选 `w1_*` / `w2_*`，这两个列永远不进模型。

证据：`outputs/tables/t11_specs.csv`：

- `P000_baseline` = 0.8283198448744786
- `P009_pre_feature` = 0.8283198448744786

**要求**

1. 修改 `feature_columns(df, level, cutoff, include_pre_start=False)`：
   ```python
   if include_pre_start and "pre_start_clicks" in df.columns:
       cols += ["pre_start_clicks", "pre_start_active_days"]
   ```
2. 在 S11 的 `add_predictive` 中，当 `pre == "feature"` 时传 `include_pre_start=True`；其他规格传 False。
3. 重跑 S11 → S12；确认 `P009_pre_feature != P000_baseline`，把新旧值写进 `FIX_REPORT.md`。
4. 如果修完后 pre-start 特征导致其他 spec 的列选择变化，检查 `t11_specs.csv` 的 `features` 列和 `config_hash`。

**验收**

- `P009_pre_feature` 与 `P000_baseline` 不同。
- `t11_specs.csv` 仍为 45 行，`n_predictive=37`（除非你 deliberately 增加/删除 spec）。
- 报告 §4.6 的 pre-start 讨论与新值一致。

---

### WP0.4 config 与主队列一致

**问题**

`config/config.yaml:20` 是 `at_risk_at_cutoff: true`；S02 的审计结论是 main cohort NOT filtered，at-risk 只作 sensitivity。报告/README 与配置不一致。

**要求**

- 改成：
  ```yaml
  at_risk_at_cutoff: false      # main cohort keeps all T1
  at_risk_sensitivity: true     # at-risk filter is a sensitivity only
  ```
- 同步更新 README / Report 的 cohort 段落。
- 因为 config 变了，使用新 `run_id`，重跑 S00–S12；先 simulation，再 full。
- 如果代码里 `at_risk_at_cutoff` 没有被实际使用，检查 `stages/s02_master.py` / `src/cohort.py`，确保报告的行为由 `at_risk_sensitivity` 控制，而不是静默依赖旧键。

**验收**

- `config.yaml`、README、Report 三处对 at-risk 的表述一致。
- 新 run 的 S02 check 仍显示“main cohort NOT filtered”。
- `config_hash` 与新 run 的 `run.jsonl` 一致。

---

## 3. WP1 — 核心科学增强

### WP1.1 轨迹分化分析（最高价值）

**问题**

现在只有 W2 平均差异（37.56 clicks），没有回答“从第几周开始分岔”。这是“same engagement, different outcomes”最直接的证据。

**实现**

1. 数据：从 `studentVle` 聚合周度点击；周 = `floor(date/7)`，只取 `date >= 0`。
2. 队列：读 `runs/<run_id>/artifacts/matched_strata.parquet`（S06 产物），只保留同时含 Pass/Distinction 与 Fail 的匹配层。
3. 对每层、每周计算：
   - mean `log1p(weekly_clicks)`；
   - mean active days；
   - 可选：活动类型份额。
4. Bootstrap：按 `id_student` / 匹配层重抽样 1,000 次，给每周均值 CI。
5. 找第一个“之后连续 2 周 CI 不重叠”的周，作为 divergence week；报告 CI。
6. 画 `f12_trajectory_divergence.png`：
   - x = week 0–8；
   - 两条线 = Pass/Distinction vs Fail；
   - 阴影 = bootstrap CI；
   - 竖线 = divergence week；
   - 另画 normalized 版本（weekly share of total）避免只看总量。
7. 输出 `t12_trajectory_weekly.csv`、`t12_divergence.csv`。

**解释边界**

- 只写“descriptive divergence”；W1 匹配、W2 是基线后；不写因果。
- 如果 divergence week 在 W1 内，说明“早期相似”本身不成立，要如实写。
- 如果两条轨迹始终重叠，写“no stable divergence”，不要硬找。

**验收**

- 每周有 n、mean、CI。
- divergence week 有定义、有 CI、有图。
- 报告/附录有图；poster 可放。

---

### WP1.2 Context-only baseline + context-adjusted ladder

**问题**

M0 是“点击量 + 活跃天数”，但没有回答“这是不是只是学生是谁”。报告里的 increment 可能被背景变量解释。

**实现**

1. 新增模型：
   - `M-1` context-only：gender、age_band、highest_education、imd_band、disability、num_of_prev_attempts、studied_credits、module/presentation（one-hot 或固定效应）。
   - `M0c`：M0 + context。
   - `M1c`：M1 + context。
2. 用同一 train/val/test split；同样的预处理只在训练折 fit。
3. 输出 `t08_context_metrics.csv`、`f08_context_comparison.png`。
4. 报告加一段：
   - “context-only baseline reaches PR-AUC X; adding click volume raises to Y; adding behaviour raises to Z.”
   - 如果 M1c ≈ M0c，说明行为结构在背景之外没有增量；这是重要负结果。

**验收**

- M-1/M0c/M1c 都在 held-out test 上评估。
- 没有把 context 变量放进 strict clickstream 特征；它们是单独模型。
- 报告明确“context vs clickstream vs behaviour”的增量。

---

### WP1.3 Per-module / subgroup 校准与公平性

**实现**

1. 按 `code_module`（7 个）和 `code_presentation`（22 个）计算：
   - n、positive rate、PR-AUC、ROC-AUC、recall@precision=0.8、Brier、calibration slope。
   - 小样本给 bootstrap CI；n < 200 的组标 `low_n`。
2. 按 gender、age_band、imd_band（粗化）、disability、highest_education 计算同样指标。
3. 画：
   - `f08_module_heatmap.png`（module × metric）；
   - `f08_subgroup_calibration.png`（每个子组的 reliability curve 或 calibration slope）。
4. 输出 `t08_module_metrics.csv`、`t08_subgroup_metrics.csv`。

**报告**

- 加“heterogeneity and fairness”小节：哪些模块/子组更差；recall 差异；小样本 caveat。
- 不写“模型对某组有因果影响”；只写“在该组上性能较低/校准较差”。

**验收**

- 每个组有 n 和 positive rate；没有空组。
- 子组指标与总体指标用同一 test split。
- 报告有限制说明。

---

### WP1.4 测量敏感性：quiz、engagement 定义、匹配分箱

**A. Quiz 纳入/排除**

- `src/features.py:57-60` 排除了 `quiz`/`externalquiz`（965,483 行，9.06%）。
- 加一个 `include_quiz_sites=True` 的 feature build；用同一 S08 模型阶梯比较 M1/M3 的 PR-AUC、ROC-AUC、校准。
- 输出 `t11_quiz_sensitivity.csv`；报告写“strict vs quiz-included”的差异。

**B. Engagement 操作化**

- 分别用以下暴露定义做 CEM：
  1. total clicks only；
  2. active days only；
  3. activity entropy only；
  4. all three（现状）。
- 比较 W2 click difference、common support、SMD；输出 `t11_exposure_sensitivity.csv`。
- 报告写“same engagement”的结果是否依赖定义；如果依赖，主结论要降级。

**C. 匹配分箱**

- CEM 分箱 5 / 10 / 20；active day bins 粗/细；entropy bins 2/3/4。
- 输出 `t11_matching_bins.csv`；报告范围。

**验收**

- 每个敏感性都有对照的 baseline 数字。
- 结论不依赖单一操作化；如果依赖，明确写。

---

### WP1.5 E-value 与 negative-control outcome

**A. E-value**

- S06 已算 `e_value=1.5545`；把它写进报告 §4.2 / limitations：
  - “An unmeasured confounder with risk ratio ~1.55 could explain the matched difference.”
  - 说明 E-value 很小，匹配结果是脆弱的。

**B. Negative-control outcome**

- 选一个不应由 W2 行为“导致”的基线变量，例如 `gender` 或 `num_of_prev_attempts`。
- 用 W1/W2 特征预测它；如果 AUC 明显高于 0.5，说明行为特征编码了背景差异，是 confounding 警告。
- 输出 `t11_placebo_outcomes.csv`；报告写“diagnostic, not a formal test”。

**验收**

- E-value 在正文出现，不只留在表里。
- Placebo outcome 的结果被解释为“行为特征可能编码背景”，不是因果证据。

---

### WP1.6 PCA 散点图 + silhouette

**实现**

- 用 S07 的 8 个 PCA 分量；画 PC1 × PC2，按 k=2 簇着色。
- 27,885 点：`alpha=0.15` 或分层抽样 5,000 点；可选 hexbin。
- 标题/脚注：
  - PC1 34.3%，PC2 11.9%，二维仅 46.2%；
  - k-means 用 8 个 PC（累计 74.7%）；
  - visualization/diagnostic，不是簇有效性证据。
- 另画 `f07_silhouette.png`（k=2 的 silhouette 分布）。
- 放进 Report appendix 和 Poster。

**验收**

- 图有 variance explained 标注。
- 报告没有用散点图证明簇“真实存在”。

---

## 4. WP2 — Report / Poster / README / ZIP

### 4.1 Report 修改点

在 `src/reporting.py` 的 `build_latex` 中改，不要只改生成的 `Report.tex`：

1. **§3.3 Methodology**：加 M-1/M0c/M1c；说明 context-only baseline。
2. **§4.2 Matching**：
   - 加 E-value 1.55；
   - 加 matching method sensitivity（CEM 37.56 / PSM 7.77 / stratified 33.24）；
   - 加 engagement 定义敏感性（如果 WP1.4 做了）。
3. **§4.3 Classification**：
   - 加 context-only / context-adjusted 结果；
   - 加 per-module / subgroup 校准与公平性。
4. **§4.4 Cutoff / updating**：
   - 加 trajectory divergence（新图/表）；
   - B2 标 failed；
   - 保留 cutoff 曲线。
5. **§4.5 Competing risks**：
   - 如果选 A，删 Cox/Table 9；
   - 如果选 B，换成 landmark Cox，写清“features available at landmark; test concordance”。
6. **§4.6 Robustness**：
   - quiz 纳入/排除；
   - exposure definitions；
   - matching bins；
   - negative controls；
   - 说明 pre-start-only 0.782 是强信号，不是“weak control”。
7. **§5 Limitations**：
   - 加一张表：threat / mitigation / residual risk（leakage、confounding、measurement、external validity、fairness）。
   - 93 个 Withdrawn 无 `date_unregistration`，被当作课程结束事件。
   - Table 7 用三分类 split n=6,508，不是主 test n=4,492。
   - zero-click 4.2% 是 T1；全队列 14.4%。
   - 页数说明：Main text 7 pages; references/disclosures/appendix excluded.
8. **Appendix**：PCA scatter、trajectory、landmark Cox（如果做）。

**验收**

- Report.pdf 编译成功；正文 6–10 页；总页数可 >10，但开头注明 main text 页数。
- 没有 0.58–0.75、0.626/0.663/0.636、69.69→95.23 等被驳回数字。
- 0.7747 仍带“Fail-or-Withdrawn + 含早期分数”的限定。
- 没有因果措辞。

### 4.2 Poster

- 修 Design 框文字溢出；填满下半部分。
- 加 trajectory divergence 图（1 张）和 PCA 散点（可选）。
- Main results 用 M1/M3/M5 + context baseline；不要只放 M1。
- 保留 “associations, not causal” 和 “random-label 0.530”。

### 4.3 README / DATA_DOWNLOAD / ZIP

- README：
  - `pip install -r REQUIREMENTS.lock`，不要 `pip freeze >` 覆盖；
  - 加 reproducibility checklist：data hash、config hash、run ID、`validate_run.py` 输出；
  - 加 `make sim` / `make full`（如果 WP3 做了）。
- ZIP：
  - 只保留 `Group_09_Project.zip`；删除 `Group_XX_Project.zip`（`stages/s12_package.py:117-119`）。
  - 包含 `STATUS.final.json`；不包含原始数据、Poster.pdf。
  - 包含新的 tables/figures、Report.pdf、README、code、DATA_DOWNLOAD.md。

---

## 5. WP3 — 工程与复现

### 5.1 单元测试

新建 `tests/`，至少：

- `test_windows.py`：
  - `bounds(28)` -> W1 0–27；`bounds(27)` -> W1 0–26；
  - first_tma ≤ 28 的 fallback 55；29–55 用 first_tma−1。
- `test_features.py`：
  - `max_date_seen < cutoff`；
  - strict 列无 `score` / `is_banked` / `final_result`；
  - `pre_start=feature` 的 `pre_start_clicks` 在表中；`feature_columns(..., include_pre_start=True)` 包含它。
- `test_splits.py`：
  - train/val/test `id_student` 交集为 0；
  - temporal/LOMO 学生不重叠。
- `test_logging.py`：
  - 事件能通过 `log_event.schema.json`；
  - `seq` 单调。

运行：`pytest -q`。

### 5.2 一键运行

- `Makefile` 或 `scripts/run_all.sh`：
  - `make sim`：S00–S12 simulation；
  - `make full`：S00–S12 full；
  - `make validate`：`validate_run.py`；
  - `make test`：`pytest`。
- README 写一条命令。

### 5.3 审计链

- 新 run 的 `STATUS.final.json` 必须存在；S12 写入并放进 ZIP。
- 旧 run 保留；不要删除。失败的 185405 run 可以移到 `04_archive/`，但不要覆盖。
- `scripts/inspect_raw*.py` 是开发脚本，不要放进 ZIP；若要保留，移到 `04_archive/`。

**验收**

- `pytest -q` 全过。
- `make sim` 从零可跑；`validate_run.py` 0/0。
- ZIP 只含 `Group_09_Project.zip`，无 raw data。

---

## 6. WP4 — 可选 TMA 附录（只做 appendix）

只有在主分析、WP0/WP1 全部完成后，才考虑。**不要放进 main text，不要当 early-warning 结果。**

### 6.1 TMA +21/+28 配对

- 锚点：`first_tma_date`；窗口 `[first_tma+21, first_tma+28]`。
- 只保留在该窗口仍在校的学生；按 W1 engagement 匹配；比较最终结果。
- 输出 `t13_tma_pairing.csv`，图 `f13_tma_pairing.png`。
- 明确：这是 alternative estimand；与 W1/W2 有重叠；描述性。

### 6.2 TMA +14 checkpoint

- 锚点 `first_tma_date`；截止 `first_tma+14`；只用该日之前的数据训练。
- 与固定 cutoff 7/14/21/28 对比；输出 `t13_tma_checkpoint.csv`。
- 明确：这是一个 alternative anchor，不是主曲线。

### 6.3 首次 TMA 分数联合匹配

- 只在 appendix 做 upper-bound / post-hoc：
  - 匹配变量：W1 clicks/active/entropy + first TMA score bin；
  - 结果：final outcome；
  - 明确“score 在早期 cutoff 是否可得无法证明；这不是 early-warning 结果”。
- 输出 `t13_score_matching.csv`。
- 如果分数匹配导致样本损失大或平衡不可达，直接放弃并写原因。

**验收**

- 这三项都不进 Abstract / Introduction / Conclusion 的主结论。
- 每项都有 availability caveat。
- 如果没做，写“not attempted because …”，不要留空。

---

## 7. WP5 — Post-revision diagnostics（本轮新增）

> **目标**：不改主模型，只把三个“解释性遗留问题”处理干净：context 增量有没有 CI、B2 是否从主表降级、negative controls 是否被误读。TMA 仍按 WP4，保持 appendix/不做。

### WP5.1 Context 增量给 CI / 检验

**问题**

`t08_context_metrics.csv` 有 M1c−M0c = 0.001 PR-AUC，但没有差异的 CI；读者无法判断它是否可与 0 区分。

**要求**

1. 在 S08 对**同一 test rows** 做 paired bootstrap（1,000 次；按 `id_student` 或行重抽样，记录 seed）：
   - `delta_pr_auc = PR_AUC(M1c) − PR_AUC(M0c)`
   - `delta_roc_auc = ROC_AUC(M1c) − ROC_AUC(M0c)`
   - 95% percentile CI。
2. 输出 `t08_context_delta.csv`，列：`comparison, delta_pr_auc, ci_low, ci_high, delta_roc_auc, roc_ci_low, roc_ci_high, n_test, seed`。
3. 报告 §4.3 + Table 5 caption 写：
   - 若 CI 含 0：“no detectable behavioural increment beyond context (ΔPR-AUC 0.001, 95% CI [−x, +y]).”
   - 若 CI 不含 0：报告方向与大小，但仍写“small”。
4. 不要为了让 CI 好看而换模型/调参；context 模型保持现状。若想加强 context，只能加预先定义的变量（如 module×presentation、prev attempts、credits），不能看 test 结果再加。

**验收**

- `t08_context_delta.csv` 存在；CI 用同一 test rows；seed 记录。
- 报告不再只写“0.001”而没有不确定性。
- 结论措辞与 CI 一致（含 0 就写“not distinguishable from zero”）。

### WP5.2 B2 从主表降级

**问题**

B2 已标 `failed_calibration`，但 Table 7 仍把它作为一行；读者仍可能把它当“顺序更新方法之一”。

**要求**

1. 主表 Table 7 只保留 `W1_only_M3`、`W2_only`、`All_features_c56`、`B1_stacking` 四个有效方法；B2 不进主表。
2. 在 appendix 加一个小表 `t09_b2_failed.csv`，或在 Table 7 下方加脚注：
   - “B2 offset was attempted; calibration slope 0.051, Brier 0.304; it is a failed specification and not evidence about sequential updating.”
3. 正文不再出现“3 of 4 methods”；只数有效方法（3 of 3 或 4 of 4，按表）。
4. 如果一定要尝试救活 B2：只做一次“无截距 offset + clip `p_W1` 到 [0.02, 0.98]”的预注册修正；若 calibration slope 仍 <0.5，停止，不再调参。B1 已经是可用的顺序更新。

**验收**

- 主表无 B2；B2 的数字只在 appendix/footnote。
- 报告明确“failed specification”。
- 没有把 B2 当负结果或正常方法。

### WP5.3 Negative controls 改为 diagnostics

**问题**

pre-start-only 0.782 是强信号，future-window 0.861 是 post-outcome persistence；它们不是 clean null。当前报告虽然写了，但表格/标题仍叫“negative controls”，容易被误读。

**要求**

1. 把 S11 的输出表 `t11_negative_controls.csv` 增加 `role` 列：
   - `random_labels` → `leak_check`
   - `pre_start_only` → `positive_control`
   - `placebo_window_56_83` → `placebo_failure`
   - `num_of_prev_attempts_gt0` → `negative_control_outcome`
2. 新增一个 `negative_control_exposure`：
   - 用非评估类资源（`homepage`、`forumng`、`resource` 等）的 W2 变化预测结果；
   - 如果它也预测结果，说明 W2 变化可能反映 generic engagement / 背景，而不是学习策略；
   - 输出 `t11_placebo_exposure.csv`，列：`exposure, metric, value, n, note`。
3. 报告把 §4.6 的 “negative controls” 改名为 “Diagnostics and placebo checks”；表 9 加 `Role` / `Interpretation` 列：
   - random-label：leakage sanity check；expected ~0.5。
   - pre-start-only：positive control for early signal；不是 null；pre-course clicks carry information。
   - future-window：post-outcome persistence；placebo failure；不是 leak detector。
   - repeater placebo：negative-control outcome；behaviour may encode background。
   - non-assessment exposure：negative-control exposure；if predictive, generic engagement / confounding。
4. 报告不要写“pre-start is weak”或“future window proves no leakage”。

**验收**

- `t11_negative_controls.csv` 有 `role` 列；新增 `t11_placebo_exposure.csv`。
- 报告表 9 有 `Role` / `Interpretation`；措辞与角色一致。
- 不再把非 null 控制称为“failed null”或“clean”。

### WP5.4 TMA 决策（保持 WP4）

- 默认仍不做；在报告/README 保留 “not attempted because …”。
- 如果要做：只进 appendix；TMA+21/+28 是 alternative estimand；TMA+14 是 alternative checkpoint；first TMA score matching 只作 upper-bound/post-hoc，明确 score 在早期 cutoff 不可得。
- 不得进入 Abstract / Introduction / Conclusion 的主结论。

---

## 8. 执行顺序

1. **备份**：把当前 `src/`、`stages/`、`config/`、`src/reporting.py` 复制到 `04_archive/pre_revision_YYYYMMDD/`（项目不是 git 仓库，必须留底）。
2. **WP0**：改代码/config；先跑 `pytest`；再 simulation；`validate_run.py`。
3. **WP1**：加轨迹、context baseline、module/subgroup、measurement sensitivity、E-value、PCA scatter；simulation 验证。
4. **WP5**：加 context Δ CI、B2 降级、negative-control diagnostics；重跑 S08/S11/S12。
5. **WP2**：改 `src/reporting.py`；重跑 S12；检查 Report/Poster/README/ZIP。
6. **WP3**：测试、Makefile、审计链。
7. **WP4**（可选）：TMA 附录。
8. **全量**：新 `run_id`；simulation 全绿后 full；`validate_run.py` 0/0。
9. **FIX_REPORT.md**：逐条记录问题、改动文件、新 run_id、前后指标、验证结果、剩余 caveat。

**任何阶段失败**：写 `blocked` + 日志 seq；不要跳过 gate，不要用旧 run 的结果冒充新 run。

---

## 9. 验收清单

### 正确性

- [ ] S10 不再有“静态 W2 + 全样本 Cox”；Table 9 删除或换成 landmark，且有 test concordance。
- [ ] B2 标为 failed specification（或修正后 calibration slope ≥0.5）。
- [ ] `P009_pre_feature != P000_baseline`；`t11_specs.csv` 记录新值。
- [ ] config/README/Report 对 at-risk 的表述一致。

### 科学增强

- [ ] `f12_trajectory_divergence.png` + `t12_trajectory_weekly.csv` 存在；divergence week 有 CI。
- [ ] `t08_context_metrics.csv` 存在；M-1/M0c/M1c 与 M0/M1 可比。
- [ ] `t08_module_metrics.csv`、`t08_subgroup_metrics.csv` 存在；子组有 n/positive rate。
- [ ] `t11_quiz_sensitivity.csv`、`t11_exposure_sensitivity.csv`、`t11_matching_bins.csv` 存在。
- [ ] E-value 1.55 出现在正文；placebo outcome 有结果。
- [ ] `f07_pca_scatter.png` 有 variance 标注；`f07_silhouette.png` 存在。

### 后续诊断（WP5）

- [ ] `t08_context_delta.csv` 存在；CI 基于同一 test rows；报告按 CI 含 0 / 不含 0 写。
- [ ] Table 7 主表无 B2；B2 只在 appendix/footnote，且明确 failed specification。
- [ ] `t11_negative_controls.csv` 有 `role` 列；`t11_placebo_exposure.csv` 存在；§4.6 改名为 diagnostics。
- [ ] TMA 仍未进入主文；若做只进 appendix。

### 报告/交付

- [ ] `Report.pdf` 编译；main text 6–10 页；开头注明页数口径。
- [ ] 没有被驳回数字；0.7747 带目标/特征限定。
- [ ] Disclosures 仍在；References 无伪造。
- [ ] Poster 无文字溢出；有 trajectory/PCA（如果做）。
- [ ] README 使用 `pip install -r REQUIREMENTS.lock`；有复现清单。
- [ ] ZIP 只有 `Group_09_Project.zip`；无 raw data；含 Report/README/code/DATA_DOWNLOAD/STATUS.final。
- [ ] `python3 scripts/validate_run.py --run-id <new>` 0 errors / 0 warnings。
- [ ] `pytest -q` 全过（如果加了 tests）。

---

## 10. 最终交付物

- 新 `runs/run-<new_id>/`：`run.jsonl`、`stages/*.stage.json`、`handoff/*.md`、`manifest*.json`、`STATUS.final.json`。
- `outputs/tables/`：新增 `t08_context_metrics.csv`、`t08_context_delta.csv`、`t08_module_metrics.csv`、`t08_subgroup_metrics.csv`、`t09_b2_failed.csv`、`t11_*_sensitivity.csv`、`t11_placebo_exposure.csv`、`t12_*`、`t13_*`（如果做）。
- `outputs/figures/`：新增 `f07_pca_scatter.png`、`f07_silhouette.png`、`f08_context_comparison.png`、`f08_module_heatmap.png`、`f08_subgroup_calibration.png`、`f12_trajectory_divergence.png`。
- `Report.pdf`、`Report.tex`、`Poster.pdf`、`README.md`、`DATA_DOWNLOAD.md`、`Group_09_Project.zip`。
- `FIX_REPORT.md`：逐条修复记录 + 新 run_id + 验证输出。

---

## 11. 一句话

**先把“未来信息/失效模型/假规格”修干净，再用轨迹分化、context baseline、子组校准把“same engagement, different outcomes”讲实；WP5 把 context 增量、B2、controls 的解释边界钉死；TMA 那三条只做 appendix。**

