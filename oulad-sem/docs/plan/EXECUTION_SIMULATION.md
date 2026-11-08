# OULAD 项目实施执行仿真 Runbook

> **交付对象**：接手的执行 agent。本文档是操作契约，不是建议清单。
> **关联方案**：`01_plan/oulad-deep-mining-plan.html`
> **版本**：1.1 · 2026-11-03（加入 deep-research 复跑证据）
> **模式**：`simulation`（500 人选样子集，端到端演练）→ `full`（全量 32,593 条选课记录）

---

## 0. 执行 agent 的硬规则

1. **一次只做一个阶段。** 读 `STATUS.json`，只执行 `next_stage`；不要并行跑有依赖的阶段。
2. **不改配置。** `config/config.yaml` 是冻结的。要改参数，新建 `CHANGE_REQUEST.md` 和新的 `run_id`，不要在旧 run 上改。
3. **不覆盖已有 run。** `runs/<run_id>/` 是只追加的审计记录；任何重跑都产生新 run_id。
4. **每个阶段必须留下三样东西**：`run.jsonl` 事件、`stages/Sxx.stage.json` 摘要、`STATUS.json` 更新。
5. **失败就停。** 写 `blocked` 和错误事件，不要绕过 gate，不要“先跑后面的看看”。
6. **日志里不写原始学生行。** 只写聚合、计数、哈希、指标；`id_student` 只以哈希或计数形式出现。
7. **任何结论必须能追溯**：`log seq` → `artifact path` → `sha256` → 生成脚本 → `config_hash`。

---

## 0.1 已核验证据更新（2026-11-03 复跑）

执行 agent 不需要重读文献，但必须知道下面这些数字是“可引用”还是“只能当 sanity check”。完整表见 `oulad-deep-mining-plan.html` 第 01 节。

| 事实 | 数字 | 用法 |
|---|---|---|
| `studentVle.date` 相对开课日，含负值 | 范围 −25..269；688,988 行（6.47%）为负 | S04 必须显式处理 pre-start |
| 行级划分的学生泄漏 | 852 人 / 15.7% 测试记录 | S05 硬断言 `student_overlap=0` |
| 时间外推衰减 | 2013→2014 AUC 0.7470（0.7393–0.7549）；LOMO 0.6426–0.7881，median 0.6990 | sanity check，不是目标值 |
| day-28 基线（目标 Fail-or-Withdrawn，含早期分数） | 0.7747（0.7624–0.7870）；去分数 0.7381；纯行为 0.7027 | 引用时必须带目标/特征限定 |
| LEAP 周级曲线（预印本） | week1 0.7151 → week2 0.7404 → week3 0.7851 → week4 0.8055 → week8 0.8602 | S09 的增量量级参考 |
| 时间泄漏幅度 | t=7 严格 0.7151 vs 含泄漏 0.9669（≈0.25 AUC） | S04 异常高的性能先查泄漏 |
| time-on-task 估计不稳定 | 15 种策略使 R² 相差 0.15（0.07–0.23） | 不构造 time-on-task 特征 |
| self-report vs trace | 反应风格解释问卷方差 43%，trace 4% | 不把点击直接叫“投入” |
| 课程情境混杂 | 课程特异 vs 合并模型显著不同（n=4,134） | S06 匹配必须含课程层协变量 |

**不要引用**：`0.58–0.75` 的“0.9+ 崩塌”、week-4 dropout 的 `0.626/0.663/0.636`、全课程 LSTM 周级 `69.69%→95.23%`、“assessment 点击是已证实泄漏通道”、time-on-task 系数反号区间、“每 15 分钟只有 6 分钟在任务上”。这些在复跑中被 0–3/1–2 驳回或仍未核验。

---

## 1. 环境与目录

### 1.1 环境

- Python 3.10+
- 内存：全量模式建议 ≥ 16 GB；磁盘 ≥ 3 GB
- 依赖：`pandas`、`numpy`、`scikit-learn`、`scipy`、`pyyaml`、`jsonschema`、`pyarrow`、`matplotlib`、`seaborn`、`statsmodels`；生存分析可选 `lifelines`
- 固定随机性：`PYTHONHASHSEED=0`，所有模型 `random_state` 来自 config；关键结果用 `n_jobs=1` 复跑一次校验
- 环境记录：`python --version`、`pip freeze`、`uname -a` 写入 `runs/<run_id>/env.json`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn scipy pyyaml jsonschema pyarrow matplotlib seaborn statsmodels lifelines
pip freeze > REQUIREMENTS.lock
```

### 1.2 目录树

```text
oulad-sem/
├── config/
│   ├── config.yaml
│   └── config.sha256
├── data/
│   ├── raw/oulad/*.csv
│   ├── interim/master.parquet
│   ├── interim/sim_sample.parquet
│   └── processed/features_c28.parquet
├── src/
│   ├── io.py
│   ├── logging_utils.py
│   ├── cohort.py
│   ├── windows.py
│   ├── features.py
│   ├── splits.py
│   ├── matching.py
│   ├── clustering.py
│   ├── modeling.py
│   ├── sequential.py
│   ├── survival.py
│   └── validate.py
├── stages/
│   ├── s00_acquire.py
│   ├── s01_raw_audit.py
│   ├── s02_master.py
│   ├── s03_eda.py
│   ├── s04_features.py
│   ├── s05_splits.py
│   ├── s06_matching.py
│   ├── s07_clustering.py
│   ├── s08_models.py
│   ├── s09_sequential.py
│   ├── s10_survival.py
│   ├── s11_robustness.py
│   └── s12_package.py
├── scripts/
│   ├── run_stage.py
│   ├── validate_run.py
│   └── make_report.py
├── runs/
│   └── run-YYYYMMDDTHHMMSSZ-xxxxxx/
│       ├── run.jsonl
│       ├── env.json
│       ├── manifest.json
│       ├── stages/S00.stage.json
│       ├── stdout/S00.stdout.log
│       ├── stderr/S00.stderr.log
│       ├── metrics/
│       └── artifacts/
├── outputs/
│   ├── tables/
│   └── figures/
├── STATUS.json
└── REQUIREMENTS.lock
```

### 1.3 数据获取

优先使用用户已放置的 zip；没有才下载。下载后**先记录哈希，再解压**。

```bash
mkdir -p data/raw/oulad
# 方式 A：用户提供
# cp /path/to/oulad.zip data/raw/oulad/oulad.zip
# 方式 B：官方 UCI
curl -L --fail --retry 3 \
  -o data/raw/oulad/oulad.zip \
  https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip

shasum -a 256 data/raw/oulad/oulad.zip | tee data/raw/oulad/oulad.zip.sha256
unzip -o data/raw/oulad/oulad.zip -d data/raw/oulad/
ls -lh data/raw/oulad/
```

必须存在 7 个 CSV：`courses.csv`、`assessments.csv`、`vle.csv`、`studentInfo.csv`、`studentRegistration.csv`、`studentAssessment.csv`、`studentVle.csv`。

**不要硬编码下载文件的 SHA256**；记录实际值即可。行数参考官方统计：`studentInfo` 32,593；`studentRegistration` 32,593；`courses` 22；`assessments` 206；`vle` 6,364；`studentAssessment` 173,912；`studentVle` 约 10.65M。若 `studentInfo` 不是 32,593，直接停，先查数据版本。

### 1.4 全局命名

- `run_id`：`run-YYYYMMDDTHHMMSSZ-<6位随机hex>`；仿真前缀 `sim-`。
- 阶段：`S00`–`S12`。
- 表格：`outputs/tables/t<stage>_<name>.csv`。
- 图：`outputs/figures/f<stage>_<name>.png`。
- 模型/预测：`runs/<run_id>/artifacts/`。
- 所有路径写入 `manifest.json`，相对项目根目录。

---

## 2. 配置契约

`config/config.yaml` 冻结后计算 `sha256` 写入 `config/config.sha256`。每个日志事件都带 `config_hash`。

```yaml
project: oulad-sem
seed: 42
mode: full                     # simulation | full

paths:
  raw_dir: data/raw/oulad
  interim_dir: data/interim
  processed_dir: data/processed
  outputs_dir: outputs
  runs_dir: runs

cohort:
  target: T1                   # T1 | T2 | T3
  exclude_withdrawn: true      # T1
  at_risk_at_cutoff: true      # 剔除 date_unregistration <= cutoff
  min_w1_clicks: 0             # 0 表示保留零点击；敏感性用 5/10/20
  require_final_result: true

windows:
  day0: 0
  w1_end: 27                   # 0..27
  w2_end: 55                   # 28..55；若 first_tma 更早则用 first_tma-1
  use_first_assessment: true
  first_assessment_types: [TMA, CMA]
  pre_start: exclude           # exclude | clip | feature
  cutoffs: [14, 21, 28, 35, 42, 49, 56]

leakage:
  include_scores: false        # 严格主分析 false
  include_quiz_sites: false
  exclude_banked: true
  assert_no_future: true
  assert_group_split: true

features:
  families: [volume, composition, breadth, regularity, timing, change]
  log1p: true
  winsorize: [0.01, 0.99]
  standardize: within_presentation
  min_active_days_for_clustering: 1

splits:
  group_key: id_student
  test_size: 0.2
  val_size: 0.2
  n_splits: 5
  temporal_train: [2013B, 2013J]
  temporal_test: [2014B, 2014J]

matching:
  method: cem
  exact: [code_module, code_presentation]
  click_bins: 10
  active_day_bins: [0, 1, 4, 8, 15, 22, 29]
  entropy_bins: 3
  smd_threshold: 0.1

clustering:
  k_range: [2, 8]
  n_init: 50
  bootstrap: 300
  ari_threshold: 0.60

models:
  ladder: [M0, M1, M2, M3, M4, M5]
  include_scores_model: M4   # 只作上界
  primary_metric: pr_auc
  recall_precision_target: 0.80
  calibration_bins: 10

logging:
  level: INFO
  jsonl: true
  flush_each_event: true
```

**配置变更流程**：写 `CHANGE_REQUEST.md`（原因、影响阶段、新参数、预期差异），新建 run，不覆盖旧 run。

---

## 3. 日志协议

### 3.1 日志层级

每次 run 一个目录，三层日志：

| 文件 | 内容 | 写入者 |
|---|---|---|
| `runs/<run_id>/run.jsonl` | 结构化事件流，一行一个 JSON | `logging_utils.log_event()` |
| `runs/<run_id>/stages/Sxx.stage.json` | 阶段摘要：状态、检查、输入输出、指标 | 阶段脚本 |
| `runs/<run_id>/stdout/Sxx.stdout.log`、`stderr/Sxx.stderr.log` | 原始输出，供调试 | orchestrator 重定向 |

`run.jsonl` 只追加；阶段失败也不删。重跑必须新 run_id。

### 3.2 事件字段

```json
{
  "schema_version": "1.0",
  "seq": 1,
  "ts": "2026-11-03T09:00:00.123Z",
  "run_id": "run-20261103T090000Z-a1b2c3",
  "mode": "full",
  "stage": "S04",
  "step": "build_w1_features",
  "event": "step_start",
  "status": "ok",
  "agent": "executor",
  "config_hash": "sha256:...",
  "git_commit": "no_git",
  "duration_ms": 0,
  "inputs": [],
  "outputs": [],
  "metrics": {},
  "message": "start W1 feature build",
  "error": null,
  "simulated": false
}
```

必填：`schema_version`、`seq`、`ts`、`run_id`、`mode`、`stage`、`event`、`status`、`config_hash`。其余按事件类型填。

### 3.3 事件类型

| `event` | 何时写 | 关键字段 |
|---|---|---|
| `stage_start` | 阶段开始 | `inputs` |
| `step_start` / `step_end` | 阶段内步骤 | `step`、`duration_ms` |
| `check` | 断言/验收 | `message`、`metrics`、`status` |
| `metric` | 产生指标 | `metrics` |
| `artifact` | 写出文件 | `outputs[].path`、`sha256`、`rows` |
| `warning` | 可继续的问题 | `message` |
| `error` | 失败 | `error.type`、`message`、`traceback_ref` |
| `gate_pass` / `gate_fail` | 阶段验收 | `metrics`、`message` |
| `stage_end` | 阶段结束 | `status`、`duration_ms` |

### 3.4 日志示例（示意，不是真实结果）

```json
{"schema_version":"1.0","seq":41,"ts":"2026-11-03T09:04:12.001Z","run_id":"sim-20261103T090000Z-a1b2c3","mode":"simulation","stage":"S04","step":"build_w1_features","event":"step_start","status":"ok","agent":"executor","config_hash":"sha256:9f2c...","git_commit":"no_git","duration_ms":0,"inputs":[{"path":"data/interim/master.parquet","sha256":"sha256:1a2b...","rows":32593}],"outputs":[],"metrics":{},"message":"cutoff=28, pre_start=exclude","error":null,"simulated":true}
{"schema_version":"1.0","seq":42,"ts":"2026-11-03T09:04:18.450Z","run_id":"sim-20261103T090000Z-a1b2c3","mode":"simulation","stage":"S04","step":"build_w1_features","event":"metric","status":"ok","agent":"executor","config_hash":"sha256:9f2c...","git_commit":"no_git","duration_ms":6450,"inputs":[],"outputs":[],"metrics":{"n_rows":32593,"n_features_w1":23,"zero_click_share":0.041,"median_clicks":214},"message":"W1 features built","error":null,"simulated":true}
{"schema_version":"1.0","seq":43,"ts":"2026-11-03T09:04:18.501Z","run_id":"sim-20261103T090000Z-a1b2c3","mode":"simulation","stage":"S04","step":"leakage_assert","event":"check","status":"ok","agent":"executor","config_hash":"sha256:9f2c...","git_commit":"no_git","duration_ms":12,"inputs":[],"outputs":[],"metrics":{"max_date":27,"score_columns":0,"student_overlap":0},"message":"all leakage assertions passed","error":null,"simulated":true}
{"schema_version":"1.0","seq":44,"ts":"2026-11-03T09:04:19.000Z","run_id":"sim-20261103T090000Z-a1b2c3","mode":"simulation","stage":"S04","step":"write_features","event":"artifact","status":"ok","agent":"executor","config_hash":"sha256:9f2c...","git_commit":"no_git","duration_ms":490,"inputs":[],"outputs":[{"path":"data/processed/features_c28.parquet","sha256":"sha256:7d4e...","rows":32593,"cols":31}],"metrics":{},"message":"features_c28 written","error":null,"simulated":true}
```

**这些数字是仿真占位符**，执行 agent 必须用真实输出替换，并保持 `simulated` 与 `mode` 一致。

### 3.5 日志规则

- UTF-8，每行一个 JSON，行尾 `\n`；不要 pretty-print。
- `seq` 在 run 内严格递增；并发写用单进程串行。
- `ts` 用 UTC ISO8601，毫秒精度。
- 每个 artifact 写完立刻算 `sha256`；大文件按块读。
- `run.jsonl` 每次 `flush()`；阶段结束时 `fsync`。
- 不写原始学生记录；错误 traceback 只进 `stderr/Sxx.stderr.log`，JSONL 里放路径。
- 日志写入失败 = 阶段失败，不能继续。
- `stdout` / `stderr` 用 orchestrator 的 `tee` 捕获，不要脚本自己 print 大表。

### 3.6 日志校验命令

```bash
RUN=runs/run-20261103T090000Z-a1b2c3

# 事件总数与各阶段事件数
jq -r '[.stage,.event] | @tsv' "$RUN/run.jsonl" | sort | uniq -c

# 任何 fail 事件
jq -c 'select(.status=="fail" or .event=="error")' "$RUN/run.jsonl"

# 每个阶段是否都有 start/end
jq -r '.stage' "$RUN/run.jsonl" | sort | uniq -c

# 所有 artifact 是否存在且哈希匹配
jq -r 'select(.event=="artifact") | .outputs[] | [.path,.sha256] | @tsv' "$RUN/run.jsonl" |
while IFS=$'\t' read -r p h; do
  test -f "$p" || echo "MISSING $p"
  test "$(shasum -a 256 "$p" | cut -d' ' -f1)" = "${h#sha256:}" || echo "HASH MISMATCH $p"
done
```

---

## 4. 交接协议

### 4.1 `STATUS.json`

```json
{
  "schema_version": "1.0",
  "run_id": "run-20261103T090000Z-a1b2c3",
  "mode": "full",
  "updated_at": "2026-11-03T09:20:00.000Z",
  "current_stage": "S05",
  "next_stage": "S06",
  "stages": {
    "S00": {"status": "pass", "finished_at": "...", "log": "runs/.../run.jsonl", "stage_json": "runs/.../stages/S00.stage.json", "artifacts": ["data/raw/oulad/studentInfo.csv"]},
    "S01": {"status": "pass", "finished_at": "...", "log": "...", "stage_json": "...", "artifacts": ["outputs/tables/t01_raw_audit.csv"]}
  },
  "blocked": null,
  "notes": "S05 passed; S06 not started"
}
```

更新方式：写 `STATUS.json.tmp` → `fsync` → `mv` 覆盖。不要直接原地改。

### 4.2 阶段 gate

一个阶段只有同时满足以下条件才能 `pass`：

1. 所有声明的输出文件存在，`sha256` 已记录。
2. 阶段内所有 `check` 事件为 `ok` 或显式 `warn`（warn 必须写原因）。
3. 没有 `error` / `status=fail` 事件。
4. `stages/Sxx.stage.json` 已写。
5. `STATUS.json` 已更新。

否则 `gate_fail`，`blocked` 写原因，停止。

### 4.3 交接 note

每个阶段结束写 `runs/<run_id>/handoff/Sxx.md`：

```markdown
# S06 handoff
- run_id:
- status: pass | fail | warn
- outputs: path (rows, sha256)
- checks: 通过/失败项
- metrics: 关键数字
- open issues: 需要下一阶段注意的
- next stage must not: 例如“不要改 cutoff；不要用 S06 的匹配结果做因果表述”
```

### 4.4 Blocked 与变更

- 数据/环境问题：`blocked.reason = "E_DATA: ..."`，附日志 seq。
- 需要改参数：不要改 `config.yaml`；写 `CHANGE_REQUEST.md`，等新 run。
- 需要跳过某阶段：不允许；只能标记 `skip` 并写明为什么该阶段对该 run 不适用。

---

## 5. 阶段执行手册

### S00 — 环境与数据获取

**目标**：可复现的环境 + 原始数据 + 哈希清单。

**操作**

```bash
python scripts/run_stage.py --stage S00 --mode full --config config/config.yaml
```

1. 检查 Python 版本、依赖、磁盘空间。
2. 若 `data/raw/oulad/*.csv` 不存在，按 1.3 下载/解压。
3. 计算 7 个 CSV 的 `sha256`、行数、列名，写 `runs/<run_id>/manifest.json`。
4. 写 `env.json`：Python、平台、包版本、`PYTHONHASHSEED`。
5. 记录 `config/config.sha256`。

**输出**：`manifest.json`、`env.json`、`data/raw/oulad/*.csv`。

**验收**

- 7 个 CSV 存在。
- `studentInfo` 行数 = 32,593。
- 所有文件 `sha256` 已记录。
- `final_result` 列存在，且至少 4 个非空类别。

**日志样例**

```json
{"schema_version":"1.0","seq":1,"ts":"2026-11-03T09:00:00.000Z","run_id":"run-...","mode":"full","stage":"S00","step":"env_check","event":"check","status":"ok","config_hash":"sha256:...","metrics":{"python":"3.11.8","pandas":"2.2.2","disk_free_gb":120},"message":"environment ok","simulated":false}
{"schema_version":"1.0","seq":2,"ts":"2026-11-03T09:00:05.000Z","run_id":"run-...","mode":"full","stage":"S00","step":"raw_manifest","event":"artifact","status":"ok","outputs":[{"path":"runs/run-.../manifest.json","sha256":"sha256:..."}],"metrics":{"files":7,"studentinfo_rows":32593},"message":"raw manifest written","simulated":false}
```

**失败处理**：缺文件 → `E_DATA`；下载失败 → 重试 3 次后 `E_RESOURCE`；行数不符 → 停，查数据版本。

---

### S01 — 原始数据审计

**目标**：确认表结构、键唯一性、日期范围、标签分布、活动类型分布。

**操作**

```bash
python scripts/run_stage.py --stage S01 --mode full --config config/config.yaml
```

1. 逐表读入，记录 shape、dtype、缺失率、重复行。
2. 检查 `studentInfo` 主键 `(code_module, code_presentation, id_student)` 唯一。
3. 统计 `final_result` 四类计数。
4. 统计 `studentVle.date` 的 min/max、负日期占比。
5. 统计 `vle.activity_type` 的取值。
6. 统计 `studentAssessment.is_banked` 分布。
7. 写 `outputs/tables/t01_raw_audit.csv`、`t01_final_result_counts.csv`、图 `f01_click_distribution.png`。

**验收**

- 主键唯一。
- `final_result` 只包含 `Pass`、`Fail`、`Withdrawn`、`Distinction`（顺序可变）。
- `studentVle.date` 的最小值 < 0（记录具体值，不要求固定）。
- 7 个表都能读入，无列名错位。

**日志要点**：`metrics` 里记录 `n_rows`、`n_unique_students`、`final_result_counts`、`date_min`、`date_max`、`negative_date_share`、`n_activity_types`。

**失败处理**：主键重复 → `E_DATA`；`final_result` 出现第五类 → 停，查版本。

---

### S02 — 主表与队列

**目标**：生成 `data/interim/master.parquet`，每个选课记录一行，带标签、退课日期、课程长度、first_tma。

**操作**

1. `studentInfo` LEFT JOIN `studentRegistration`，再 JOIN `courses`。
2. 计算 `first_tma_date`：`assessments` 中 `assessment_type ∈ {TMA, CMA}` 且 `weight > 0` 的最小 `date`，按 `(code_module, code_presentation)`。
3. 计算 `is_withdrawn = final_result == "Withdrawn"`。
4. 计算 `withdrawn_before(c)`：`date_unregistration` 非空且 `≤ c`。
5. 计算 cohort 标记：`in_full_cohort = True`；`in_at_risk_cohort(c) = not withdrawn_before(c)`。
6. 写 `master.parquet`，并输出 `outputs/tables/t02_withdrawal_audit.csv`。

**先审计 Withdrawn，不要直接写剔除规则**

本轮关于 `date_unregistration` 与“退课后点击流被截断”的捆绑断言被 0–3 驳回。S02 必须先输出以下审计表，再决定 `at-risk` 队列：

- `date_unregistration` 非空/空的比例，按 presentation；
- 退课者在 `date_unregistration` 前后各 4 周的周点击分布；
- 退课者是否仍有 `date > date_unregistration` 的点击（按 module presentation 统计）；
- 若存在退课后点击，说明“退课事件”与“点击流停止”不是同一件事。

只有当审计支持“退课后没有新点击/结果已定”时，才用 `date_unregistration ≤ c` 剔除；否则保留全队列，把退课状态放进 T3/竞争风险，而不是在 S02 静默过滤。

**验收**

- 行数 = 32,593。
- 主键唯一。
- 每个 presentation 的 `first_tma_date` 要么是整数，要么为 `NaT` 并记录数量。
- `final_result` 无缺失。
- 退课审计表已生成；`date_unregistration` 非空比例已记录（不预设 100%）。

**日志要点**：`join_loss`、`n_withdrawn`、`n_repeaters`、`first_tma_missing`、`cohort_counts`、`withdrawn_after_unregistration_clicks`。

---

### S03 — 描述性审计与重叠检查

**目标**：回答“相似点击”是否真的存在，决定匹配是否有共同支撑。

**操作**

1. 每个 cutoff，按 at-risk 队列计算 W1 点击、活跃天数。
2. 按 `final_result` 分组，输出中位数、IQR、零点击比例。
3. 在同一 presentation 内，比较 Pass/Distinction 与 Fail 的 W1 点击分布；计算重叠系数（histogram intersection）或简单的分位数重叠。
4. 画 `f03_click_distribution_by_outcome.png`、`f03_withdrawal_timing.png`。
5. 写 `t03_cohort_flow.csv`、`t03_overlap.csv`。

**验收**

- 每个 cutoff 的 at-risk 人数、base rate 都有记录。
- 重叠统计有数值；如果重叠很低，写 `warning`，但阶段可 pass（这是发现，不是错误）。
- 零点击比例被记录；不静默丢弃。

**日志要点**：`cohort_n`、`base_rate`、`overlap_coefficient`、`zero_click_share`、`withdrawn_by_w1`、`withdrawn_by_w2`。

---

### S04 — 特征构造

**目标**：对每个 cutoff 生成 W1/W2/Δ 特征，严格隔离未来信息。

**函数契约**

```python
def build_features(master: pd.DataFrame, vle: pd.DataFrame, assessments: pd.DataFrame,
                   cutoff: int, config: dict) -> pd.DataFrame:
    """返回每个 (code_module, code_presentation, id_student) 一行的特征表。
    只使用 date < cutoff 的点击；pre_start 按 config 处理；
    严格模式下不读取 score / is_banked。
    """
```

**操作**

1. 按 `(code_module, code_presentation)` 分组，保证日期是相对日。
2. 过滤 `date < cutoff`；按 `pre_start` 处理负日期。
3. 聚合到日 × 活动类型：`sum_click`。
4. 构造六族特征（见方案 2.5）。
5. W2 若 `cutoff ≥ first_tma_date`，使用 `28..first_tma_date-1`；否则用 `28..min(55, cutoff-1)`。
6. 写 `data/processed/features_c<cutoff>.parquet`。
7. 跑单元测试：
   - `date=27` 在 cutoff 28 时被包含，`date=28` 不被包含。
   - 负日期按配置处理。
   - 抽 3 个学生，手工核对总点击。
8. 跑泄漏断言（见下）。

**验收**

- `max(feature_date) < cutoff`。
- 严格特征列中没有 `score`、`is_banked`、`final_result`。
- 同一配置、同一种子跑两次，特征表哈希一致。
- 行数 = 当前队列人数；无重复主键。
- 泄漏 sanity check：严格协议的 day-28 ROC-AUC 若 > 0.90，或比同特征集的 group CV 高 0.15 以上，先触发 `E_LEAK` 审计。参照：LEAP t=7 严格 0.7151 vs 含泄漏 0.9669；OULAD day-28 的合理量级约 0.70–0.81（目标与特征不同，仅作 sanity check）。
- 负日期处理有计数：`n_negative_date_rows`、`negative_date_share`、`pre_start` 策略。

**日志要点**：`n_features_w1`、`n_features_w2`、`n_features_delta`、`zero_click_share`、`max_date`、`n_negative_date_rows`、`assertions`。

**失败处理**：断言失败 → `E_LEAK`，停止；不要用“下一阶段再修”。

---

### S05 — 划分

**目标**：生成互不重叠、可复现的训练/验证/测试索引。

**操作**

1. `GroupShuffleSplit` by `id_student`：60/20/20。
2. 5 折 `GroupKFold`。
3. 时间外推：`2013B/2013J` 训练，`2014B/2014J` 测试。
4. leave-one-module-out：7 折。
5. 写 `splits.json`：每折的索引、学生集合哈希、类别分布。

**验收**

- 每个 split 的训练/测试 `id_student` 交集 = 0。
- GroupKFold 每折覆盖所有行且不重复。
- 时间外推的测试 presentation 不出现在训练。
- 类别分布在每个 split 都有记录。
- 输出 `student_overlap` 对照：如果不按学生分组，预计约 852 人 / 15.7% 的测试记录会重叠（Wu et al. 2026）。S05 的 `student_overlap` 必须为 0，并在日志里记录“未分组时的重叠人数”。
- 时间外推 sanity check：2013→2014 AUC 约 0.7470（0.7393–0.7549）；LOMO 0.6426–0.7881（median 0.6990）。若明显高于区间，先查泄漏；这些数字来自 all-features 模型，不是纯 clickstream 目标值。

**日志要点**：`split`、`n_train`、`n_val`、`n_test`、`student_overlap`、`ungrouped_overlap_n`、`positive_rate`。

---

### S06 — 匹配

**目标**：在同一 `module × presentation` 内构造“W1 参与度可比”的层，检查平衡，找不一致层。

**操作**

1. 用 S04 的 `features_c28`（或 config 指定 cutoff）。
2. 粗化：点击十分位、活跃天数分箱、活动熵三分位。
3. CEM：精确层 `code_module × code_presentation`；写匹配层 id。
4. 计算每个匹配变量的 SMD，输出 `t06_balance.csv`；画 Love plot。
5. 写匹配层表 `runs/<run_id>/artifacts/matched_strata.parquet`。
6. 不一致层：同层内同时有 Pass/Distinction 与 Fail 的层，输出 `t06_discordant_strata.csv`。
7. 在层内比较 W2 特征，给 bootstrap CI；写 `t06_w2_differences.csv`。
8. 计算 E-value（可选，但建议）。

**验收**

- 所有匹配变量 `|SMD| < 0.1`。
- 报告未匹配数、样本损失、共同支撑。
- 匹配变量没有 cutoff 后的字段。
- 不一致层的样本量 > 0；如果为 0，写 `warning`（说明“相似点击但结果不同”在该定义下不存在）。

**日志要点**：`n_matched`、`n_unmatched`、`max_smd`、`n_strata`、`n_discordant`、`att_like_delta`。

---

### S07 — 聚类与稳定性

**目标**：得到可重复的行为 profile，并量化其不稳定程度。

**操作**

1. 取 W1 特征，按 presentation 内 z-score；PCA 可选，记录解释方差。
2. k = 2…8；每个 k 用 `n_init=50`、`k-means++`。
3. 计算 silhouette、Calinski–Harabasz、Davies–Bouldin。
4. bootstrap / 子采样 300 次：每次重跑 k-means，计算 ARI 与每簇 Jaccard；写 `t07_stability.csv`。
5. 选 k：优先稳定性，其次 silhouette；项目预设门槛为最大平均 ARI ≥ 0.60（方法学共识，不是文献定值）。低于门槛标记“不稳定”，降低结论强度。
6. 画 consensus matrix、ARI 分布、簇 × 结果。
7. 对 W2−W1 变化向量重复一次（可选）。

**证据边界**：被评述的 EM 研究只报 BIC 与 MANOVA/ANOVA+Bonferroni，未报 bootstrap、ARI 或 silhouette；这证明“实务常缺”，不证明“0.60 是文献阈值”。报告里写“本项目预设”。

**验收**

- 每个 k 的稳定性指标都有记录。
- 选定 k 的理由写入 `stages/S07.stage.json`。
- 不稳定簇已合并/放弃，并记录决定。
- 簇 × 结果表有人数与比例。

**日志要点**：`k_chosen`、`mean_ari`、`min_cluster_jaccard`、`cluster_sizes`、`outcome_rates_by_cluster`。

---

### S08 — 模型阶梯

**目标**：在统一划分上比较 M0–M5，回答“行为结构是否有增量”。

**操作**

1. 用 `sklearn.pipeline.Pipeline`，所有预处理在 fold 内 fit。
2. 对每个模型、每个 split：
   - 训练、验证、测试；
   - 计算 PR-AUC、ROC-AUC、recall@precision=0.8、F1（验证集选阈值）、Brier、校准斜率/截距；
   - 1000 次 bootstrap CI。
3. M4 只作上界：明确标注“含评估分数，不可用于早期预警”。
4. 保存预测概率 `predictions.parquet`；保存模型卡 `model_card.json`。
5. 画校准曲线、PR 曲线、permutation importance。

**验收**

- 所有指标来自 held-out，不在训练集上报。
- 阈值在验证集选，测试集只用一次。
- 每个模型都有校准指标。
- 预测表包含 `id_student`、`presentation`、`y_true`、`y_prob`、`split`。
- 没有使用 S06 匹配结果来选模型（匹配是描述性分析，不参与模型选择）。

**日志要点**：`model`、`split`、`pr_auc`、`roc_auc`、`recall_at_precision`、`f1`、`brier`、`calibration_slope`、`n_train`、`n_test`。

---

### S09 — 分阶段与先验/后验

**目标**：回答“多看到 W2，预警提升多少”，并实现两种更新方式。

**操作**

1. 对每个 cutoff 训练 M0–M3；固定队列与 at-risk 队列各跑一遍。
2. 画 `f09_cutoff_curve.png`：PR-AUC / recall vs cutoff。
3. B1 stacking：用 cross-fitted 的 W1 预测概率作为 W2 模型特征；确保 W1 概率在训练折外生成。
4. B2 offset：`logit(p_W1)` 作为 offset，拟合 W2 特征。
5. 计算增量：`ΔPR-AUC`、`Δrecall@precision`、校准变化，bootstrap CI。
6. 写 `t09_incremental.csv`。

**参照量级（LEAP 预印本，目标 Pass/Distinction vs Fail/Withdrawn）**：week1 0.7151 → week2 0.7404 → week3 0.7851 → week4 0.8055 → week8 0.8602；2→3 周 +0.0447。你的目标/特征不同，不能直接对齐；但可以把“4→8 周增量约 +0.05 AUC”作为 sanity check。若增量是 +0.30，先查泄漏或队列变化。

**验收**

- 每个 cutoff 的队列定义一致（固定队列 vs at-risk 分开报）。
- cross-fitting 的折与最终评估折不重叠。
- 增量有 CI，不只报点估计。
- 如果 B1/B2 没有增量，如实写“没有”。

**日志要点**：`cutoff`、`cohort`、`pr_auc`、`delta_pr_auc`、`ci_low`、`ci_high`、`method`。

---

### S10 — 竞争风险

**目标**：把 Withdrawn 作为独立结果，而不是折叠进 Fail。

**操作**

1. 在每个 cutoff 做三分类 multinomial logistic（或 HistGradientBoosting）。
2. 以周为时间单位，构造 `(student, week, event)` 长表；事件 = Pass / Fail / Withdrawn / censored。
3. 拟合 cause-specific Cox（`lifelines.CoxPHFitter` 或 `statsmodels`）；若不可用，至少做三分类。
4. 计算 cumulative incidence（Fine–Gray 可选）。
5. 写 `t10_class_metrics.csv`、`t10_hazard_ratios.csv`、`f10_cumulative_incidence.png`。

**验收**

- 三类样本量都报告。
- 时间原点、censoring 定义写清楚。
- 没有把 Withdrawn 并入 Fail。
- 若 survival 依赖不可用，阶段标记 `warn` 但仍输出三分类结果。

**日志要点**：`n_pass`、`n_fail`、`n_withdrawn`、`concordance`、`class_metrics`。

---

### S11 — 稳健性与规格曲线

**目标**：把所有合理选择摊开，检验结论是否脆弱。

**操作**

1. 定义规格网格：
   - cutoff：7 / 28 / first_tma
   - pre_start：exclude / clip / feature
   - min_w1_clicks：0 / 5 / 10 / 20
   - target：T1 / T2 / T3
   - features：M0 / M1 / M2 / M3
   - matching：CEM / PSM / stratified
   - clustering k：2…6
   - model：logistic / RF
2. 对每个规格跑同一评估流程；记录性能或效应量。
3. 画 `f11_specification_curve.png`：按估计值排序。
4. 负对照：
   - 仅用开课前点击预测结果；
   - 随机打乱标签；
   - 安慰剂 cutoff（用未来窗口做同样分析）。
5. 写 `t11_specs.csv`、`t11_negative_controls.csv`。

**验收**

- 每个规格有 `config_hash` 或规格 id。
- 报告中位数、范围、方向一致比例。
- 负对照接近 baseline；否则查泄漏。

**日志要点**：`n_specs`、`median_metric`、`min_metric`、`max_metric`、`share_positive`、`negative_control_metric`。

---

### S12 — 打包与报告

**目标**：把可复现的结果整理成课程要求的交付物：Report.pdf、README、Poster.pdf、代码与数据说明。

**操作**

1. 汇总 `outputs/tables/`、`outputs/figures/`。
2. 生成 `manifest.json`：每个文件 path、sha256、rows、生成阶段。
3. 生成 `REPORT.md` 骨架，按课程 guide 的六节组织：
   - Title & Group Members
   - 1. Introduction
   - 2. Problem Formulation
   - 3. Methodology（预处理、窗口、特征、模型、评估）
   - 4. Experiments（数据 schema、划分、baseline、指标、参数、结果）
   - 5. Conclusion and Limitations
   - 6. Disclosures（外部协作与 AIGC 使用）
   - References（只列读过的；不编造）
   - Appendix（可选）
4. 生成 `README.md`：依赖（`REQUIREMENTS.lock`）、数据获取/解压、预处理命令、逐阶段运行命令、样本数据说明、已知不能复现的部分。
5. 生成 `Poster.pdf`（或 poster 源文件），复用 FIG 01–09；Poster 单独提交，不放进最终 ZIP。
6. 打包 `Group_XX_Project.zip`：Report.pdf、code、README、数据下载说明；不包含原始全量数据，除非课程允许。
7. 运行 `scripts/validate_run.py --run-id <id>`。
8. 更新 `STATUS.json` 为 `complete`。

**验收**

- `validate_run.py` 全部通过。
- Report 正文 6–10 页（不含 references / disclosures / appendix）；每个数字都有表/图引用。
- README 能让另一个 agent 从零跑通 `simulation` 模式。
- Disclosures 明确写了 AIGC 使用方式；没有伪造引用。
- 未核验/被驳回的数字没有出现在正文；`0.7747` 等已核验数字必须带目标与特征限定（见 9.4 引用分级）。
- ZIP 里没有原始全量数据；有下载说明或样本数据例外说明。

---

## 6. 仿真演练（必须先跑）

### 6.1 目的

在 500 人选样子集上跑通 S00–S12，验证：

- 管道能端到端执行；
- 日志完整、可校验；
- 泄漏断言有效；
- 交接协议可用；
- 全量运行前发现 schema / 内存 / 依赖问题。

### 6.2 采样

`--mode simulation` 时：

1. 按 `(code_module, code_presentation, final_result)` 分层，抽 500 条选课记录；若某层不足，按比例补。
2. 只保留这些学生的 `studentVle` / `studentAssessment` 行。
3. 写 `data/interim/sim_sample.parquet`，记录抽样种子与每层人数。
4. 后续所有阶段只读样本，禁止回读全量。

### 6.3 命令

```bash
export RUN_ID="sim-$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
for S in S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
  python scripts/run_stage.py --stage "$S" --mode simulation --run-id "$RUN_ID" --config config/config.yaml
done
python scripts/validate_run.py --run-id "$RUN_ID"
```

### 6.4 仿真通过标准

| 检查 | 标准 |
|---|---|
| 阶段完成 | S00–S12 全部 `pass` 或显式 `warn` |
| 日志完整 | 每阶段有 `stage_start` 与 `stage_end` |
| 错误 | 无 `status=fail` / `event=error` |
| artifact | 每个声明输出存在且哈希匹配 |
| 泄漏 | S04/S05 断言全过 |
| 交接 | 每阶段有 `handoff/Sxx.md`，`STATUS.json` 连续 |
| 负对照 | S11 的随机标签接近 baseline |

### 6.5 仿真日志量级（用于判断是否“跑完了”）

不是硬指标，但低于下表一半通常意味着有阶段被跳过：

| 阶段 | 事件数（约） |
|---|---|
| S00 | 20–40 |
| S01 | 40–80 |
| S02 | 30–60 |
| S03 | 50–100 |
| S04 | 100–200 |
| S05 | 30–60 |
| S06 | 80–150 |
| S07 | 100–200 |
| S08 | 200–400 |
| S09 | 100–200 |
| S10 | 80–150 |
| S11 | 100–300 |
| S12 | 30–60 |

### 6.6 全量切换

仿真全部通过后，才允许：

```bash
python scripts/run_stage.py --stage S00 --mode full --config config/config.yaml
```

全量 run 使用新的 `run_id`；不要复用 `sim-` run 的目录。

---

## 7. 错误分类与重试

| 类型 | 例子 | 处理 |
|---|---|---|
| `E_ENV` | 缺包、Python 版本不对 | 修环境，重跑当前阶段；记录 `env.json` |
| `E_DATA` | CSV 缺列、主键重复、行数不符 | 停；查数据版本；不要猜 |
| `E_LEAK` | 未来日期、score 进入严格特征、学生重叠 | 停；修管道；不得绕过 |
| `E_METRIC` | 某类样本为 0、指标 NaN | 停；检查队列/标签；报告样本量 |
| `E_RESOURCE` | 内存/磁盘不足 | 降采样或分块；重试一次；记录 |
| `E_LOGIC` | 代码异常、索引错位 | 停；修代码；新 run |
| `E_TIMEOUT` | 阶段超时 | 检查数据规模；不要盲目重试 |

重试规则：`E_RESOURCE` 允许一次同 run 内重试；其他类型修复后必须新 run。任何重试都要在 `run.jsonl` 写 `warning` 或 `error`，不能静默。

---

## 8. 最终交付物清单

- `runs/<run_id>/run.jsonl`：完整事件流。
- `runs/<run_id>/manifest.json`：所有 artifact 的哈希。
- `runs/<run_id>/stages/*.stage.json`：每阶段摘要。
- `runs/<run_id>/handoff/*.md`：阶段交接。
- `STATUS.json`：最终状态。
- `outputs/tables/`：样本流、审计、平衡、稳定性、模型指标、增量、竞争风险、规格曲线。
- `outputs/figures/`：FIG 01–09。
- `REPORT.md`：报告骨架；最终导出 `Report.pdf`（6–10 页正文）。
- `README.md`：依赖、数据获取、预处理、逐阶段命令、样本数据说明。
- `Poster.pdf`：poster 源文件/导出（单独提交）。
- `Group_XX_Project.zip`：课程提交包（Report.pdf + code + README + 数据下载说明）。
- `REQUIREMENTS.lock`：环境快照。

---

## 9. 附录

### 9.1 日志 JSON Schema

见 `02_contracts/log_event.schema.json`。

### 9.2 STATUS JSON Schema

见 `02_contracts/status.schema.json`。

### 9.3 配置模板

见 `02_contracts/config.example.yaml`。

### 9.4 引用分级

**可引用，但必须带限定**（3–0 核验）：

- `0.7747` / `0.7504` / `0.7381` / `0.7027`：Wu et al. (2026)，目标 Fail-or-Withdrawn，22 特征含早期提交与早期均分；不是纯 clickstream、Pass/Distinction vs Fail。
- `852` 人 / `15.7%`：行级划分的学生重叠；`27,553` 是同研究 day-28 仍注册的选课数。
- `0.7470`（0.7393–0.7549）、LOMO `0.6426–0.7881`（median `0.6990`）：all-features 模型的时间/模块外推。
- `0.7151 → 0.8602`：LEAP 预印本周级曲线，目标 Pass/Distinction vs Fail/Withdrawn。
- `0.15`（R² 差，0.07–0.23）：time-on-task 估计策略差异。
- `43%` / `4%`：反应风格解释问卷 vs trace 方差。
- `n=4,134`：Gašević et al. (2016) 九门课程。

**不得引用**（被驳回、未核验或覆盖缺口）：

`0.58–0.75` 的“0.9+ 崩塌”、week-4 dropout 的 `0.626 / 0.663 / 0.636`、全课程 LSTM 的周级 `69.69% → 95.23%`、`5,040`、`35/53`、“assessment 点击是已证实泄漏通道”、time-on-task 系数反号区间、“每 15 分钟只有 6 分钟在任务上”，以及 PSM/CEM/PR-AUC/校准规范的具体阈值（本轮无已核验 claim）。

`validate_run.py` 应扫描 `REPORT.md`：命中“不得引用”项直接报 `error`；命中“可引用”项但同一句没有目标/特征限定，报 `warning`。

### 9.5 复跑新增来源

- Wu, Jia & Pham (2026). *Frontiers in Education*, 11:1917092. https://doi.org/10.3389/feduc.2026.1917092
- Le, Abel & Laforge (2026). LEAP: leakage-aware early prediction. arXiv:2605.25794. https://arxiv.org/abs/2605.25794
- Aljohani, Fayoumi & Hassan (2019). *Sustainability*, 11(24):7238. https://doi.org/10.3390/su11247238
- Baker et al. (2020). *International Journal of Educational Technology in Higher Education*, 17. https://doi.org/10.1186/s41239-020-00187-1
- Kovanović et al. (2015). *LAK '15*, 184–193. https://doi.org/10.1145/2723576.2723623
- Tempelaar et al. (2020). *PLOS ONE*, 15(6):e0233977. https://doi.org/10.1371/journal.pone.0233977
- Saint et al. (2022). *Computers and Education: Artificial Intelligence*, 3:100060. https://doi.org/10.1016/j.caeai.2022.100060
- Fan et al. (2021). *International Journal of Artificial Intelligence in Education*, 31. https://doi.org/10.1007/s40593-021-00249-z
- Gašević et al. (2016). *Internet and Higher Education*, 28:68–84. https://doi.org/10.1016/j.iheduc.2015.10.002
- Fredricks & McColskey (2012). In *Handbook of Research on Student Engagement*, 763–782. https://doi.org/10.1007/978-1-4614-2018-7_37

### 9.6 一句话总结

**每个阶段只做一件事，每个动作留下一条日志，每个输出留下一个哈希；上一阶段没有 pass，下一阶段不许开始。**
