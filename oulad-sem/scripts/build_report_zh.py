"""Chinese translation of the report, generated from the same tables.

The English report is authoritative; this script mirrors its structure so the two
can be compared section by section. It reads the same pipeline tables plus the
verification outputs and writes ``Report_zh.tex`` / ``Report_zh.pdf``.

Run from ``oulad-sem/``::

    python scripts/build_report_zh.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_root  # noqa: E402
from src.reporting import (_tex, compile_pdf, f, load_tables,  # noqa: E402
                           load_verification, pct)

LQ = "\u201c"   # full-width opening quote
RQ = "\u201d"   # full-width closing quote

FLOW_ZH = {
    "all_enrolments": "全部选课记录",
    "has_first_weighted_tma": "存在首个正权重 TMA",
    "registered_by_deadline": "截止前已注册",
    "registered_at_cutoff": "截点时仍在读",
    "banked_excluded": "排除承接旧成绩",
    "known_final_result": "结局已知",
    "withdrawn_at_cutoff": "截点时最终退课",
    "binary_records": "二分类记录",
    "fail_records": "其中 Fail",
}
BAND_ZH = {
    "not_submitted_by_14": "截点前未提交",
    "early_7plus": "提前 7 天以上",
    "early_1to6": "提前 1--6 天",
    "due_day": "截止日当天",
    "late_1to7": "逾期 1--7 天",
    "late_8to14": "逾期 8--14 天",
}
METRIC_ZH = {
    "submitted_by_cutoff": "截止时已提交",
    "submitted_after_due": "截止后逾期提交",
    "relative_submission_day": "提交日相对截止日",
    "post_active_days": "后段活跃天数",
    "post_longest_inactive_run": "最长无点击间隔",
    "post_week2_share": "第二周点击占比",
}


def _preamble(date_str: str) -> str:
    return (
        "\\documentclass[11pt]{ctexart}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{caption}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{array}\n"
        "\\usepackage{float}\n"
        "\\captionsetup{font=small}\n"
        "\\setlength{\\parskip}{4pt}\n"
        "\\setlength{\\parindent}{2em}\n"
        "\\graphicspath{{./}}\n"
        "\\title{同一截止日，不同的持续性\\\\\\large 点击量相近的学生之间，提交时机与截止后活动}\n"
        "\\author{第 9 组：刘嘉阳（50027216）、陈彦斌（50026846）、\\\\"
        "王彦舜（50026688）、张育铭（50025933）}\n"
        "\\date{数据挖掘课程项目 \\\\ " + date_str + "}\n"
        "\\begin{document}\n"
        "\\maketitle"
    )


def _table(headers, rows, colspec, caption, label, size="\\small"):
    out = ["\\begin{table}[H]", "\\centering", size,
           "\\begin{tabular}{" + colspec + "}", "\\toprule",
           " & ".join(headers) + " \\\\", "\\midrule"]
    for row in rows:
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\caption{" + caption + "}",
            "\\label{" + label + "}", "\\end{table}"]
    return "\n".join(out)


def build(ctx, date_str: str) -> str:
    t, stages = load_tables(ctx)
    vtab, vverdict = load_verification()
    counts = stages.get("S02", {}).get("metrics", {})
    flow = t.get("t02_cohort_flow", pd.DataFrame())
    balance = t.get("t06_balance", pd.DataFrame())
    effects = t.get("t06_effects", pd.DataFrame())
    primary = t.get("t06_primary_holm", pd.DataFrame())
    bands = t.get("t03_submission_timing", pd.DataFrame())
    models = t.get("t08_model_metrics", pd.DataFrame())
    cut = t.get("t09_cutoff_metrics", pd.DataFrame())
    specs = t.get("t11_specs", pd.DataFrame())
    classes = t.get("t10_class_metrics", pd.DataFrame())
    splits = t.get("t05_split_summary", pd.DataFrame())
    bg = t.get("t11_x_background_balance", pd.DataFrame())
    bg_eff = t.get("t11_x_background_effects", pd.DataFrame())
    bg_info = t.get("t11_x_background_constrained", pd.DataFrame())
    rematch = t.get("t11_x_rematching", pd.DataFrame())
    nonsub = t.get("t11_x_non_submitter_outcomes", pd.DataFrame())
    atrisk = t.get("t11_x_at_risk_warning", pd.DataFrame())

    n_binary = int(counts.get("n_binary", 0))
    n_fail = int(counts.get("n_fail", 0))
    n_elig = int(counts.get("n_eligible", 0))
    n_withdrawn = int(counts.get("n_withdrawn_eligible", 0))
    n_total = int(counts.get("n_total", 0))
    fail_rate = float(counts.get("fail_rate", float("nan")))
    main_pairs = (balance.loc[balance["tolerance"].sub(0.05).abs() < 1e-9].iloc[0]
                  if len(balance) else None)
    main_models = models[models["split"].eq("main_test")] if len(models) else pd.DataFrame()
    ap_primary = (float(main_models.loc[main_models["model"].eq("M2"),
                                        "average_precision"].mean())
                  if len(main_models) and "M2" in set(main_models["model"])
                  else float("nan"))
    no_skill = (float(main_models["no_skill_pr_auc"].mean())
                if len(main_models) else fail_rate)

    def effect_of(name):
        if not len(effects) or name not in set(effects["feature"]):
            return float("nan")
        row = effects[np.isclose(effects["tolerance"], 0.05) & effects["feature"].eq(name)]
        return float(row["fail_minus_pass"].iloc[0]) if len(row) else float("nan")

    parts = [_preamble(date_str)]
    parts.append("\\noindent\\textbf{页数说明。}正文在第 \\pageref{endmainzh} 页结束；"
                 "参考文献、披露与附录不计入 6--10 页的正文要求。")
    parts.append("\\noindent 本文件是英文报告的中文对照版。两份报告的表格与数字由同一条流水线"
                 "生成；如出现不一致，以英文版为准。")

    # ----------------------------------------------------------- 1 问题
    parts.append("\\section{研究问题}")
    parts.append(
        "在线课程平台会为每一次资源浏览留下时间戳。点击总量是这份记录最直观的概括，"
        "也是预测谁会不及格时最直观的基线；然而两名学生可能点击次数完全相同，"
        "最终结果却差别很大。本项目要问的是：围绕一个强制截止日的活动\\emph{时机}，"
        "是否携带了点击量本身没有的信息。")
    parts.append(
        "具体问题是：在同一课程批次内，在首次计分作业（TMA）截止后 14 天点击量相近的学生中，"
        "提交时机与截止后持续性是否与最终不及格（而非通过或优秀）相关？本报告把 Fail 作为正类，"
        "因此关心的指标是 Fail 的平均精确率（AP），参照值是同一批测试记录中 Fail 的占比。"
        "最终退课（Withdrawn）单独处理：它不进二分类比较，只在第 6 节的风险集描述里出现。")
    parts.append(
        "三个设计选择定义了这项工作。第一，锚点不是开课日，而是每个批次\\emph{第一个权重为正的 "
        "TMA 截止日}，因为截止日才是大量学生要么提交、要么消失的时刻。第二，观察窗口对称："
        "截止前 14 天（$d-13\\ldots d$）提供起点基线，截止后 14 天（$d+1\\ldots d+14$）"
        "才是关心的行为。第三，比较按截止后点击量配对，因此持续性的差异不能用一方点得更多来解释。")
    parts.append(
        "本报告全部是观察性关联。配对是按最终结局构造的，所以样本在构造上就是 1:1，"
        "不能用它估计总体不及格率。截止日取作业表里的计划日期；OULAD 不记录个人延期，"
        "也不记录成绩公布日，因此提前或逾期只相对于计划日期，截点时已知成绩无从谈起。")

    # ----------------------------------------------------------- 2 数据
    parts.append("\\section{数据与设计}")
    parts.append(
        "数据是 Open University Learning Analytics Dataset（OULAD），公开版本包含 7 门课、"
        "22 个开课批次、32,593 条选课记录。分析单位是一条选课记录，因为同一名学生可能出现在"
        "多个批次里，而截止日是批次的属性。")
    rows = []
    for _, r in flow.iterrows():
        step = str(r["step"])
        if step in FLOW_ZH:
            rows.append([FLOW_ZH[step], str(int(r["records"]))])
    if rows:
        parts.append(_table(["样本流程步骤", "记录数"], rows, "lr",
                            "有序的入组流程。一条记录要进入分析，必须有首个正权重 TMA、"
                            "在截止日前注册、在 $d+14$ 仍在读、且不是承接旧成绩。"
                            "最终退课的记录不进二分类比较，但保留在第 6 节的风险集里。",
                            "tab:flowzh"))
    parts.append(
        f"{n_total:,} 条选课记录中，{n_elig:,} 条满足全部入组规则；其中 {n_withdrawn:,} 条最终退课，"
        f"{n_binary:,} 条有二分类结局（{n_fail:,} 条 Fail 对 {n_binary - n_fail:,} 条通过或优秀），"
        f"Fail 占比为 {pct(fail_rate)}。AAA--FFF 六门课共 19 个批次存在正权重 TMA；GGG 没有，"
        f"因此不进入设计。")
    parts.append(
        "时间特征来自日粒度点击流。对每条记录，流水线统计 $d-13$ 至 $d+14$ 每日点击，"
        "再派生截止后活跃天数、最长连续零点击天数、第二周点击占比、最高 7 天占比、"
        "日分布归一化熵、以及首次活动日。提交字段一律掩蔽到 $d+14$：更晚的提交视为未提交，"
        "两个指示变量加一个有符号的天数距离构成提交块。")

    # ----------------------------------------------------------- 3 配对
    parts.append("\\section{配对比较}")
    parts.append(
        "配对在每个批次内部进行。Fail 记录与点击量首先精确相等、其次在容差内的通过或优秀记录做 "
        "1:1 匹配；正点击要求比值上限，零点击只能和零点击配。任何记录不被重复使用。"
        "贪心顺序不保证全局最优，实际配对会被报告出来，以便检查残差。")
    if main_pairs is not None:
        rows = []
        for _, r in balance.iterrows():
            rows.append([pct(r["tolerance"], 0), str(int(r["fail_records"])),
                         str(int(r["pairs"])), pct(r["fail_coverage"]),
                         str(int(r["zero_post_pairs"])),
                         f(r["mean_post_click_gap"], 4), f(r["pre_log_click_smd"])])
        parts.append(_table(
            ["容差", "Fail 记录", "配对", "Fail 覆盖率", "零点击配对", "平均点击差", "前段 SMD"],
            rows, "rrrrrrr",
            "按截止后点击配对。5\\% 是主规格，0\\% 与 10\\% 为敏感性。前段标准化差被一并报告，"
            "因为配对\\emph{故意}不均衡截止前的窗口。", "tab:pairingzh",
            size="\\footnotesize\\setlength{\\tabcolsep}{4pt}"))
        parts.append(
            f"主配对得到 {int(main_pairs['pairs']):,} 对 Fail，覆盖全部 Fail 的 "
            f"{pct(main_pairs['fail_coverage'])}，平均相对点击差 "
            f"{f(main_pairs['mean_post_click_gap'], 4)}。截止后点击量几乎被完全均衡（标准化差 "
            f"{f(main_pairs['post_log_click_smd'])}），但截止前的窗口没有：标准化差 "
            f"{f(main_pairs['pre_log_click_smd'])}，原始差 {f(effect_of('pre_clicks'), 1)} 次点击。"
            f"Fail 在截止前就已经略不活跃，所以配对差值回答的是条件问题："
            f"在截止后点击相近的前提下，还有什么不同？")
    if len(primary):
        rows = []
        for _, r in primary.iterrows():
            rows.append([METRIC_ZH.get(r["metric"], _tex(r["metric"])),
                         str(int(r["pairs"])), f(r["fail_minus_pass"], 4),
                         f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]",
                         f(r["p_value"], 4), f(r["p_holm"], 4),
                         "是" if bool(r["reject_holm_0_05"]) else "否"])
        parts.append(_table(
            ["主要指标", "配对", "Fail $-$ Pass", "95\\% 区间", "$p$",
             "$p_{\\mathrm{Holm}}$", "拒绝"],
            rows, "rrrcccc",
            "六项预先指定的配对差值，附学生关联组件 bootstrap 区间，以及族内 Holm 校正。"
            "负数表示 Fail 一侧更低。提交距离只在双方都在截点前提交时有定义，"
            "所以它的配对更少。区间以实际发生的配对为条件。",
            "tab:primaryzh", size="\\footnotesize"))
        parts.append(
            "六项指标方向一致，且都不接近零。Fail 记录更不可能在截点前提交、更可能逾期提交；"
            "在已提交者中提交得更晚；截止之后活跃天数更少、沉默更长、第二周点击占比更低。"
            "Holm 校正后六项全部显著，因此故事的方向不取决于把哪一项当作主要指标。")
    if len(bands):
        rows = []
        for _, r in bands.iterrows():
            rows.append([BAND_ZH.get(r["submission_band"], _tex(r["submission_band"])),
                         str(int(r["records"])), str(int(r["fail"])),
                         str(int(r["pass_distinction"])), str(int(r["withdrawn"])),
                         pct(r["fail_share_of_binary"])])
        parts.append(_table(
            ["提交分段", "记录", "Fail", "通过或优秀", "退课", "Fail 占比"],
            rows, "rrrrrr",
            "按提交分段看结局构成。分段由掩蔽后的提交日构造，所以截点前没有提交的记录"
            "一律记为未提交到 $d+14$，即使之后交了。", "tab:bandszh",
            size="\\footnotesize"))
        not_sub = bands[bands["submission_band"].eq("not_submitted_by_14")]
        early = bands[bands["submission_band"].eq("early_7plus")]
        if len(not_sub) and len(early):
            parts.append(
                f"描述性差异很陡：截点前未提交的记录里有 "
                f"{pct(not_sub['fail_share_of_binary'].iloc[0])} 最终 Fail，"
                f"而提前一周以上提交的只有 {pct(early['fail_share_of_binary'].iloc[0])}。"
                f"这是描述而非因果——提交与结局都反映同一份努力——但它解释了为什么每个模型"
                f"都保留提交状态。")

    # ----------------------------------------------------------- 4 预测
    parts.append("\\section{预测信息}")
    parts.append(
        "配对描述的是点击量相当时谁不同。另一个问题是：时机是否在点击量之外提供预测信息。"
        "模型阶梯在学生分组 60/20/20 切分上拟合（没有任何学生跨两部分），阈值在验证集上选择，"
        "再把训练加验证重新拟合后对测试集打分。")
    if len(main_models):
        rows = []
        for _, r in main_models.groupby("model").mean(numeric_only=True).reset_index().iterrows():
            rows.append([_tex(r["model"]), str(int(r["n_features"])),
                         f(r["average_precision"], 4), f(r["no_skill_pr_auc"], 4),
                         f(r["roc_auc"], 4), f(r["brier"], 4),
                         f(r["calibration_slope"], 3), f(r["f1"], 3)])
        parts.append(_table(
            ["模型", "特征数", "Fail AP", "无技能 AP", "ROC-AUC", "Brier", "校准斜率", "F1"],
            rows, "rrrrrrrr",
            "分组主测试集上的模型阶梯。M0 = 截止前后点击量 + 提交状态；M1 加后段时间特征；"
            "M2 加提交距离，是主模型；M3 加资源构成。M4（加首次 TMA 成绩）因冻结配置把成绩"
            "排除在严格特征之外而未列入，只作为回顾性上界单独讨论。阈值在验证集上选择。",
            "tab:modelszh", size="\\footnotesize\\setlength{\\tabcolsep}{3pt}"))
        m0 = float(main_models.loc[main_models["model"].eq("M0"), "average_precision"].mean())
        m1 = float(main_models.loc[main_models["model"].eq("M1"), "average_precision"].mean())
        parts.append(
            f"主模型的 Fail AP 是 {f(ap_primary, 4)}，无技能基线是 {f(no_skill, 4)}。"
            f"阶梯说明了增益来自哪里：M0 已含提交状态，达到 {f(m0, 4)}；后段时间块把它推到 "
            f"{f(m1, 4)}；提交距离项再加到 {f(ap_primary, 4)}。也就是说，大部分预测信号是提交状态，"
            f"而不是点击序列的细结构。梯度提升替代模型与之在噪声范围内，说明线性模型没有漏掉"
            f"明显的非线性模式。")
    if len(cut):
        rows = [[str(int(r["post_days"])), _tex(r["model"]),
                 f(r["average_precision"], 4), f(r["brier"], 4)]
                for _, r in cut.iterrows()]
        parts.append(_table(["截止后天数", "模型", "Fail AP", "Brier"], rows, "rrrr",
                            "同一切分上的观察窗曲线。窗口越长信息越多，所以 AP 上升；"
                            "这不是方法上的增益，不能这样读。", "tab:cutoffzh",
                            size="\\footnotesize"))
        first = cut[cut["post_days"].eq(1) & cut["model"].eq("M2")]
        last = cut[cut["post_days"].eq(14) & cut["model"].eq("M2")]
        if len(first) and len(last):
            parts.append(
                f"把截点从 $d+1$ 移到 $d+14$，主模型的 AP 从 "
                f"{f(first['average_precision'].iloc[0], 4)} 升到 "
                f"{f(last['average_precision'].iloc[0], 4)}，特征清单完全不变。"
                f"所以这个提升是观察得更久换来的信息，不是模型更好。")

    # ----------------------------------------------------------- 5 验证设计
    parts.append("\\section{验证设计}")
    if len(splits):
        rows = [[_tex(r["split"]), str(int(r["n_train"])), str(int(r["n_test"])),
                 pct(r["train_positive_rate"]), pct(r["test_positive_rate"]),
                 str(int(r["student_overlap"]))] for _, r in splits.iterrows()]
        parts.append(_table(
            ["切分", "训练", "测试", "训练 Fail 占比", "测试 Fail 占比", "学生重叠"],
            rows, "rrrrrr",
            "所有切分都按学生 ID 分组：没有任何学生同时出现在两侧。重叠列在构造上恒为零，"
            "并在运行时被断言。留一课程外推只有 6 折而不是 7 折，因为只有 6 门课有正权重 TMA。",
            "tab:splitszh", size="\\footnotesize"))

    # ----------------------------------------------------------- 6 稳健性
    parts.append("\\section{稳健性与范围}")
    if len(specs):
        rows = [[_tex(r["spec"]), str(int(r["post_days"])), pct(r["tolerance"], 0),
                 _tex(r["match_on"]), str(int(r["pairs"])), pct(r["fail_coverage"]),
                 f(r["pre_log_click_smd"])] for _, r in specs.iterrows()]
        parts.append(_table(
            ["规格", "后段天数", "容差", "按什么匹配", "配对", "Fail 覆盖率", "前段 SMD"],
            rows, "rrrrrrr",
            "稳健性曲线。+21/+28 的两个变体各出现两次：按截止后点击配对（与主设计可比）"
            "和按全窗口点击配对（与早期的 +21/+28 研究可比）。两者回答不同问题，不能互相排名。",
            "tab:specszh", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}"))
    if len(classes):
        rows = [[_tex(r["y_t3"]), str(int(r["records"])), pct(r["share"])]
                for _, r in classes.iterrows()]
        parts.append(_table(["风险集中的最终结局", "记录", "占比"], rows, "rrr",
                            "风险集把退课保留为独立类别。它与二分类比较是不同人群、不同问题。",
                            "tab:riskzh", size="\\footnotesize"))
    parts.append(
        "所示指标在 8 个规格里符号全部一致，包括两个按不同量匹配的全窗口变体。幅度会动："
        "按总点击匹配后截止后的差值变小，这是预期内的，因为全窗口总量本身有一部分是截止前的量。"
        "风险集描述是唯一建模退课的地方。把二分类结论当作退课预警是错的：那些记录在截点时仍在读，"
        "而预测幸存者里谁不及格和标记可能离开的人不是同一个模型。")

    # ----------------------------------------------------------- 7 局限
    parts.append("\\section{局限}")
    parts.append(
        "\\begin{itemize}\\itemsep2pt\n"
        "\\item AP 是 Fail 的平均精确率，不是准确率；无技能参照是同一批测试记录里的 Fail 占比。\n"
        "\\item 配对是设计出来的 1:1，不能估计总体不及格率。贪心顺序不保证全局最优，"
        "bootstrap 只在实际配对上重抽学生，因此不含重新配对的不确定性。\n"
        "\\item 二分类队列排除了截点前退课的记录，因此不是实时预警人群。\n"
        "\\item 首次 TMA 成绩本身参与最终结果；用它的模型只是回顾性上界，OULAD 也没有成绩公布日。\n"
        "\\item OULAD 的切分在多轮探索中被反复查看，所以区间与增量都是探索性的，即使协议已经冻结。\n"
        "\\item VLE 的活动类型是平台类别，不等于内容质量；日粒度点击次数无法换算学习时长。\n"
        "\\end{itemize}")
    parts.append(
        "有两点必须在答辩时强调。第一，截止前窗口没有被主配对均衡，把它也均衡的敏感性只剩"
        "很小一部分配对，所以诚实的说法是比较固定了截止后点击量，并报告截止前历史仍差多少。"
        "第二，时间特征带来的增益很小。截止后持续性在描述上区分结局，远比它在预测上改善模型"
        "来得清楚；改善最多的那个模型加的是提交距离，而提交距离本身就是截止行为，不是点击模式。")

    parts.append("\\label{endmainzh}")
    parts.append("\\section*{披露}")
    parts.append(
        "AIGC 工具参与了代码脚手架、文字编辑与 LaTeX 排版。全部分析代码、数值结果与论断"
        "均由全组复核，每一张图表都由流水线从公开数据重新生成。没有外部合作者提供数据或结果。")
    parts.append("\\section*{数据与代码可用性}")
    parts.append(
        "未经修改的 OULAD 压缩包随仓库提交并附 SHA-256。解压后的 CSV、学生级特征、"
        "配对索引与预测不进入版本控制。一条命令即可复现分析；提交用的 ZIP 包含代码、"
        "冻结配置与运行日志。")

    # ----------------------------------------------------------- 附录 A
    parts.append("\\appendix")
    parts.append("\\section{探索性扩展}")
    parts.append(
        "主分析冻结之后又跑了四个探索块。它们不改变任何主要指标或主模型，因此放在附录"
        "而不是正文。")
    if len(bg):
        rows = [[_tex(r["covariate"]), "分类" if r["kind"] == "categorical" else "数值",
                 f(r["smd_unmatched"], 3), f(r["smd_matched"], 3),
                 "是" if bool(r["balanced_matched"]) else "否"] for _, r in bg.iterrows()]
        parts.append(_table(["协变量", "类型", "配对前 SMD", "配对后 SMD", "$|$SMD$|\\le$0.10"],
                            rows, "rrrrr",
                            "配对样本的入学前协变量。配对本来就只均衡截止后点击，"
                            "所以这里的残差是预期内的，报告而不修正。", "tab:c1zh",
                            size="\\footnotesize"))
        worst = bg.sort_values("abs_smd_matched", ascending=False).iloc[0]
        parts.append(
            f"最大的残差不平衡是 {_tex(worst['covariate'])} 的 {f(worst['smd_matched'], 3)} "
            f"个标准差；教育程度、IMD 分带与先前尝试次数仍是最不平衡的几项。"
            f"若再要求教育与 IMD 完全一致，只剩 "
            f"{pct(bg_info['share_of_main_pairs'].iloc[0]) if len(bg_info) else '--'} 的配对，"
            f"太少，不能替代主比较。")
        if len(bg_eff):
            parts.append(
                "在这个子集上，活跃天数差为 "
                f"{f(bg_eff.loc[bg_eff['feature'].eq('post_active_days'), 'fail_minus_pass'].iloc[0], 3)} 天，"
                f"区间明显更宽：方向保持，精度不足。")
    if len(rematch):
        rows = [[_tex(r["metric"]), f(r["observed"], 3), f(r["mean_rematched"], 3),
                 f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]"] for _, r in rematch.iterrows()]
        parts.append(_table(["指标", "固定配对估计", "重配对均值", "重配对 95\\% 区间"],
                            rows, "rrrr",
                            "重新配对的不确定性：每个重复都重抽学生、重建队列、重跑贪心配对，"
                            "因此区间包含配对本身的不确定性。重复次数不多，是宽度检查，不是定论。",
                            "tab:c2zh", size="\\footnotesize"))
        parts.append("两项指标在重配对下符号与幅度都保持，区间只略微变宽，说明配对选择不是结论的来源。")
    if len(nonsub):
        outcome_zh = {"Fail": "Fail", "Withdrawn": "退课", "Pass": "通过"}
        rows = [[outcome_zh.get(r["outcome"], _tex(r["outcome"])),
                 str(int(r["records"])), pct(r["share_of_non_submitters"])]
                for _, r in nonsub.iterrows()]
        parts.append(_table(["结局", "记录", "占未提交者比例"], rows, "rrr",
                            "截点前未提交记录的结局分布。这是二分类比较无法覆盖的子群，"
                            "因为退课与未提交在这里高度重叠。", "tab:c3zh",
                            size="\\footnotesize"))
        parts.append(
            f"{int(nonsub['non_submitters'].iloc[0]):,} 条截点前未提交的记录中，"
            f"{pct(nonsub.loc[nonsub['outcome'].eq('Withdrawn'), 'share_of_non_submitters'].sum())} "
            f"最终退课，"
            f"{pct(nonsub.loc[nonsub['outcome'].eq('Fail'), 'share_of_non_submitters'].sum())} 不及格。"
            f"所以未提交更像离开的信号而不是失败的信号——这正是二分类要排除退课的原因。")
    if len(atrisk):
        rows = [[_tex(r["model"]), str(int(r["n_test"])), pct(r["positive_share"]),
                 f(r["average_precision"], 4), f(r["roc_auc"], 4), f(r["brier"], 4)]
                for _, r in atrisk.iterrows()]
        parts.append(_table(["模型", "测试记录", "Fail 或退课占比", "AP", "ROC-AUC", "Brier"],
                            rows, "rrrrrr",
                            "预警目标：截点时在读者的 Fail 或退课，把退课当作正例。"
                            "人群与标签都与二分类不同，AP 不可与第 4 节比较。", "tab:c4zh",
                            size="\\footnotesize"))

    # ----------------------------------------------------------- 附录 B
    parts.append("\\clearpage")
    parts.append("\\section{探索扩展的验证}")
    if not vtab:
        parts.append("未找到验证表，本节略去。重新运行 explore 与 verify 系列脚本即可生成。")
    else:
        parts.append(
            "附录 A 的内容都是探索性的。之后补了四项检查，看其中哪一条站得住："
            "预注册的增量检验、外推汇总、带最小可检测效应的子群审计，以及两个负对照"
            "（延迟分型与日序）。它们都不改变任何主要数字。")
        v1 = vtab.get("v1_frozen_increment")
        if v1 is not None and len(v1):
            rows = []
            for _, r in v1.iterrows():
                if str(r["model"]).startswith("delta"):
                    continue
                rows.append([_tex(r["model"]), str(int(r["n_features"])),
                             f(r["average_precision"], 5)])
            for _, r in v1[v1["model"].astype(str).str.startswith("delta")].iterrows():
                rows.append(["$\\Delta$ " + _tex(str(r["model"]).replace("delta_", "")), "–",
                             f"{float(r['average_precision']):+.5f}"])
            parts.append(_table(["增量臂", "特征数", "Fail AP"], rows, "rrr",
                                "预注册的增量检验（协议在运行前冻结）。判定规则是区间不含零"
                                "\\emph{且}点估计不低于 0.005；两个臂都不满足，"
                                "因此深度指标没有带来可辨别的预测信息。", "tab:v1zh",
                                size="\\footnotesize"))
            obs = vverdict.get("v1_frozen_verdict", {}).get("observed", {})
            if obs:
                parts.append(
                    f"冻结臂把 Fail AP 从 {f(obs.get('ap_baseline'), 5)} 移到 "
                    f"{f(obs.get('ap_candidate'), 5)}，差值为 "
                    f"{float(obs.get('point', float('nan'))):+.5f}，95\\% 区间 "
                    f"[{float(obs.get('ci_low', float('nan'))):+.5f}, "
                    f"{float(obs.get('ci_high', float('nan'))):+.5f}]。"
                    f"剔除与主特征完全重复的两列后，结论相同。")
        v2 = vtab.get("v2_transfer_rows")
        if v2 is not None and len(v2) and "average_precision" in v2:
            m2 = v2[v2["model"].eq("M2")]
            rows = [[_tex(r["split"]), str(int(r["n_test"])), pct(r["no_skill_pr_auc"]),
                     f(r["average_precision"], 4),
                     f(float(r["average_precision"] - r["no_skill_pr_auc"]), 3)]
                    for _, r in m2.iterrows()]
            if rows:
                parts.append(_table(["切分", "测试记录", "Fail 占比", "Fail AP", "AP $-$ 无技能"],
                                    rows, "rrrrr",
                                    "主模型的外推汇总。折间原始 AP 不可比，因为 Fail 占比从 0.14 "
                                    "到 0.34，可比的是最后一列。CCC 没有 2013 批次，"
                                    "所以时间外推同时部分是跨课程。", "tab:v2zh",
                                    size="\\scriptsize"))
        v3 = vtab.get("v3_contrasts_mde")
        if v3 is not None and len(v3):
            rows = [[_tex(r["attribute"]),
                     "仅行为" if r["feature_set"] == "behaviour_only" else "全部特征",
                     f(r["auc_gap"], 3), f"[{f(r['ci_low'], 3)}, {f(r['ci_high'], 3)}]",
                     f(r["mde_0.80"], 3), "是" if bool(r["reject_bh_0_05"]) else "否"]
                    for _, r in v3.iterrows()]
            parts.append(_table(["属性", "特征集", "AUC 差", "95\\% 区间",
                                 "MDE（80\\% 功效）", "过 BH"],
                                rows, "llrrrr",
                                "子群审计。差值小于最小可检测效应，意思是样本量分辨不出来，"
                                "不等于没有差异。bootstrap 的 p 值下限是 1 除以重复次数，"
                                "落在下限的对比只能说在 bootstrap 分辨率内显著。",
                                "tab:v3zh", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}"))
            surviving = v3[v3["reject_bh_0_05"]]
            below = v3[~v3["gap_exceeds_mde"]]
            if len(surviving):
                names = "、".join(
                    f"{_tex(r['attribute'])}（{'仅行为' if r['feature_set'] == 'behaviour_only' else '全部特征'}，"
                    f"差 {r['auc_gap']:+.3f}，BH p {r['p_bh']:.3f}）"
                    for _, r in surviving.iterrows())
                parts.append(
                    f"{len(v3)} 个对比中有 {len(surviving)} 个通过 BH 校正：{names}。"
                    f"另有 {len(below)} 个对比的差值小于最小可检测效应，"
                    f"这些属于未能检出，而不是不存在。")
            else:
                parts.append(
                    f"{len(v3)} 个对比没有一个通过 BH 校正；其中 {len(below)} 个的差值"
                    f"小于最小可检测效应，属于未能检出而不是不存在。")
        if vverdict.get("v4_profile_verdict"):
            parts.append(
                "两个负对照补完了图景。附录 A 的延迟剖面无法检验文献里高成就、低成就、"
                "偶发延迟者的三分型：聚类跑在（提交分数，窗口长度）上，而窗口长度强烈双峰，"
                "剖面几乎是窗口长度的切分。诚实的说法是这套操作化分不开延迟与窗口长度，"
                "而不是分型本身错了。")
        v5 = vtab.get("v5_sequence_models")
        v5c = vtab.get("v5_sequence_contrasts")
        if v5 is not None and len(v5):
            rows = [[_tex(r["model"]), str(int(r["n_features"])),
                     f(r["average_precision"], 4), f(r["roc_auc"], 4)]
                    for _, r in v5.iterrows()]
            parts.append(_table(["序列臂", "特征数", "Fail AP", "ROC-AUC"], rows, "rrrr",
                                "同一切分上的日序对照。PyTorch 装不上，所以在有序日向量上用的是"
                                "前馈网络而不是循环网络；关键对比是有序与打乱。", "tab:v5zh",
                                size="\\footnotesize"))
            if v5c is not None and len(v5c):
                ordered = v5c[v5c["contrast"].eq("ordered_minus_shuffled")]
                if len(ordered):
                    r = ordered.iloc[0]
                    parts.append(
                        f"打乱 14 天的日向量使 Fail AP 变化 {float(r['point']):+.4f}"
                        f"（{f(r['ci_low'], 4)} 至 {f(r['ci_high'], 4)}），"
                        f"说明日的顺序没有携带多重集之外的信息。这不能否定已发表的序列模型结果"
                        f"——那些用不同的标签、不同的切分；只能说在本设计里，"
                        f"序列表示没有增加什么。")

    parts.append("\\section{复现}")
    parts.append(
        "\\begin{verbatim}\n"
        "make test\n"
        "make sim && make validate RUN_ID=$(cat .last_sim_run)\n"
        "make full && make validate RUN_ID=$(cat .last_full_run)\n"
        "\\end{verbatim}\n"
        "每个阶段写入阶段摘要、交接说明与 JSONL 事件；校验器按 \\texttt{docs/contracts} "
        "的 schema 检查日志，要求 0 错误、0 警告。")
    parts.append("\\end{document}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026 年 11 月")
    args = parser.parse_args()
    cfg = load_config()
    root = project_root()
    out = root / "outputs"
    sys.path.insert(0, str(root))
    from src.logging_utils import RunContext

    run_id = (root / ".last_full_run").read_text().strip()
    ctx = RunContext(cfg, run_id)
    tex = out / "Report_zh.tex"
    tex.write_text(build(ctx, args.date), encoding="utf-8")
    ok, log = compile_pdf(tex, out, runs=2)
    if not ok:
        print(log[-4000:])
        raise SystemExit(f"Chinese report failed to compile: {tex}")
    shutil.copyfile(out / "Report_zh.pdf", root / "Report_zh.pdf")
    print(f"wrote {root / 'Report_zh.pdf'}")


if __name__ == "__main__":
    main()
