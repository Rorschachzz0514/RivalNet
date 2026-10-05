"""
【07】汇总实验 0 的六个检验，按设计文档 3.4 节的判定标准给出结论

用途
  读取【02】–【06】的结果 csv，逐项按预设标准判定"通过 / 未通过"，统计通过数并对应设计文档的分支：
    ≥4 项通过 → 按原计划推进；≤1 项 → 启用后备方案；0-1 的 b 接近 0 → Dirichlet 份额头合理。
  事后追加的诊断（控制需求后的系数）单独列出，不计入通过数。

输入（results/）
  02_macro.csv、03_micro.csv、04_placebo.csv、05_twins.csv、06_crossdomain.csv

输出（results/）
  实验0结果.md

用法
  python 07_report.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

R = lambda n: pd.read_csv(os.path.join(C.RES_DIR, n))
fp = lambda p: "< 1e-4" if p < 1e-4 else f"{p:.3g}"


def main():
    m2, m3, m4, m5, m6 = R("02_macro.csv"), R("03_micro.csv"), R("04_placebo.csv"), R("05_twins.csv"), R("06_crossdomain.csv")
    rows = []

    r = m2.iloc[0]
    ok = r.b_supply < 1 and r.p_b_lt_1 < 0.05
    rows.append(("0-1 宏观", f"b = {r.b_supply:.3f}（95% CI {r.ci_low:.2f}～{r.ci_high:.2f}），H0: b≥1 的 p = {fp(r.p_b_lt_1)}",
                 "b < 1 且显著", ok))

    p, o = m3.iloc[0], m3.iloc[1]
    ok = p.beta < 0 and p.p < 0.05 and o.beta < 0 and o.p < 0.05
    rows.append(("0-2 微观", f"Poisson β = {p.beta:+.3f}（p {fp(p.p)}），OLS β = {o.beta:+.3f}（p {fp(o.p)}）",
                 "β < 0 且显著（两种模型）", ok))

    rob = m3[m3.group == "0-3"]
    n_neg, n_sig = int((rob.beta < 0).sum()), int(((rob.beta < 0) & (rob.p < 0.05)).sum())
    n_pos_sig = int(((rob.beta > 0) & (rob.p < 0.05)).sum())
    rows.append(("0-3 稳健", f"{len(rob)} 个设定：{n_neg} 个为负（{n_sig} 个显著），{n_pos_sig} 个显著为正；S3 文献耦合 ≈ 0",
                 "各设定方向一致为负", n_neg == len(rob)))

    q = m4.set_index("spec").loc["main"]
    ok = bool(q.outside_95) and q.real_beta < 0
    rows.append(("0-4 假对手", f"真实 β = {q.real_beta:+.3f}，置换 95% 范围 [{q.q025:+.3f}, {q.q975:+.3f}]，置换 p = {q.perm_p:.3f}",
                 "落在置换范围之外（且为负，才算竞争证据）", ok))

    t = m5.iloc[0]
    ok = t.beta < 0 and t.p < 0.05
    rows.append(("0-5 撞车", f"{int(t.n_pairs):,} 对；后发表者被引 {t.pct_diff:+.1f}%（p {fp(t.p)}）；加控制变量 {m5.iloc[1].pct_diff:+.1f}%",
                 "后发表者显著更少（参照 −21%）", ok))

    c = m6[m6.group == "0-6"]
    ok = bool((c.gamma < 0).all() and (c.p < 0.05).all())
    rows.append(("0-6 跨域", f"Poisson γ = {c.iloc[0].gamma:+.3f}（p {fp(c.iloc[0].p)}），OLS γ = {c.iloc[1].gamma:+.3f}（p {fp(c.iloc[1].p)}）",
                 "γ < 0 且显著", ok))

    n_pass = sum(x[3] for x in rows)
    q_formal = bool(q.outside_95)
    if n_pass >= 4:
        branch = "≥4 项通过 → 按原计划推进"
    elif n_pass <= 1:
        branch = "≤1 项通过 → 按设计文档启用后备方案"
    else:
        branch = "2–3 项通过 → 设计文档未规定，需讨论"

    d3 = m3[m3.group == "诊断"]
    d6 = m6[m6.group == "诊断"]
    d4 = m4.set_index("spec").loc["D1"]
    diag = [f"| 0-2 + 控制发表后相似论文数（D1，Poisson） | {d3.iloc[0].beta:+.3f} | {fp(d3.iloc[0].p)} |",
            f"| 0-2 + 控制发表后相似论文数（D1，OLS） | {d3.iloc[1].beta:+.3f} | {fp(d3.iloc[1].p)} |",
            f"| 0-2 同方向版本（D2） | {d3.iloc[2].beta:+.3f} | {fp(d3.iloc[2].p)} |",
            f"| 邻域内发表前占比（D3，Poisson，邻域大小固定） | {d3.iloc[3].beta:+.3f} | {fp(d3.iloc[3].p)} |",
            f"| D1 的置换检验：95% 范围 [{d4.q025:+.3f}, {d4.q975:+.3f}] | {d4.real_beta:+.3f} | 置换 p = {d4.perm_p:.3f} |",
            f"| 0-6 + 控制第二方向需求（Poisson） | {d6.iloc[0].gamma:+.3f} | {fp(d6.iloc[0].p)} |",
            f"| 0-6 + 控制第二方向需求（OLS） | {d6.iloc[1].gamma:+.3f} | {fp(d6.iloc[1].p)} |"]
    tw = m5.set_index("spec")

    n_focal = len(pd.read_parquet(os.path.join(C.RES_DIR, "focal.parquet"), columns=["paper_id"]))
    md = [f"# 实验 0 结果（{C.LABEL}）\n",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}；样本：2017–2021 年 AI 焦点论文 {n_focal:,} 篇（{C.LABEL}）；变量与方法见 `实验0说明.md`；"
          "各检验的完整表格见 `results/0x_*.md`。\n",
          "## 1. 按预设标准的判定\n",
          "| 检验 | 结果 | 通过标准 | 判定 |", "|---|---|---|---|",
          *[f"| {a} | {b} | {s} | {'✅ 通过' if k else '❌ 未通过'} |" for a, b, s, k in rows],
          f"\n**通过 {n_pass} / 6 项**（若 0-4 只看\"是否落在置换范围之外\"、不看方向，则为 {n_pass + (q_formal and not rows[3][3])} 项）。"
          f"按设计文档：{branch}。\n",
          "## 2. 主要发现\n",
          "1. **宏观上接近零和（0-1）**：给定施引方规模，方向内论文数增加几乎不增加该方向的总被引（b ≈ 0，"
          "显著小于 1）。这支持\"份额加起来等于 1\"的 Dirichlet 份额头。但只有 385 格、供给与需求高度共线，检验力有限；"
          "加入方向固定效应后 b 也接近 0。\n",
          "2. **不控制需求时，\"撞题对手多\"反而被引更多（0-2、0-3、0-6）**：发表前一年内相似论文越多（以及第二方向越拥挤），"
          "被引显著更高；所有 SPECTER2 阈值（0.90–0.98）都如此。0-4 置换检验说明这不是巧合（真实系数远在置换分布之外），"
          "但方向与\"竞争\"相反。最可能的解释是**热门子方向的混淆**：方向 × 年份的固定效应只有 77 个方向这一粒度，"
          "同一方向内，相似论文扎堆的子方向同时也是引用需求大的子方向。\n",
          "3. **控制需求后，系数一致转为负（事后诊断，不计入判定）**：\n",
          "| 诊断设定 | 系数 | p |", "|---|---|---|", *diag, "",
          "   即：邻域大小（需求）固定时，前面对手越多 / 越晚进入，被引越少；第二方向供给在控制该方向需求后也为负。"
          "这与 0-1\"给定需求、供给增加摊薄每篇被引\"的结论一致。\n",
          "   **注意**：D1–D3 用\"发表后 365 天内的相似论文数\"做需求代理，它也会受焦点论文自身质量影响（好论文会吸引跟进者），"
          "属于可能的\"坏控制\"，可能把系数往负方向偏。0-6 的诊断用的是方向层面的需求（单篇论文几乎不影响），"
          "这个问题小得多，因此更可信。\n",
          f"4. **撞车论文对没有发现\"后发表者吃亏\"（0-5）**：{int(t.n_pairs):,} 对中，后发表者被引反而略高 {t.pct_diff:+.1f}%；"
          f"加入控制变量后为 {m5.iloc[1].pct_diff:+.1f}%（不显著）；剔除相似度 ≥ 0.99 的对后为 {tw.loc['剔除相似度≥0.99', 'pct_diff']:+.1f}%（不显著）。"
          f"时间差 1–30 天的对后发表者高出 {tw.loc['时间差 1–30 天', 'pct_diff']:+.1f}%，很可能混有未合并的同一论文的不同版本"
          "（例如会议版与期刊版），需人工核验 `05_twins_for_manual_check.csv` 后再下结论。"
          "与 Hill & Stein（2025，结构生物学，−21%）不同；AI 领域预印本普及、\"先发\"优势可能被会议周期抵消。\n",
          "## 3. 对后续实验的含义\n",
          "- 按预设标准只通过 1 项（0-1），形式上触发设计文档的后备方案。\n",
          "- 但诊断结果显示：竞争效应在**控制需求**后存在，而且 0-1 本身就是\"给定需求\"的检验。"
          "这正是份额模型的设定——需求（总被引）单独建模、论文之间分配份额。"
          "建议：把\"需求\"作为模型的显式输入（例如邻域 / 方向层面的施引方规模），而不是假设相似论文越多被引越少。\n",
          "- 待决定（需要你确认）：(1) 是否将\"控制需求后的检验\"补入实验 0 的预设检验；"
          "(2) 0-5 人工核验的抽样规模；(3) 是否按后备方案调整。\n"]
    out = os.path.join(C.RES_DIR, "实验0结果.md")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print("07 DONE")


if __name__ == "__main__":
    main()
