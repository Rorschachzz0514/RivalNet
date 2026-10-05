"""
【17】实验 0 v2 · 汇总：按预先登记的标准判定六个检验，并单列事后分析（预先登记见 实验0说明.md 第 7 节）

用途
  读取【12】–【16】的结果，逐项按 7.4 节的标准判定；与第一版（results/）并排比较；单列【13】的事后诊断与【18】的浪潮分析
  （不计入通过数）；给出对后续实验的含义。

输入
  results_v2/12_macro.csv、13_micro.csv、14_placebo.csv、15_twins.csv、16_crossdomain.csv、18_wave.csv、18_wave_placebo.csv、focal.parquet
  results/02_macro.csv、03_micro.csv、04_placebo.csv、05_twins.csv、06_crossdomain.csv（第一版，对照）

输出
  results_v2/实验0结果_v2.md

用法
  python 17_v2_report.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

fp = lambda p: "< 1e-4" if p < 1e-4 else f"{p:.2g}"
pct = lambda b: (2 ** b - 1) * 100


def main():
    R = lambda n: pd.read_csv(os.path.join(C.V2_RES, n))
    V1 = lambda n: pd.read_csv(os.path.join(C.RES_DIR, n))
    m2, m3, m4, m5, m6, w, wp = (R(n) for n in ("12_macro.csv", "13_micro.csv", "14_placebo.csv", "15_twins.csv",
                                                 "16_crossdomain.csv", "18_wave.csv", "18_wave_placebo.csv"))
    f = pd.read_parquet(os.path.join(C.V2_RES, "focal.parquet"), columns=["in_main"])
    rows = []
    r = m2.iloc[0]
    rows.append(("0-1 宏观", f"b = {r.b_supply:.2f}（95% CI {r.ci_low:.2f}～{r.ci_high:.2f}）：论文数翻倍，总被引 × {2 ** r.b_supply:.2f}",
                 r.b_supply < 1 and r.p_b_lt_1 < 0.05))
    p, o = m3.iloc[0], m3.iloc[1]
    rows.append(("0-2 微观", f"β = {p.beta:+.3f}（OLS {o.beta:+.3f}）：前一年相似论文翻倍，被引 {pct(p.beta):+.1f}%",
                 p.beta < 0 and p.p < 0.05 and o.beta < 0 and o.p < 0.05))
    rob = m3[m3.group == "0-3"]
    rows.append(("0-3 稳健", f"{len(rob)} 个设定：{int((rob.beta < 0).sum())} 个为负，{int(((rob.beta > 0) & (rob.p < 0.05)).sum())} 个显著为正"
                 f"（为负的只有 S3 文献耦合两项）", bool((rob.beta < 0).all())))
    q = m4.iloc[0]
    rows.append(("0-4 假对手", f"真实 β = {q.real_beta:+.3f}，置换范围 [{q.q025:+.3f}, {q.q975:+.3f}]：不是巧合，但方向为正", bool(q.passed)))
    t = m5.iloc[0]
    rows.append(("0-5 撞车", f"{int(t.n_pairs):,} 对，后发表者被引 {t.pct_diff:+.1f}%（p {fp(t.p)}）", t.beta < 0 and t.p < 0.05))
    c = m6[m6.group == "0-6"]
    rows.append(("0-6 跨域", f"γ = {c.iloc[0].gamma:+.3f}（OLS {c.iloc[1].gamma:+.3f}）：第二方向论文翻倍，被引 {pct(c.iloc[0].gamma):+.1f}%",
                 bool((c.gamma < 0).all() and (c.p < 0.05).all())))
    n_pass = sum(x[2] for x in rows)

    # 第一版对照
    a2, a3, a4, a5, a6 = (V1(n) for n in ("02_macro.csv", "03_micro.csv", "04_placebo.csv", "05_twins.csv", "06_crossdomain.csv"))
    v1 = [f"b = {a2.iloc[0].b_supply:+.2f}", f"β = {a3.iloc[0].beta:+.3f}", f"{int((a3[a3.group == '0-3'].beta < 0).sum())} / 16 为负",
          f"β = {a4.set_index('spec').loc['main', 'real_beta']:+.3f}", f"{a5.iloc[0].pct_diff:+.1f}%", f"γ = {a6.iloc[0].gamma:+.3f}"]
    tw = m5.set_index("spec")
    het = m3[m3.group == "0-7"]
    post = m3[m3.group == "事后诊断"]
    md = [f"# 实验 0 结果（第二版 v2）\n",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}。数据 `subsets/exp0_v2`（剔除无效条目、补合并、修正日期、子课题聚类）；"
          f"焦点论文主分析集 {int(f.in_main.sum()):,} 篇（期刊 / 会议 / 论文集 / arXiv），竞争对手 = 全部 v2 样本论文。"
          "检验与标准在运行前登记于 `实验0说明.md` 第 7 节。\n",
          "## 1. 按预先登记标准的判定\n",
          "| 检验 | 第一版 | 第二版结果 | 判定 |", "|---|---|---|---|",
          *[f"| {a} | {v} | {b} | {'✅ 通过' if k else '❌ 未通过'} |" for (a, b, k), v in zip(rows, v1)],
          f"\n**通过 {n_pass} / 6 项**（第一版 1 / 6）。设计文档 3.4 节对 2–3 项没有规定分支；0-1 的 b 在 0 与 1 之间 → 份额头需要加\"流向集合外\"选项。\n",
          "## 2. 发现\n",
          f"1. **宏观：部分竞争。** 在 9,943 个子课题 × 年份格子上，给定施引方规模，论文数翻倍时总被引只增加约 {(2 ** m2.iloc[0].b_supply - 1) * 100:.0f}%"
          f"（b = {m2.iloc[0].b_supply:.2f}）。只用年份 FE 时 b = {m2.iloc[1].b_supply:.2f}，K = 5000 时 b = {m2[(m2.K == 5000) & (m2.fe == 'cluster + Y')].b_supply.iloc[0]:.2f}，"
          "都显著小于 1、大于 0：既不是互不影响，也不是完全零和。第一版只有 385 格，b ≈ 0 是检验力不足和粒度太粗的结果。\n",
          f"2. **撞车论文对：先发优势真实存在。** 修正日期、补合并、剔除共同作者后，后发表者被引少 {-t.pct_diff:.0f}%；"
          f"同一年份的对少 {-tw.loc['只用同一年份', 'pct_diff']:.0f}%，相似度 ≥ 0.98 的对少 {-tw.loc['相似度 ≥ 0.98', 'pct_diff']:.0f}%"
          "（Hill & Stein 2025 在结构生物学中为 −21%）；相差不到 30 天的对没有差异。第一版\"后发者没吃亏\"是日期占位值与重复记录造成的。\n",
          f"3. **跨域：控制需求后，第二方向越拥挤被引越少**（翻倍 {pct(c.iloc[0].gamma):+.1f}%）；不控制需求时为正（{pct(m6.iloc[2].gamma):+.1f}%）。\n",
          f"4. **\"相似论文数\"不是竞争强度。** 即使在 2000 / 5000 个子课题 × 年份内比较，前一年相似论文越多被引越多（翻倍 {pct(p.beta):+.1f}%），"
          "所有阈值、粒度、焦点集合都一样；分组看，各渠道、各年份、有无预印本、各需求增长组也都为正。"
          "高相似论文扎堆反映的是一小片研究前沿的热度和可见度，而不是对引用的争夺。只有文献耦合（参考文献高度重合）的前一年对手为负"
          f"（{pct(m3[m3.spec == 'S3 文献耦合（前一年）'].beta.iloc[0]):+.1f}%）。\n",
          "## 3. 事后分析（不计入判定）\n",
          "| 分析 | 系数 | p | 说明 |", "|---|---|---|---|",
          *[f"| {r.spec} | {r.beta:+.3f} | {fp(r.p)} | 【13】 |" for r in post.itertuples()],
          *[f"| {r.spec} | {r.beta:+.3f} | {fp(r.p)} | 【18】 |" for r in w.itertuples()],
          f"\n- 加入当年、后一年的全部相似论文后，前一年系数变成 {post.iloc[1].beta:+.2f}，看起来像\"浪潮里越晚越吃亏\"。"
          f"但后一年的相似论文里有一部分**引用了焦点论文**（跟随者），其数量本身由焦点论文的质量决定（跟随者数的系数 {w.iloc[8].beta:+.2f}）。"
          f"只数与焦点论文互不引用的同期 / 后来者时，前一年系数变成 {w.iloc[0].beta:+.3f}（Poisson）/ {w.iloc[1].beta:+.3f}（OLS），"
          f"\"前一年占比\"为 {w.iloc[4].beta:+.3f}（p {fp(w.iloc[4].p)}）。**第一版 D1 / D3 的负系数主要来自这个坏控制，不能作为竞争证据。**\n",
          "## 4. 对后续实验的含义\n",
          "- **份额头**：b ≈ 0.4 → Dirichlet 份额头加\"流向集合外\"（outside option），或用非零和的份额 + 总量结构；实验 2 的消融里比较三者。\n",
          "- **需求塔必不可少**：0-6 不控制需求时符号相反，0-1 的 b 也依赖需求口径。实验 1（评价诊断）应量化\"方向 / 子课题热度\"解释了多少被引差异。\n",
          "- **竞争特征**：对手的影响必须允许为正（可见度 / 前沿效应）也允许为负（替代）；单纯的\"相似论文数\"会以正号进入模型。"
          "有清楚负效应的是：近乎相同的工作被别人抢先发表（撞车，−14%）、参考文献高度重合的前一年论文（文献耦合）、第二方向的拥挤（给定需求）。"
          "实验 2 应把\"是否已有独立团队的近乎相同工作先发表、早多少天\"作为显式特征，并区分文献耦合对手与语义相似对手。\n",
          "- **需要重复验证**：0-5 v2 的先发优势与 0-6 v2 的跨域效应在实验 6（其他学科）中预先登记后重复；【18】的事后分析不再作为证据。\n",
          "## 5. 异质性（0-7 v2，描述）\n", het[["spec", "beta", "se", "p", "n"]].round(4).to_markdown(index=False), "\n"]
    out = os.path.join(C.V2_RES, "实验0结果_v2.md")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print("17 DONE")


if __name__ == "__main__":
    main()
