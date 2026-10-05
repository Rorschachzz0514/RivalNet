"""
【02】实验 8 R7：原口径（counts_by_year，含自引）与剔除自引口径（引用边表）的实验 0 v2 核心结果并排比较
输出：/path/to/mpcc/exp8/results/R7_selfcite.md
"""
import os

import pandas as pd

A = "/path/to/mpcc/exp0_analysis/"
rows = []
for lab, d in (("原口径（含自引）", "results_v2"), ("剔除自引", "results_v2_noself")):
    m2 = pd.read_csv(A + d + "/12_macro.csv").iloc[0]
    m3 = pd.read_csv(A + d + "/13_micro.csv")
    m5 = pd.read_csv(A + d + "/15_twins.csv").iloc[0]
    m6 = pd.read_csv(A + d + "/16_crossdomain.csv").iloc[0]
    rows.append({"口径": lab, "0-1 宏观 b": f"{m2.b_supply:.3f} [{m2.ci_low:.2f}, {m2.ci_high:.2f}]",
                 "0-2 微观 β（Poisson）": f"{m3.iloc[0].beta:+.3f}（p {m3.iloc[0].p:.2g}）",
                 "0-5 撞车 后发表者": f"{m5.pct_diff:+.1f}%（p {m5.p:.2g}，{int(m5.n_pairs):,} 对）",
                 "0-6 跨域 γ（Poisson）": f"{m6.gamma:+.3f}（p {m6.p:.2g}）"})
os.makedirs("/path/to/mpcc/exp8/results", exist_ok=True)
md = "# 实验 8 R7：剔除自引\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n"
open("/path/to/mpcc/exp8/results/R7_selfcite.md", "w", encoding="utf-8").write(md)
print(md)
