"""
【09】并排比较实验 0 在不同样本下的结果：全部论文 / 只保留期刊 + 会议 / 期刊 + 会议 + 会议论文集

用途
  读取各结果目录（results、results_jc、results_jcb，由【25】的子样本变体 + 【01】–【08】生成）中的 csv，
  把六个检验的关键数字、预设判定、事后诊断放到一张表里，便于看"删掉其他类型论文后结论是否改变"。

输入
  results*/02_macro.csv、03_micro.csv、04_placebo.csv、05_twins.csv、06_crossdomain.csv、08_twins_by_class.csv、focal.parquet

输出（results/）
  09_compare_variants.csv、09_compare_variants.md

用法
  python 09_compare_variants.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

VARIANTS = [("results", "全部论文"), ("results_jc", "期刊 + 会议"), ("results_jcb", "期刊 + 会议 + 论文集")]


def fmt(b, p, pct=False):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{b:+.1f}%{star}" if pct else f"{b:+.3f}{star}"


def summarize(d):
    R = lambda n: pd.read_csv(os.path.join(C.HERE, d, n))
    m2, m3, m4, m5, m6, m8 = (R(n) for n in ("02_macro.csv", "03_micro.csv", "04_placebo.csv", "05_twins.csv",
                                              "06_crossdomain.csv", "08_twins_by_class.csv"))
    n = len(pd.read_parquet(os.path.join(C.HERE, d, "focal.parquet"), columns=["paper_id"]))
    r2 = m2.iloc[0]
    p3, o3 = m3.iloc[0], m3.iloc[1]
    rob = m3[m3.group == "0-3"]
    q = m4.set_index("spec")
    t5 = m5.iloc[0]
    c6 = m6[m6.group == "0-6"]
    dg3 = m3[m3.group == "诊断"].set_index(["spec", "model"])
    dg6 = m6[m6.group == "诊断"]
    m8 = m8.set_index("spec")
    passes = [
        r2.b_supply < 1 and r2.p_b_lt_1 < 0.05,
        p3.beta < 0 and p3.p < 0.05 and o3.beta < 0 and o3.p < 0.05,
        bool((rob.beta < 0).all()),
        bool(q.loc["main", "outside_95"]) and q.loc["main", "real_beta"] < 0,
        t5.beta < 0 and t5.p < 0.05,
        bool((c6.gamma < 0).all() and (c6.p < 0.05).all()),
    ]
    mark = lambda ok: "✅" if ok else "❌"
    return {
        "焦点论文数": f"{n:,}",
        "0-1 宏观 b（施引方规模，年份 FE）": f"{r2.b_supply:+.3f}（CI {r2.ci_low:+.2f}～{r2.ci_high:+.2f}）{mark(passes[0])}",
        "0-1 加方向 FE": f"{m2.iloc[1].b_supply:+.3f}",
        "0-2 微观 Poisson β": fmt(p3.beta, p3.p) + mark(passes[1]),
        "0-2 微观 OLS β": fmt(o3.beta, o3.p),
        "0-3 稳健：为负 / 显著为正（共 16）": f"{int((rob.beta < 0).sum())} / {int(((rob.beta > 0) & (rob.p < 0.05)).sum())} {mark(passes[2])}",
        "0-3 S3 文献耦合 Poisson β": fmt(rob.set_index("spec").loc["S3 文献耦合", "beta"], rob.set_index("spec").loc["S3 文献耦合", "p"]),
        "0-4 置换：真实 β / 95% 范围": f"{q.loc['main', 'real_beta']:+.3f} / [{q.loc['main', 'q025']:+.3f}, {q.loc['main', 'q975']:+.3f}] {mark(passes[3])}",
        "0-5 撞车：后发表者被引差异": f"{t5.pct_diff:+.1f}%（{int(t5.n_pairs):,} 对，p {t5.p:.2g}）{mark(passes[4])}",
        "0-5 只用不同团队": fmt(m8.loc["只用不同团队（主）", "pct_diff"], m8.loc["只用不同团队（主）", "p"], pct=True)
                         + f"（{int(m8.loc['只用不同团队（主）', 'n_pairs']):,} 对）",
        "0-6 跨域 Poisson γ": fmt(c6.iloc[0].gamma, c6.iloc[0].p) + mark(passes[5]),
        "**预设标准通过数**": f"**{sum(passes)} / 6**",
        "诊断 D1：控制发表后相似论文数（Poisson）": fmt(dg3.iloc[0].beta, dg3.iloc[0].p),
        "诊断 D3：邻域内发表前占比（Poisson）": fmt(dg3.iloc[3].beta, dg3.iloc[3].p),
        "诊断：0-6 控制第二方向需求（Poisson）": fmt(dg6.iloc[0].gamma, dg6.iloc[0].p),
    }


def main():
    cols = {}
    for d, lab in VARIANTS:
        if os.path.exists(os.path.join(C.HERE, d, "08_twins_by_class.csv")):
            cols[lab] = summarize(d)
        else:
            print(f"跳过 {d}（结果不全）")
    t = pd.DataFrame(cols)
    t.index.name = "指标"
    t.to_csv(os.path.join(C.HERE, "results", "09_compare_variants.csv"), encoding="utf-8-sig")
    md = ["# 实验 0：不同样本的结果对比\n",
          "全部论文 = 原样本；期刊 + 会议 = 只保留正式版发表在期刊或会议的论文（焦点、竞争对手、方向供给都删除其他论文）；"
          "+ 论文集 = 再加 OpenAlex 中归为 book series 的会议论文集（LNCS、CCIS 等）。被引仍按全部来源统计。"
          "星号：* p<0.05，** p<0.01，*** p<0.001。\n",
          t.to_markdown(), "\n"]
    open(os.path.join(C.HERE, "results", "09_compare_variants.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
