"""
【02】实验 6 · 三个学科并排汇总，按预先登记的"重复成功"标准判定（实验6说明.md 第 2、3 节）

输入
  /path/to/mpcc/exp0_analysis/results_v2{,_sf2730,_sf2604}/{12_macro,13_micro,15_twins,16_crossdomain,18_wave}.csv
  /path/to/mpcc/exp2/results{,_sf2730,_sf2604}/03_tests_{v5|field}.csv、03_summary_{v5|field}.csv
输出
  /path/to/mpcc/exp6/results/实验6结果.md、02_fields.csv

用法
  python 02_compare_fields.py
"""
import os

import pandas as pd

A0 = "/path/to/mpcc/exp0_analysis/"
A2 = "/path/to/mpcc/exp2/"
FIELDS = [("人工智能（1702）", "", "v5", "MPCNet5_l1"), ("肿瘤学（2730）", "_sf2730", "field", "MPCNet5_l1"),
          ("应用数学（2604）", "_sf2604", "field", "MPCNet5_l1")]
mk = lambda ok: "✅" if ok else "❌"


def one(lab, tag, ver, main):
    d = A0 + f"results_v2{tag}/"
    if not os.path.exists(d + "16_crossdomain.csv"):
        return None
    r = {"学科": lab}
    m2 = pd.read_csv(d + "12_macro.csv").iloc[0]
    r["0-1 b"] = f"{m2.b_supply:.2f} [{m2.ci_low:.2f}, {m2.ci_high:.2f}] {mk(m2.b_supply < 1 and m2.p_b_lt_1 < 0.05)}" + \
                 (" (b>0)" if m2.b_supply > 0 and m2.p_b_eq_0 < 0.05 else "")
    m3 = pd.read_csv(d + "13_micro.csv")
    p3, o3 = m3.iloc[0], m3.iloc[1]
    r["0-2 β（重复 AI：>0）"] = f"{p3.beta:+.3f} / OLS {o3.beta:+.3f} {mk(p3.beta > 0 and p3.p < 0.05)}"
    m5 = pd.read_csv(d + "15_twins.csv").iloc[0]
    r["0-5 撞车"] = f"{m5.pct_diff:+.1f}%（p {m5.p:.2g}，{int(m5.n_pairs):,} 对）{mk(m5.beta < 0 and m5.p < 0.05)}" if "beta" in m5 and pd.notna(m5.get("beta")) else f"对数不足（{int(m5.n_pairs)}）"
    m6 = pd.read_csv(d + "16_crossdomain.csv")
    c6 = m6[m6.group == "0-6"]
    r["0-6 γ"] = f"{c6.iloc[0].gamma:+.3f} / OLS {c6.iloc[1].gamma:+.3f} {mk(bool((c6.gamma < 0).all() and (c6.p < 0.05).all()))}"
    w = pd.read_csv(d + "18_wave.csv")
    w2 = w[w.spec.str.startswith("W2") & (w.model == "Poisson")].iloc[0]
    r["浪潮：前一年占比（重复 = 不显著）"] = f"{w2.beta:+.3f}（p {w2.p:.2g}）{mk(w2.p > 0.05)}"
    t = A2 + f"results{tag}/03_tests_{ver}.csv"
    sm = A2 + f"results{tag}/03_summary_{ver}.csv"
    if os.path.exists(t):
        tt = pd.read_csv(t).set_index("test")
        sm_ = pd.read_csv(sm, header=[0, 1], index_col=[0, 1])
        te = sm_.xs("test", level=1)
        r["MPC-Net MALE / 子课题内 Spearman"] = f"{te.loc[main, ('MALE', 'mean')]:.3f} / {te.loc[main, ('Spearman_子课题内', 'mean')]:.3f}"
        p1, p2, p4 = tt.loc["P1"], tt.loc["P2"], tt.loc["P4"]
        r["vs 最好对比方法 Spearman"] = f"{p2.diff_mean:+.3f} [{p2.ci_lo:+.3f}, {p2.ci_hi:+.3f}] {mk(bool(p2.passed))}（{p2.comparison.split('− ')[-1]}）"
        r["vs 最好对比方法 MALE"] = f"{p1.diff_mean:+.3f} [{p1.ci_lo:+.3f}, {p1.ci_hi:+.3f}] {mk(bool(p1.passed))}"
        r["vs 去掉对手 Spearman"] = f"{p4.diff_mean:+.3f} [{p4.ci_lo:+.3f}, {p4.ci_hi:+.3f}] {mk(bool(p4.passed))}"
    return r


def main():
    os.makedirs("/path/to/mpcc/exp6/results", exist_ok=True)
    rows = [x for x in (one(*f) for f in FIELDS) if x]
    t = pd.DataFrame(rows).set_index("学科").T
    t.to_csv("/path/to/mpcc/exp6/results/02_fields.csv")
    md = ["# 实验 6 结果：跨学科重复\n",
          "定义、阈值与超参数与 AI 完全相同（预先登记见 `实验6说明.md`）；✅ = 按预先登记的标准重复成功。"
          "AI 的实验 2 为 5 个种子，其他学科为 3 个种子。\n", t.to_markdown(), "\n"]
    open("/path/to/mpcc/exp6/results/实验6结果.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
