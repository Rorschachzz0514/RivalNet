"""
【19】实验 0 v2 · 论文 Introduction 的引子图（Figure 1）：引用竞争真实存在，而且是双向的

三格（英文标签，供论文使用；数字全部来自实验 0 v2 的预先登记检验，图只是可视化）
  (a) Citation demand is shared（0-1 宏观）：子课题 × 年份格子上，论文数与该届总被引的关系
      （去掉子课题 FE、年份 FE 与施引方规模后的分箱散点，FWL）；参考线：斜率 1 = 互不影响，斜率 0 = 完全零和；
      标注预先登记的 Poisson 估计 b = 0.41。
  (b) Being scooped costs citations（0-5 撞车论文对）：后发表者相对先发表者的被引差异，按时间差分组（论文对 FE Poisson，95% 区间）。
  (c) ... yet similar papers bring visibility（0-2 微观）：前一年高度相似论文（SPECTER2 ≥ 0.95）的数量分组，
      焦点论文被引相对同子课题同年平均的差异（去掉 子课题×年份、发表月份、渠道 FE 后的残差，95% 区间）。

输入
  results_v2/cell_cohort.parquet、results_v2/focal.parquet、results_v2/15_twins.csv、results_v2/12_macro.csv、results_v2/13_micro.csv
输出（results_v2/）
  19_teaser.pdf、19_teaser.png、19_teaser_data.csv

用法
  python 19_teaser_figure.py
"""
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pyfixest as pf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
BLUE, ORANGE, GREEN, GRAY = "#2F5597", "#C55A11", "#548235", "#7F7F7F"


def demean(v, groups, tol=1e-10, max_iter=500):
    """多组固定效应的交替投影去均值（保留全部行，不删单例组）"""
    v = v.astype(float).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(max_iter):
        old = v.copy()
        for c in codes:
            v = v - (np.bincount(c, weights=v) / np.bincount(c))[c]
        if np.abs(v - old).max() < tol:
            break
    return v


def resid(df, y, x_rhs, fe):
    """FWL：y 对控制变量与固定效应回归后的残差（全部行都保留）"""
    groups = [df[g.strip()].to_numpy() for g in fe.split("+")]
    ry = demean(df[y].to_numpy(), groups)
    if x_rhs:
        X = np.column_stack([demean(df[x].to_numpy(), groups) for x in x_rhs.split("+")])
        ry = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return pd.Series(ry, index=df.index)


def main():
    rows = []
    # ---------------- (a) 宏观
    c = pd.read_parquet(os.path.join(R, "cell_cohort.parquet"))
    # 与预先登记的检验同一模型：Poisson，子课题 FE + 年份 FE + log(施引方规模)；
    # 把 log(论文数) 换成"去掉 FE 与需求后的残差"的十分位虚拟变量，各箱系数即该箱相对最低箱的 log 总被引（分箱 Poisson）
    c = c[(c.K == 2000) & (c.supply > 0) & (c.demand_followers > 0) & (c.cohort_y3 >= 0)].copy()
    c["l_sup"], c["l_dem"] = np.log(c.supply), np.log(c.demand_followers)
    c["rx"] = resid(c, "l_sup", "l_dem", "cluster + Y")
    c["bin"] = pd.qcut(c.rx, 10, labels=False, duplicates="drop")
    m = pf.fepois("cohort_y3 ~ C(bin) + l_dem | cluster + Y", data=c, vcov={"CRV1": "cluster"})
    co, se_ = m.coef(), m.se()
    ga = c.groupby("bin").agg(x=("rx", "mean"), n=("rx", "size")).reset_index()
    ga["y"] = [0.0 if k == 0 else float(co[[i for i in co.index if i.startswith("C(bin)") and i.endswith(f"{k}]")][0]]) for k in ga.bin]
    ga["se"] = [0.0 if k == 0 else float(se_[[i for i in se_.index if i.startswith("C(bin)") and i.endswith(f"{k}]")][0]]) for k in ga.bin]
    x0 = float(ga.x.iloc[0])
    ga["x"] = ga.x - x0                                   # 以最低箱为原点
    mac = pd.read_csv(os.path.join(R, "12_macro.csv"))
    main_row = mac.iloc[0]
    b_ols = float(main_row["b_supply"])                    # 画线用预先登记的 Poisson 估计
    print("macro main row:", main_row.to_dict(), flush=True)
    print(ga.round(3).to_string(), flush=True)
    for r in ga.itertuples():
        rows.append({"panel": "a", "x": r.x, "y": r.y, "n": r.n})

    # ---------------- (b) 撞车论文对
    tw = pd.read_csv(os.path.join(R, "15_twins.csv"))
    pick = [("时间差 1–30 天", "≤30 days"), ("时间差 31–90 天", "31–90 days"), ("时间差 91–183 天", "91–183 days"),
            ("主设定（加控制变量）", "All pairs"), ("相似度 ≥ 0.98", "Sim. ≥ 0.98")]
    gb = []
    for k, lab in pick:
        r = tw[tw.spec == k].iloc[0]
        lo, hi = 100 * (np.exp(r.beta - 1.96 * r.se) - 1), 100 * (np.exp(r.beta + 1.96 * r.se) - 1)
        gb.append({"label": lab, "pct": 100 * (np.exp(r.beta) - 1), "lo": lo, "hi": hi, "n": int(r.n_pairs_used), "p": r.p})
        rows.append({"panel": "b", "label": lab, "y": gb[-1]["pct"], "lo": lo, "hi": hi, "n": gb[-1]["n"]})
    gb = pd.DataFrame(gb)

    # ---------------- (c) 微观
    f = pd.read_parquet(os.path.join(R, "focal.parquet"),
                        columns=["paper_id", "Y", "in_main", "y3", "c2000", "pub_month", "venue_class", "n_ym1_ge0.95"])
    f = f[f.in_main & f.y3.notna() & f.c2000.notna()].copy()
    f["cy"] = f.c2000.astype(int).astype(str) + "_" + f.Y.astype(str)
    f["ly"] = np.log1p(f.y3)
    f["venue"] = f.venue_class.astype(str)
    f["pub_month"] = f.pub_month.fillna(0).astype(int)
    f["ry"] = resid(f, "ly", None, "cy + pub_month + venue")
    f = f.dropna(subset=["ry"])
    edges = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 16, "9–16"), (17, 10 ** 9, "≥17")]
    gc = []
    x = f["n_ym1_ge0.95"]
    for lo_, hi_, lab in edges:
        s = f.loc[(x >= lo_) & (x <= hi_), "ry"]
        m, se = s.mean(), s.std() / np.sqrt(len(s))
        gc.append({"label": lab, "pct": 100 * (np.exp(m) - 1), "lo": 100 * (np.exp(m - 1.96 * se) - 1), "hi": 100 * (np.exp(m + 1.96 * se) - 1), "n": len(s)})
        rows.append({"panel": "c", "label": lab, "y": gc[-1]["pct"], "lo": gc[-1]["lo"], "hi": gc[-1]["hi"], "n": len(s)})
    gc = pd.DataFrame(gc)
    mic = pd.read_csv(os.path.join(R, "13_micro.csv"))
    eff = float(mic[(mic.spec == "主设定") & (mic.model == "Poisson")]["effect_if_doubled_%"].iloc[0])
    coup = float(mic[mic.spec == "S3 文献耦合（前一年）"]["effect_if_doubled_%"].iloc[0])
    pd.DataFrame(rows).to_csv(os.path.join(R, "19_teaser_data.csv"), index=False)
    print(gb.round(2).to_string(), "\n", gc.round(2).to_string(), flush=True)

    # ---------------- 画图
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]})
    # (a)
    a = ax[0]
    xs = np.linspace(0, ga.x.max() * 1.05, 10)
    a.plot(xs, xs, ls="--", color=GRAY, lw=1)
    a.plot(xs, 0 * xs, ls=":", color=GRAY, lw=1)
    a.plot(xs, b_ols * xs, color=BLUE, lw=1.5)
    a.errorbar(ga.x, ga.y, yerr=1.96 * ga.se, fmt="o", color=BLUE, ms=3.5, lw=0.8, capsize=1.5, zorder=3)
    a.text(0.66, 0.96, "no competition\n(slope 1)", ha="right", va="center", fontsize=6.5, color=GRAY)
    a.text(xs[-1], 0.02, "zero-sum (slope 0)", ha="right", va="bottom", fontsize=6.5, color=GRAY)
    b, lo, hi = main_row["b_supply"], main_row["ci_low"], main_row["ci_high"]
    a.text(0.02, 1.3, f"b = {b:.2f} [{lo:.2f}, {hi:.2f}]\n2× papers → {2 ** b:.2f}× citations",
           ha="left", va="top", fontsize=7, color=BLUE)
    a.set_xlabel("Δ log # papers in subtopic-year")
    a.set_ylabel("Δ log total citations of cohort")
    a.set_title("(a) Citation demand is shared", loc="left", fontweight="bold", fontsize=8.5)
    # (b)
    a = ax[1]
    cols = [ORANGE, ORANGE, ORANGE, "#8C3B0C", "#8C3B0C"]
    xb = np.arange(len(gb))
    a.bar(xb, gb.pct, color=cols, width=0.62, alpha=0.9)
    a.errorbar(xb, gb.pct, yerr=[gb.pct - gb.lo, gb.hi - gb.pct], fmt="none", ecolor="black", lw=0.8, capsize=2)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(xb)
    a.set_xticklabels(gb.label, rotation=28, ha="right")
    a.set_ylabel("Citations of the later twin (%)")
    a.set_title("(b) Being scooped hurts", loc="left", fontweight="bold", fontsize=8.5)
    a.axvline(2.5, color=GRAY, lw=0.6, ls=":")
    for i, r in gb.iterrows():
        a.text(i, min(r.lo, r.pct) - 2.5, f"{r.pct:+.0f}%", ha="center", va="top", fontsize=6.5)
    a.set_ylim(min(gb.lo.min() - 10, -50), 15)
    # (c)
    a = ax[2]
    xc = np.arange(len(gc))
    a.errorbar(xc, gc.pct, yerr=[gc.pct - gc.lo, gc.hi - gc.pct], fmt="o-", color=GREEN, ms=4, lw=1.3, capsize=2)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(xc)
    a.set_xticklabels(gc.label)
    a.set_xlabel("# highly similar papers in the prior year")
    a.set_ylabel("Citations vs. cell average (%)")
    a.set_title("(c) …yet neighbors bring visibility", loc="left", fontweight="bold", fontsize=8.5)
    a.text(0.97, 0.05, f"2× similar papers → {eff:+.1f}%\n(bib. coupled: {coup:+.1f}%)", transform=a.transAxes, ha="right", va="bottom",
           fontsize=7, color=GREEN)
    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(R, f"19_teaser.{ext}"), dpi=300, bbox_inches="tight")
    print("TEASER DONE", flush=True)


if __name__ == "__main__":
    main()
