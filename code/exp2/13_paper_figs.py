"""
【13】论文图 3（分析）：英文标签，数据全部来自已完成的实验

  (a) 竞争贡献 c = pred(完整) − pred(去掉对手)（实验 4，测试集 84,566 篇，5 个种子平均）按真实三年被引分档：均值与 95% 区间、c > 0 的比例
  (b) 各届论文（2017–2021）发表后第 1、2、3 年的平均被引，横轴为日历年：展示 2022 年起的整体下降（T 时不可见的需求冲击）

输入
  /path/to/mpcc/exp4/results/01_contrib.parquet（实验 4）、subsets/pred_v2/samples.parquet
输出
  results/13_fig3_analysis.pdf / .png、13_fig3_data.csv

用法
  python 13_paper_figs.py
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

BLUE, ORANGE, GREEN, GRAY = "#2F5597", "#C55A11", "#548235", "#7F7F7F"


def main():
    rows = []
    c = pd.read_parquet("/path/to/mpcc/exp4/results/01_contrib.parquet")
    print(c.columns.tolist(), len(c), flush=True)
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "Y", "y1", "y2", "y3"])
    if "y3" not in c.columns:
        c = c.merge(s[["paper_id", "y3"]], on="paper_id", how="left")
    ccol = "c" if "c" in c.columns else [x for x in c.columns if x.startswith("contrib") or x == "comp"][0]
    bins = [(0, 0, "0"), (1, 3, "1–3"), (4, 10, "4–10"), (11, 30, "11–30"), (31, 10 ** 9, ">30")]
    ga = []
    for lo, hi, lab in bins:
        v = c.loc[(c.y3 >= lo) & (c.y3 <= hi), ccol]
        m, se = v.mean(), v.std() / np.sqrt(len(v))
        ga.append({"label": lab, "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "pos": (v > 0).mean(), "n": len(v)})
        rows.append({"panel": "a", **ga[-1]})
    ga = pd.DataFrame(ga)
    # (b) 第 t 年被引 = y1、y2、y3 − y1 − y2（各届，均值）
    s["c1"], s["c2"] = s.y1, s.y2
    s["c3"] = s.y3 - s.y1 - s.y2
    gb = []
    for Y, g in s.groupby("Y"):
        for t in (1, 2, 3):
            gb.append({"cohort": int(Y), "t": t, "year": int(Y) + t, "mean": g[f"c{t}"].mean()})
            rows.append({"panel": "b", **gb[-1]})
    gb = pd.DataFrame(gb)
    pd.DataFrame(rows).to_csv(os.path.join(C.RES_DIR, "13_fig3_data.csv"), index=False)
    print(ga.round(3).to_string(), "\n", gb.pivot(index="cohort", columns="t", values="mean").round(2).to_string(), flush=True)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1, 1.15]})
    a = ax[0]
    x = np.arange(len(ga))
    a.bar(x, ga["mean"], color=[ORANGE if m < 0 else BLUE for m in ga["mean"]], width=0.65)
    a.errorbar(x, ga["mean"], yerr=[ga["mean"] - ga.lo, ga.hi - ga["mean"]], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(x)
    a.set_xticklabels(ga.label, fontsize=6.2, rotation=35, ha="right")
    a.set_xlabel("True 3-year citations", fontsize=7)
    a.set_ylabel("Competition contribution", fontsize=7)
    a.set_title("(a)", loc="left", fontsize=8, fontweight="bold")
    a = ax[1]
    cols = {2017: "#9DC3E6", 2018: "#6FA8DC", 2019: "#3D85C6", 2020: BLUE, 2021: ORANGE}
    for Y, g in gb.groupby("cohort"):
        a.plot(g.year, g["mean"], "-o", ms=2.5, lw=1.1, color=cols[Y], label=str(Y))
    a.axvspan(2021.5, 2024.6, color=GRAY, alpha=0.12, lw=0)
    a.set_xlabel("Calendar year", fontsize=7)
    a.set_ylabel("Citations per paper", fontsize=7)
    a.set_xticks([2018, 2020, 2022, 2024])
    a.set_ylim(top=gb["mean"].max() * 1.32)
    a.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right", handlelength=0.9, columnspacing=0.6, borderaxespad=0.1, labelspacing=0.2)
    a.set_title("(b)", loc="left", fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.RES_DIR, f"13_fig3_analysis.{ext}"), dpi=300, bbox_inches="tight")
    print("FIG3 DONE", flush=True)


if __name__ == "__main__":
    main()
