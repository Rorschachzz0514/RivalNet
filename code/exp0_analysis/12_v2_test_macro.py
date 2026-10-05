"""
【12】实验 0 v2 · 检验 0-1 v2（宏观）：子课题内论文变多，是否摊薄每篇论文的被引（预先登记见 实验0说明.md 7.4）

用途
  一行 = 子课题 × 年份（2017–2021）。Poisson（pyfixest fepois）：
      cohort_y3 ~ b·log(supply) + c·log(demand) | 子课题 FE + 年份 FE        （主：K = 2000，demand = 施引方规模）
  标准误按子课题聚类。b = 1 互不影响，b = 0 完全零和。通过标准：b < 1 且单侧 p < 0.05。
  稳健（不计入判定）：只用年份 FE；K = 1000 / 5000；demand 换成净流入量（子课题 Y+1～Y+3 收到的被引减去本格自身）。

输入
  results_v2/cell_cohort.parquet（【11】）

输出（results_v2/）
  12_macro.csv、12_macro.md

用法
  python 12_v2_test_macro.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

warnings.filterwarnings("ignore")


def fit(df, k, demand, fe):
    d = df[(df.K == k) & (df.supply > 0) & (df[demand] > 0) & (df.cohort_y3 >= 0)].copy()
    d["l_sup"], d["l_dem"] = np.log(d.supply), np.log(d[demand])
    m = pf.fepois(f"cohort_y3 ~ l_sup + l_dem | {fe}", data=d, vcov={"CRV1": "cluster"})
    t = m.tidy()
    b, se = t.loc["l_sup", "Estimate"], t.loc["l_sup", "Std. Error"]
    c, sec = t.loc["l_dem", "Estimate"], t.loc["l_dem", "Std. Error"]
    return {"K": k, "demand": demand, "fe": fe, "n_cells": m._N, "b_supply": b, "se_b": se,
            "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
            "p_b_lt_1": stats.norm.cdf((b - 1) / se), "p_b_eq_0": 2 * stats.norm.sf(abs(b / se)),
            "c_demand": c, "se_c": sec}


def main():
    df = pd.read_parquet(os.path.join(C.V2_RES, "cell_cohort.parquet"))
    rows = [fit(df, C.V2_K, "demand_followers", "cluster + Y")]                     # 主设定
    rows += [fit(df, C.V2_K, "demand_followers", "Y"),
             fit(df, C.V2_K, "inflow_net", "cluster + Y"), fit(df, C.V2_K, "inflow_net", "Y")]
    for k in C.V2_KS:
        if k != C.V2_K:
            rows += [fit(df, k, "demand_followers", "cluster + Y"), fit(df, k, "demand_followers", "Y")]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.V2_RES, "12_macro.csv"), index=False)
    r = res.iloc[0]
    passed = r.b_supply < 1 and r.p_b_lt_1 < 0.05
    md = ["# 检验 0-1 v2（宏观，子课题 × 年份）\n",
          "Poisson：`cohort_y3 ~ b·log(supply) + c·log(demand) | FE`，标准误按子课题聚类。b = 1 互不影响；b = 0 完全零和。\n",
          res.round(4).to_markdown(index=False), "\n",
          f"**主设定（K = {C.V2_K}，施引方规模，子课题 + 年份 FE）**：b = {r.b_supply:.3f}（95% CI {r.ci_low:.3f}～{r.ci_high:.3f}），"
          f"H0: b ≥ 1 的单侧 p = {r.p_b_lt_1:.2g}；**{'通过' if passed else '未通过'}**。\n"]
    open(os.path.join(C.V2_RES, "12_macro.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
