"""
【16】实验 0 v2 · 检验 0-6 v2（跨域）（预先登记见 实验0说明.md 7.4）

用途
  主分析集中有第二方向的论文：
    Poisson  y3 ~ γ·log(第二方向当年论文数) + log(第二方向 Y+1～Y+3 施引方规模) + 第二方向匹配分数
                 + log1p(P1) + 控制变量 | 子课题(K=2000)×年份 + 发表月份 + 发表渠道
    OLS      log1p(y3) ~ 同上
  标准误按子课题聚类。通过标准：Poisson 与 OLS 的 γ 都 < 0 且 p < 0.05。
  稳健（不计入判定）：不控制第二方向需求（第一版的设定）；K = 5000。

输入
  results_v2/focal.parquet（【11】）

输出（results_v2/）
  16_crossdomain.csv、16_crossdomain.md

用法
  python 16_v2_test_crossdomain.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

M13 = import_module("13_v2_test_micro")
warnings.filterwarnings("ignore")


def main():
    t0 = time.time()
    df = M13.prepare()
    n_main = int(df.in_main.sum())
    df = df[df.in_main & df.topic2.notna() & (df.supply_topic2 > 0) & (df.demand_topic2 > 0)].copy()
    df["l_sup2"], df["l_dem2"], df["t2s"] = np.log(df.supply_topic2), np.log(df.demand_topic2), df.topic2_score.fillna(0)
    X = f"x_m1_{C.V2_TAU}"
    rows = []
    for name, extra, model, y, k, grp in [
        ("主设定", ["l_dem2"], "pois", "y3", C.V2_K, "0-6"),
        ("主设定", ["l_dem2"], "ols", "ly3", C.V2_K, "0-6"),
        ("不控制第二方向需求（第一版设定）", [], "pois", "y3", C.V2_K, "稳健"),
        ("K = 5000", ["l_dem2"], "pois", "y3", 5000, "稳健"),
    ]:
        fml = f"{y} ~ l_sup2 + {' + '.join(extra + ['t2s', X] + M13.CONTROLS)} | cy{k} + pub_month + venue"
        m = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": f"c{k}"})
        t = m.tidy().loc["l_sup2"]
        rows.append({"group": grp, "spec": name, "model": "Poisson" if model == "pois" else "OLS", "K": k,
                     "gamma": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": m._N,
                     "effect_if_doubled_%": (2 ** t["Estimate"] - 1) * 100})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.V2_RES, "16_crossdomain.csv"), index=False)
    mm = res[res.group == "0-6"]
    passed = bool((mm.gamma < 0).all() and (mm.p < 0.05).all())
    md = ["# 检验 0-6 v2（跨域）\n",
          f"主分析集中有第二方向的论文 {len(df):,} 篇（占 {len(df) / n_main:.1%}）。`gamma` = log(第二方向当年论文数) 的系数。\n",
          res.round(4).to_markdown(index=False), "\n",
          f"**0-6 v2**：Poisson γ = {mm.iloc[0].gamma:+.4f}（p = {mm.iloc[0].p:.2g}），OLS γ = {mm.iloc[1].gamma:+.4f}（p = {mm.iloc[1].p:.2g}）；"
          f"**{'通过' if passed else '未通过'}**。\n"]
    open(os.path.join(C.V2_RES, "16_crossdomain.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"16 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
