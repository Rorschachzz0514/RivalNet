"""
【06】检验 0-6（跨域）：论文的第二方向越拥挤，被引是否越少

用途
  只用有第二方向（topic2 非空）且第二方向当年供给 > 0 的焦点论文。在 0-2 主设定上加入第二方向的供给：
    Poisson  y3 ~ γ·log(supply_topic2) + log1p(prior_s2_95) + 控制变量 + topic2_score | 主方向×年份
    supply_topic2 = 第二方向当年论文数（n_rule，含 AI 以外的方向）
  通过标准：γ < 0 且 p < 0.05（Poisson 与 OLS 都满足）。
  诊断（事后追加，不计入判定）：再控制第二方向在 Y+1～Y+3 年的施引方规模 log(demand_topic2)
    （与 0-2 的问题相同：拥挤的方向往往也是热门方向，需要把需求分开）。
  标准误按主方向聚类。

输入
  results/focal.parquet（【01】）、DATA_DIR/topic_year.parquet（诊断用的第二方向需求）

输出（results/）
  06_crossdomain.csv   各设定的 γ、标准误、p 值、样本量
  06_crossdomain.md    结果表与解读

用法
  python 06_test_crossdomain.py           全量
  python 06_test_crossdomain.py --test    只用 2019 年（冒烟测试）

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
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

M3 = import_module("03_test_micro")
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sfx = "_test" if args.test else ""
    k = int(round(C.TAU * 100))
    df = M3.prepare(args.test)
    n_all = len(df)
    df = df[df.topic2.notna() & (df.supply_topic2 > 0)].copy()
    a, b = C.Y_WINDOW
    ty = pd.read_parquet(os.path.join(C.DATA_DIR, "topic_year.parquet"), columns=["topic_id", "year", "demand_followers_rule"])
    dem = []
    for lag in range(a, b + 1):
        t = ty.rename(columns={"topic_id": "topic2", "demand_followers_rule": f"d{lag}"})
        t["Y"] = t.year - lag
        dem.append(t.drop(columns="year").set_index(["topic2", "Y"]))
    dem = pd.concat(dem, axis=1).fillna(0).sum(axis=1).rename("demand_topic2").reset_index()
    df["topic2"] = df.topic2.astype(dem.topic2.dtype)
    df = df.merge(dem, on=["topic2", "Y"], how="left")
    df["l_sup2"] = np.log(df.supply_topic2)
    df["l_dem2"] = np.log1p(df.demand_topic2.fillna(0))
    df["t2s"] = df.topic2_score.fillna(0)

    base = [f"x_s2_{k}"] + M3.CONTROLS + ["t2s"]
    rows = []
    for name, extra, model, y, grp in [
        ("主设定", [], "pois", "y3", "0-6"),
        ("主设定", [], "ols", "ly3", "0-6"),
        ("诊断：控制第二方向需求", ["l_dem2"], "pois", "y3", "诊断"),
        ("诊断：控制第二方向需求", ["l_dem2"], "ols", "ly3", "诊断"),
        ("诊断：控制第二方向需求 + 发表后相似论文数", ["l_dem2", "l_after"], "pois", "y3", "诊断"),
    ]:
        fml = f"{y} ~ l_sup2 + {' + '.join(base + extra)} | ty"
        fit = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": "topic"})
        t = fit.tidy().loc["l_sup2"]
        rows.append({"group": grp, "spec": name, "model": "Poisson" if model == "pois" else "OLS", "gamma": t["Estimate"],
                     "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": fit._N})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, f"06_crossdomain{sfx}.csv"), index=False)
    m = res[res.group == "0-6"]
    passed = bool((m.gamma < 0).all() and (m.p < 0.05).all())
    md = ["# 检验 0-6（跨域）\n",
          f"有第二方向的焦点论文 {len(df):,} 篇（占 {len(df) / n_all:.1%}）。`gamma` = log(第二方向供给) 的系数。\n",
          res.round(4).to_markdown(index=False), "\n",
          f"**0-6 判定**：Poisson γ = {m.iloc[0].gamma:.4f}（p = {m.iloc[0].p:.2g}），OLS γ = {m.iloc[1].gamma:.4f}（p = {m.iloc[1].p:.2g}）；"
          f"**{'通过' if passed else '未通过'}**（标准：γ < 0 且显著）。\n"]
    open(os.path.join(C.RES_DIR, f"06_crossdomain{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"06 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
