"""
【02】检验 0-1（宏观）：方向内论文变多，是否摊薄每篇论文的被引

用途
  一行 = 一个 AI 方向 × 首次公开年份（2017–2021，共 385 格）。Poisson 回归
      cohort_y3 ~ b·log(supply) + c·log(demand) + 年份固定效应 [+ 方向固定效应]
  cohort_y3 = 这批论文第 1–3 个完整年份的被引之和；supply = 该格论文数；
  demand 两种口径：施引方规模（Y+1～Y+3 年该方向新论文的参考文献总数，主）、
                   净流入量（该方向全部论文在 Y+1～Y+3 年收到的被引，减去这批论文自身的被引）。
  解读：b = 1 表示论文之间互不影响（总被引与篇数同比例增长）；b = 0 表示完全零和（篇数增加不增加总被引）；
       0 < b < 1 表示部分竞争。通过标准：b < 1 且显著。标准误按方向聚类。

输入
  results/topic_cohort.parquet（【01】）

输出（results/）
  02_macro.csv   四个设定的系数、标准误、b=1 与 b=0 的检验
  02_macro.md    结果表与解读

用法
  python 02_test_macro.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C


def fit(df, demand_col, topic_fe):
    d = df[(df.supply > 0) & (df[demand_col] > 0)].copy()
    X = pd.DataFrame({"log_supply": np.log(d.supply), "log_demand": np.log(d[demand_col])})
    X = X.join(pd.get_dummies(d.Y, prefix="yr", drop_first=True, dtype=float))
    if topic_fe:
        X = X.join(pd.get_dummies(d.topic, prefix="tp", drop_first=True, dtype=float))
    X = sm.add_constant(X)
    m = sm.GLM(d.cohort_y3, X, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": d.topic.astype("category").cat.codes})
    b, se = m.params["log_supply"], m.bse["log_supply"]
    c, sec = m.params["log_demand"], m.bse["log_demand"]
    return {
        "demand": demand_col, "topic_fe": topic_fe, "n_cells": len(d),
        "b_supply": b, "se_b": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se,
        "p_b_lt_1": stats.norm.cdf((b - 1) / se),          # H0: b >= 1, 单侧
        "p_b_eq_0": 2 * (1 - stats.norm.cdf(abs(b / se))),
        "c_demand": c, "se_c": sec, "p_c_eq_0": 2 * (1 - stats.norm.cdf(abs(c / sec))),
        "pseudo_r2": 1 - m.deviance / m.null_deviance,
    }


def main():
    df = pd.read_parquet(os.path.join(C.RES_DIR, "topic_cohort.parquet"))
    rows = [fit(df, dc, fe) for dc in ("demand_followers", "demand_inflow_net") for fe in (False, True)]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "02_macro.csv"), index=False)
    main_row = res.iloc[0]
    passed = (main_row.b_supply < 1) and (main_row.p_b_lt_1 < 0.05)
    show = res[["demand", "topic_fe", "n_cells", "b_supply", "se_b", "ci_low", "ci_high", "p_b_lt_1", "p_b_eq_0",
                "c_demand", "se_c", "pseudo_r2"]].round(4)
    md = ["# 检验 0-1（宏观）\n",
          "Poisson：`cohort_y3 ~ b·log(supply) + c·log(demand) + 年份 FE [+ 方向 FE]`，标准误按方向聚类。\n",
          "b = 1：互不影响；b = 0：完全零和；0 < b < 1：部分竞争。\n",
          show.to_markdown(index=False), "\n",
          f"**主设定（施引方规模、年份 FE）**：b = {main_row.b_supply:.3f}（95% CI {main_row.ci_low:.3f}–{main_row.ci_high:.3f}），"
          f"H0: b ≥ 1 的单侧 p = {main_row.p_b_lt_1:.4f}；**{'通过' if passed else '未通过'}**（标准：b < 1 且显著）。\n"]
    open(os.path.join(C.RES_DIR, "02_macro.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
