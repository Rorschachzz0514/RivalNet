"""
【18】实验 0 v2 · 事后分析（不计入判定）：给定"浪潮"大小，进场越晚是否被引越少

用途
  0-2 v2 显示：即使在子课题 × 年份内，前一年相似论文越多，被引越多——相似论文的密度主要反映这一小片研究的热度。
  竞争更合理的度量是"在同一波浪潮中的位置"：浪潮大小一定时，前面的人越多（进场越晚），分到的越少。
  浪潮 = 焦点论文前一年、当年、后一年的高相似论文（SPECTER2 ≥ 0.95）。为避免"跟随者"（引用了焦点论文的后来者，
  数量受焦点论文质量影响）带来的偏差，当年与后一年只数**与焦点论文互不引用**的论文（independent）。
  模型（主分析集，固定效应与控制变量同 0-2 v2）：
    W1  y3 ~ log1p(前一年) + log1p(当年·独立) + log1p(后一年·独立) + 控制
    W2  y3 ~ 前一年占比 + log1p(浪潮大小)，浪潮大小 = 前一年 + 当年·独立 + 后一年·独立，只用浪潮大小 ≥ 1 的论文
    W3  W1 的"全部论文"版本（不剔除跟随者，与【13】事后诊断一致，作对照）
    W4  W1 加上"跟随者"数（后一年引用了焦点论文的相似论文数），看剔除跟随者后结论是否依赖这一项
  W1 的前一年系数做 200 次置换检验（子课题 × 年份内打乱）。
  **这是看到 0-2 v2 结果之后提出的分析**，结论需在实验 6（其他学科）中预先登记后重复验证。

输入
  results_v2/focal.parquet（【11】）、DATA/sim_pairs_95.parquet（【29】）

输出（results_v2/）
  18_wave.csv、18_wave_placebo.csv、18_wave.md

用法
  python 18_v2_posthoc_wave.py [--workers 64]

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import sys
import time
import warnings
from importlib import import_module
from multiprocessing import get_context

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

M13 = import_module("13_v2_test_micro")
warnings.filterwarnings("ignore")
CELL = f"cy{C.V2_K}"
DF = None


def load():
    df = M13.prepare()
    df = df[df.in_main].copy()
    con = duckdb.connect()
    w = con.execute(f"""select focal_id as paper_id,
            count(*) filter (where dyear = -1) as m1,
            count(*) filter (where dyear = 0 and not comp_cites_focal and not focal_cites_comp) as y0_ind,
            count(*) filter (where dyear = 1 and not comp_cites_focal and not focal_cites_comp) as p1_ind,
            count(*) filter (where dyear = 1 and comp_cites_focal) as p1_fol,
            count(*) filter (where dyear = 0) as y0_all, count(*) filter (where dyear = 1) as p1_all
        from read_parquet('{os.path.join(C.V2_DATA, 'sim_pairs_95.parquet')}') group by 1""").df()
    df = df.merge(w, on="paper_id", how="left")
    for c in ("m1", "y0_ind", "p1_ind", "p1_fol", "y0_all", "p1_all"):
        df[c] = df[c].fillna(0)
        df[f"l_{c}"] = np.log1p(df[c])
    diff = (df.m1 - df[f"n_ym1_ge{C.V2_TAU}"]).abs()
    # 【28】与【29】在相似度恰好贴着 0.95 时可能因 float16 舍入差 ±1（应用数学有 2 篇）；超过容差才报错
    assert diff.max() <= 1 and (diff > 0).mean() <= 0.001, f"前一年计数应与【28】一致：{int((diff > 0).sum())} 篇不一致，最大差 {diff.max()}"
    if (diff > 0).any():
        print(f"提示：{int((diff > 0).sum())} 篇论文的前一年计数与【28】差 1（float16 贴阈值舍入），本分析统一用逐对导出的计数", flush=True)
    df["wave"] = df.m1 + df.y0_ind + df.p1_ind
    df["l_wave"] = np.log1p(df.wave)
    df["share_before"] = np.where(df.wave > 0, df.m1 / df.wave.where(df.wave > 0, 1), np.nan)
    return df.reset_index(drop=True)


def fit(df, x, extra, y="y3", model="pois"):
    fml = f"{y} ~ {' + '.join([x] + extra + M13.CONTROLS)} | {CELL} + pub_month + venue"
    m = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": f"c{C.V2_K}"})
    t = m.tidy().loc[x]
    return t["Estimate"], t["Std. Error"], t["Pr(>|t|)"], m._N


def init_worker():
    global DF
    warnings.filterwarnings("ignore")
    DF = load()


def one(rep):
    rng = np.random.default_rng(30_000 + rep)
    df = DF.copy()
    order = np.lexsort((rng.random(len(df)), df[CELL].to_numpy()))
    base = np.argsort(df[CELL].to_numpy(), kind="stable")
    v = df.l_m1.to_numpy()
    new = np.empty_like(v)
    new[base] = v[order]
    df["l_m1"] = new
    return fit(df, "l_m1", ["l_y0_ind", "l_p1_ind"])[0]


def main():
    global DF
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    DF = load()
    rows = []
    specs = [
        ("W1 前一年（给定当年·独立、后一年·独立）", "l_m1", ["l_y0_ind", "l_p1_ind"], "y3", "pois"),
        ("W1 前一年（OLS）", "l_m1", ["l_y0_ind", "l_p1_ind"], "ly3", "ols"),
        ("W1 当年·独立", "l_y0_ind", ["l_m1", "l_p1_ind"], "y3", "pois"),
        ("W1 后一年·独立", "l_p1_ind", ["l_m1", "l_y0_ind"], "y3", "pois"),
        ("W2 前一年占比（给定浪潮大小）", "share_before", ["l_wave"], "y3", "pois"),
        ("W2 前一年占比（OLS）", "share_before", ["l_wave"], "ly3", "ols"),
        ("W3 前一年（给定当年·全部、后一年·全部）", "l_m1", ["l_y0_all", "l_p1_all"], "y3", "pois"),
        ("W4 前一年（W1 + 跟随者数）", "l_m1", ["l_y0_ind", "l_p1_ind", "l_p1_fol"], "y3", "pois"),
        ("W4 跟随者数", "l_p1_fol", ["l_m1", "l_y0_ind", "l_p1_ind"], "y3", "pois"),
    ]
    for name, x, extra, y, model in specs:
        sub = DF[DF.share_before.notna()] if x == "share_before" else DF
        b, se, p, n = fit(sub, x, extra, y, model)
        rows.append({"spec": name, "x": x, "model": "Poisson" if model == "pois" else "OLS", "beta": b, "se": se, "p": p, "n": n})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.V2_RES, "18_wave.csv"), index=False)
    print(res.round(4).to_string(), flush=True)

    with get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        draws = np.array(pool.map(one, range(C.N_PLACEBO)))
    real = res.iloc[0].beta
    lo, hi = np.quantile(draws, [0.025, 0.975])
    pl = pd.DataFrame([{"real_beta": real, "n_draws": len(draws), "placebo_mean": draws.mean(), "q025": lo, "q975": hi,
                        "outside_95": bool(real < lo or real > hi)}])
    pl.to_csv(os.path.join(C.V2_RES, "18_wave_placebo.csv"), index=False)
    share = lambda b: (np.exp(b) - 1) * 100
    md = ["# 事后分析：浪潮中的位置（不计入判定）\n",
          "浪潮 = 前一年、当年、后一年 SPECTER2 ≥ 0.95 的相似论文；当年与后一年只数与焦点论文互不引用的（排除跟随者）。"
          "主分析集，固定效应与控制变量同 0-2 v2。**看到 0-2 v2 结果之后提出，需在其他学科预先登记后重复验证。**\n",
          res.round(4).to_markdown(index=False), "\n",
          "W1 前一年系数的置换检验（子课题 × 年份内打乱 200 次）：\n", pl.round(4).to_markdown(index=False), "\n",
          f"解读：浪潮大小一定时，前一年的相似论文翻倍，被引变化 {(2 ** res.iloc[0].beta - 1) * 100:+.1f}%；"
          f"从\"浪潮里最早\"到\"全部对手都在前一年\"（占比 0 → 1），被引变化 {share(res.iloc[4].beta):+.1f}%。\n"]
    open(os.path.join(C.V2_RES, "18_wave.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"18 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
