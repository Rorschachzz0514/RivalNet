"""
【14】实验 0 v2 · 检验 0-4 v2（假对手）：在子课题 × 年份内打乱前一年相似论文数（预先登记见 实验0说明.md 7.4）

用途
  在每个子课题(K=2000) × 年份格子内随机打乱 log1p(P1)，重跑 0-2 v2 主设定（Poisson），重复 N_PLACEBO（200）次。
  通过标准：真实 β < 0 且落在置换分布 2.5%–97.5% 之外。
  多进程（spawn，每个子进程自己读数据），每完成一次追加写入 csv，中断后重跑会跳过已完成的编号。

输入
  results_v2/focal.parquet（【11】）

输出（results_v2/）
  14_placebo_draws.csv、14_placebo.csv、14_placebo.md、14_placebo.png

用法
  python 14_v2_test_placebo.py [--workers 64] [--test]

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import csv
import sys
import time
import warnings
from importlib import import_module
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

M13 = import_module("13_v2_test_micro")
warnings.filterwarnings("ignore")
X = f"x_m1_{C.V2_TAU}"
CELL = f"cy{C.V2_K}"
DF = None


def fit_beta(df):
    fml = f"y3 ~ {' + '.join([X] + M13.CONTROLS)} | {CELL} + pub_month + venue"
    return float(pf.fepois(fml, data=df, vcov={"CRV1": f"c{C.V2_K}"}).coef()[X])


def load(test):
    df = M13.prepare(test)
    return df[df.in_main].reset_index(drop=True)


def init_worker(test):
    global DF
    warnings.filterwarnings("ignore")
    DF = load(test)


def one(rep):
    rng = np.random.default_rng(20_000 + rep)
    df = DF.copy()
    key = rng.random(len(df))
    order = np.lexsort((key, df[CELL].to_numpy()))                  # 格内随机顺序
    base = np.argsort(df[CELL].to_numpy(), kind="stable")             # 格内原顺序
    v = df[X].to_numpy()
    new = np.empty_like(v)
    new[base] = v[order]
    df[X] = new
    return rep, fit_beta(df)


def main():
    global DF
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    sfx = "_test" if args.test else ""
    n_rep = 8 if args.test else C.N_PLACEBO
    DF = load(args.test)
    real = fit_beta(DF)
    print(f"真实 β = {real:.4f}", flush=True)
    draws_p = os.path.join(C.V2_RES, f"14_placebo_draws{sfx}.csv")
    done = set(pd.read_csv(draws_p).rep) if os.path.exists(draws_p) else set()
    if not done:
        with open(draws_p, "w", newline="") as fh:
            csv.writer(fh).writerow(["rep", "beta"])
    tasks = [r for r in range(n_rep) if r not in done]
    with get_context("spawn").Pool(min(args.workers, max(1, len(tasks))), initializer=init_worker, initargs=(args.test,)) as pool, \
            open(draws_p, "a", newline="") as fh:
        w = csv.writer(fh)
        for i, (r, b) in enumerate(pool.imap_unordered(one, tasks), 1):
            w.writerow([r, b])
            fh.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    b = pd.read_csv(draws_p).beta.to_numpy()
    lo, hi = np.quantile(b, [0.025, 0.975])
    p = (1 + np.sum(np.abs(b - b.mean()) >= abs(real - b.mean()))) / (1 + len(b))
    outside = bool(real < lo or real > hi)
    res = pd.DataFrame([{"real_beta": real, "n_draws": len(b), "placebo_mean": b.mean(), "placebo_sd": b.std(ddof=1),
                         "q025": lo, "q975": hi, "outside_95": outside, "perm_p": p, "passed": outside and real < 0}])
    res.to_csv(os.path.join(C.V2_RES, f"14_placebo{sfx}.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(b, bins=30, color="#999")
        ax.axvline(real, color="red")
        ax.set_title(f"placebo (v2): real={real:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(C.V2_RES, f"14_placebo{sfx}.png"), dpi=120)
    except Exception as e:
        print("画图失败:", e)
    md = ["# 检验 0-4 v2（假对手置换）\n",
          f"在子课题(K={C.V2_K}) × 年份内打乱 `log1p(P1)`，重复 {len(b)} 次。\n", res.round(4).to_markdown(index=False), "\n",
          f"**0-4 v2**：真实 β = {real:+.4f}，置换 95% 范围 [{lo:+.4f}, {hi:+.4f}]；"
          f"{'在范围外' if outside else '在范围内'}{'，但为正' if real > 0 else ''}；**{'通过' if outside and real < 0 else '未通过'}**。\n"]
    open(os.path.join(C.V2_RES, f"14_placebo{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"14 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
