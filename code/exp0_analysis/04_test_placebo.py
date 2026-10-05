"""
【04】检验 0-4（假对手）：把对手数在同一方向 × 年份内随机打乱，系数是否还在

用途
  在每个方向 × 年份格子内随机打乱竞争变量，再跑一遍回归，重复 N_PLACEBO（200）次，得到"假对手"系数的置换分布；
  真实系数落在置换分布的 95% 范围（2.5%–97.5% 分位）之外 → 效应不是巧合。
  两个系数各做一遍：
    main  0-2 主设定（Poisson，log1p(prior_s2_95) + 控制变量 | 方向×年份）——预设的判定对象
    D1    【03】诊断 D1（同上 + log1p(after_95)），只打乱 prior_s2_95，after_95 不动——事后追加，不计入判定
  多进程（spawn：每个子进程自己读数据；fork 会因主进程已初始化 numba 线程而死锁），
  每完成一次就追加写入 csv，中断后重跑会跳过已完成的编号。

输入
  results/focal.parquet（【01】）、DATA_DIR/sim_counts.parquet（D1 用）

输出（results/）
  04_placebo_draws.csv   每次置换的系数（spec, rep, beta）
  04_placebo.csv         真实系数、置换分布的均值 / 2.5% / 97.5% 分位、置换 p 值
  04_placebo.md          结果表与解读
  04_placebo.png         置换分布直方图与真实系数

用法
  python 04_test_placebo.py                 全量（200 次 × 2 个设定）
  python 04_test_placebo.py --test          只用 2019 年，每个设定 8 次（冒烟测试）
  python 04_test_placebo.py --workers 48

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
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C
from importlib import import_module

M3 = import_module("03_test_micro")
warnings.filterwarnings("ignore")
K = int(round(C.TAU * 100))
X = f"x_s2_{K}"
SPECS = {"main": [], "D1": ["l_after"]}
DF = None


def fit_beta(df, extra):
    fml = f"y3 ~ {' + '.join([X] + M3.CONTROLS + extra)} | ty"
    return float(pf.fepois(fml, data=df, vcov={"CRV1": "topic"}).coef()[X])


def init_worker(test):
    global DF
    warnings.filterwarnings("ignore")
    DF = M3.prepare(test)


def one(task):
    spec, rep = task
    rng = np.random.default_rng(10_000 + rep)
    df = DF.copy()
    # 在方向 x 年份格子内打乱竞争变量
    order = rng.permutation(len(df))
    perm = df.iloc[order].copy()
    perm["_r"] = rng.random(len(perm))
    perm = perm.sort_values(["ty", "_r"], kind="stable")
    base = df.sort_values("ty", kind="stable")
    shuffled = pd.Series(perm[X].to_numpy(), index=base.index)
    df[X] = shuffled.reindex(df.index).to_numpy()
    return spec, rep, fit_beta(df, SPECS[spec])


def main():
    global DF
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    DF = M3.prepare(args.test)
    n_rep = 8 if args.test else C.N_PLACEBO
    suffix = "_test" if args.test else ""
    draws_p = os.path.join(C.RES_DIR, f"04_placebo_draws{suffix}.csv")

    real = {s: fit_beta(DF, e) for s, e in SPECS.items()}
    print("真实系数:", real, flush=True)

    done = set()
    if os.path.exists(draws_p):
        old = pd.read_csv(draws_p)
        done = set(zip(old.spec, old.rep))
    else:
        with open(draws_p, "w", newline="") as fh:
            csv.writer(fh).writerow(["spec", "rep", "beta"])
    tasks = [(s, r) for s in SPECS for r in range(n_rep) if (s, r) not in done]
    print(f"待跑 {len(tasks)} 次（已完成 {len(done)}）", flush=True)

    with get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(args.test,)) as pool, open(draws_p, "a", newline="") as fh:
        w = csv.writer(fh)
        for i, (s, r, b) in enumerate(pool.imap_unordered(one, tasks), 1):
            w.writerow([s, r, b])
            fh.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)

    d = pd.read_csv(draws_p)
    rows = []
    for s in SPECS:
        b = d[d.spec == s].beta.to_numpy()
        lo, hi = np.quantile(b, [0.025, 0.975])
        p = (1 + np.sum(np.abs(b - b.mean()) >= abs(real[s] - b.mean()))) / (1 + len(b))
        rows.append({"spec": s, "real_beta": real[s], "n_draws": len(b), "placebo_mean": b.mean(),
                     "placebo_sd": b.std(ddof=1), "q025": lo, "q975": hi,
                     "outside_95": bool(real[s] < lo or real[s] > hi), "perm_p": p})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, f"04_placebo{suffix}.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SPECS), figsize=(10, 3.5))
        for ax, r in zip(axes, rows):
            ax.hist(d[d.spec == r["spec"]].beta, bins=30, color="#999")
            ax.axvline(r["real_beta"], color="red")
            ax.set_title(f"{r['spec']}: real={r['real_beta']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(C.RES_DIR, f"04_placebo{suffix}.png"), dpi=120)
    except Exception as e:  # 画图失败不影响结果
        print("画图失败:", e)

    m = res.set_index("spec")
    md = ["# 检验 0-4（假对手置换）\n",
          f"在方向 × 年份格子内打乱 `log1p(prior_s2_{K})`，每个设定重复 {n_rep} 次。"
          "main = 0-2 主设定（预设判定对象）；D1 = 加控制发表后相似论文数（事后诊断，不计入判定）。\n",
          res.round(4).to_markdown(index=False), "\n",
          f"**0-4 判定（main）**：真实系数 {m.loc['main', 'real_beta']:.4f}，置换分布 95% 范围 "
          f"[{m.loc['main', 'q025']:.4f}, {m.loc['main', 'q975']:.4f}]；"
          f"{'落在范围之外' if m.loc['main', 'outside_95'] else '落在范围之内'}。"
          f"{'但方向为正，不是竞争（被引更少）的证据。' if m.loc['main', 'real_beta'] > 0 else ''}\n",
          f"D1：真实系数 {m.loc['D1', 'real_beta']:.4f}，95% 范围 [{m.loc['D1', 'q025']:.4f}, {m.loc['D1', 'q975']:.4f}]。\n"]
    open(os.path.join(C.RES_DIR, f"04_placebo{suffix}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"04 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
