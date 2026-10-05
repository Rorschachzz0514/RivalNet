"""
【07】实验 2 · 集成：多个 MPC-Net 配置与种子的平均 + 在验证集上学非负权重与对比方法融合；测试集只在最后评价一次

用途
  1. 种子平均：每个入选配置把各种子的预测取平均；
  2. 配置集成：入选配置的种子平均再取平均（等权）；
  3. 融合（stacking）：在验证集（2020）上用非负最小二乘学权重 w ≥ 0 与截距，组合 [MPC-Net 集成, MLP-文本, LightGBM L4]，
     目标 log1p(y3)；权重只在验证集上学，然后原样用于测试集（2021）。
  评价指标与实验 1、2 完全相同（调用实验 1【04】的指标函数），并给出相对最好对比方法的差异（按子课题重抽样 1,000 次）。

输入
  runs/<配置>_s<种子>/preds.parquet；/path/to/mpcc/exp1/results/preds_cold.parquet；样本表
输出（results/）
  07_ensemble.csv、07_ensemble_weights.json、实验2结果_集成.md

用法
  python 07_ensemble.py --configs AS_r04_3,AS_r04_4,...
"""
import argparse
import glob
import json
import os
import sys
from importlib import util

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)


def seed_avg(cfg):
    fs = sorted(glob.glob(os.path.join(C.RUNS, f"{cfg}_s*", "preds.parquet")))
    p = pd.concat([pd.read_parquet(f, columns=["paper_id", "split", "pred"]) for f in fs])
    return p.groupby(["split", "paper_id"]).pred.mean(), len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    args = ap.parse_args()
    cfgs = args.configs.split(",")
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
    s = s[s.split.isin(["val", "test"])].set_index(["split", "paper_id"])
    cols, nseeds = {}, {}
    for c in cfgs:
        cols[c], nseeds[c] = seed_avg(c)
    P = pd.DataFrame(cols).reindex(s.index)
    P["MPC-Net 集成"] = P[cfgs].mean(1)
    pc = pd.read_parquet(os.path.join(C.EXP1_RES, "preds_cold.parquet"))
    for m in ("MLP-文本", "L4 +竞争"):
        P[m] = pc[pc.model == m].set_index(["split", "paper_id"]).pred.reindex(s.index)
    y = np.log1p(s.y3)
    base = ["MPC-Net 集成", "MLP-文本", "L4 +竞争"]
    v = P.index.get_level_values(0) == "val"
    Xv = np.column_stack([P.loc[v, base].to_numpy(), np.ones(v.sum())])
    # 非负最小二乘：截距放在最后一列，允许为负（用正负两列表示）
    Xv2 = np.column_stack([Xv, -np.ones(v.sum())])
    w, _ = nnls(Xv2, y[v].to_numpy())
    wts = dict(zip(base + ["截距+", "截距-"], map(float, w)))
    P["融合（验证集学权重）"] = np.column_stack([P[base].to_numpy(), np.ones(len(P)), -np.ones(len(P))]) @ w
    json.dump({"weights": wts, "configs": cfgs, "n_seeds": nseeds}, open(os.path.join(C.RES_DIR, "07_ensemble_weights.json"), "w"),
              ensure_ascii=False, indent=1)
    rows = []
    for sp in ("val", "test"):
        for m in cfgs + base + ["融合（验证集学权重）"]:
            d = P.loc[sp, [m]].rename(columns={m: "pred"}).join(s.loc[sp]).reset_index()
            d = d.assign(y=d.y3, ly=np.log1p(d.y3))
            rows.append({"split": sp, "model": m, **EV.metrics(d)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "07_ensemble.csv"), index=False)
    # 测试集：融合 vs 最好的对比方法（MLP-文本），按子课题重抽样
    t = P.loc["test"].join(s.loc["test"])
    t["ly"] = np.log1p(t.y3)
    sub = np.sort(t.eval_c2000.unique())
    rng = np.random.default_rng(2026)
    W = rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=1000).astype(float)
    def agg(col):
        g = t.assign(ae=(t[col] - t.ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "size")).reindex(sub).fillna(0)
        return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy())
    comp = []
    for m in ("MPC-Net 集成", "融合（验证集学权重）"):
        dd = agg(m) - agg("MLP-文本")
        comp.append({"比较": f"{m} − MLP-文本（MALE）", "差": dd.mean(), "95% 区间": f"[{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]"})
    show = res[res.model.isin(base + ["融合（验证集学权重）"])][["split", "model", "MALE", "RMSLE", "Spearman_子课题内", "NDCG@10_子课题内", "份额L1_子课题内"]]
    md = ["# 实验 2 · 集成与融合\n", f"入选配置（各自种子数）：{nseeds}\n", f"融合权重（只在验证集上学）：{ {k: round(v_, 4) for k, v_ in wts.items()} }\n",
          show.round(4).to_markdown(index=False), "\n", pd.DataFrame(comp).round(4).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, "实验2结果_集成.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
