"""
【05】修改计划 A5 · MLP 与 LightGBM 在验证集上调参（原 02_models.py 中超参数固定）

用途
  与 02_models.py 相同的数据与特征（L4 = 元数据 + 热度 + SPECTER2 + 竞争计数；目标 log1p(y3)），只在验证集（2020）上选择超参数，
  测试集（2021）只评价一次：
    LightGBM：num_leaves ∈ {31, 127, 511} × learning_rate ∈ {0.02, 0.05} × min_data_in_leaf ∈ {20, 100, 500}（文本 = PCA-64），
              第一轮选中网格角点（511, 0.02, 500），故扩展：num_leaves ∈ {511, 1023} × min_data_in_leaf ∈ {1000, 2000}（学习率 0.02）
              以及 (1023, 0.02, 500)；选定后 3 个种子（bagging 种子）平均。
              （原计划的 PCA-256 文本输入对照因服务器过载、单次运行超过 2 小时而取消。）
    MLP：宽度 ∈ {512, 1024} × 隐藏层数 ∈ {2, 3} × dropout ∈ {0.1, 0.2, 0.3} × 学习率 ∈ {3e-4, 1e-3}；选定后 5 个种子平均。
  选择标准：验证集 MALE。

输出（results/）
  preds_tuned.parquet     paper_id, split（val/test）, model（"L4 +竞争（调参）" / "MLP-文本（调参）"）, pred
  05_tuning.csv           全部组合的验证集结果；05_tuning.md 速览

用法
  python 05_tune_baselines.py --gpu 0 [--jobs 6]
"""
import argparse
import os
import sys
import time
from importlib import util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_exp1 as C

spec = util.spec_from_file_location("m02", os.path.join(HERE, "02_models.py"))
M = util.module_from_spec(spec)
spec.loader.exec_module(M)
log = M.log


def male(p, s):
    d = p.merge(s[["paper_id", "ly3"]], on="paper_id")
    return d.groupby("split").apply(lambda g: (g.pred - g.ly3).abs().mean())


def fit_mlp(s, X, feats, gpu, width, depth, drop, lr, seed):
    import torch
    torch.manual_seed(seed)
    dev = torch.device(f"cuda:{gpu}")
    num = [c for c in feats if c not in C.CATEG]
    tr = (s.split == "train").to_numpy()
    A = s[num].astype(np.float32)
    miss = [c for c in num if A[c].isna().any()]
    Mi = A[miss].isna().astype(np.float32).add_suffix("_na")
    mu, sd = A[tr].mean(), A[tr].std().replace(0, 1)
    A = ((A - mu) / sd).fillna(0).clip(-10, 10)
    O = pd.get_dummies(s[C.CATEG], dtype=np.float32)
    Xn = np.hstack([X / np.linalg.norm(X, axis=1, keepdims=True), A.to_numpy(), Mi.to_numpy(), O.to_numpy()]).astype(np.float32)
    T = lambda a: torch.from_numpy(a).to(dev)
    Xt, yt = T(Xn), T(s.ly3.to_numpy(np.float32))
    itr, iva = np.where(tr)[0], np.where((s.split == "val").to_numpy())[0]
    layers, din = [], Xn.shape[1]
    for _ in range(depth):
        layers += [torch.nn.Linear(din, width), torch.nn.GELU(), torch.nn.Dropout(drop)]
        din = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(din, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(100):
        net.train()
        perm = itr[np.random.default_rng(seed * 1000 + ep).permutation(len(itr))]
        for b in range(0, len(perm), 1024):
            i = T(perm[b:b + 1024])
            loss = torch.nn.functional.mse_loss(net(Xt[i]).squeeze(1), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = (net(Xt[T(iva)]).squeeze(1) - yt[T(iva)]).abs().mean().item()      # 早停也用验证 MALE
        if v < best - 1e-4:
            best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    net.eval()
    ev = np.where(s.split.isin(["val", "test"]).to_numpy())[0]
    with torch.no_grad():
        pr = torch.cat([net(Xt[T(ev[b:b + 8192])]).squeeze(1) for b in range(0, len(ev), 8192)]).cpu().numpy()
    return pd.DataFrame({"paper_id": s.paper_id.iloc[ev].to_numpy(), "split": s.split.iloc[ev].to_numpy(), "pred": pr}), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    s, X = M.prepare(False)
    PCS = [f"pc{i}" for i in range(C.N_PCA)]
    L4 = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PCS + C.COMP
    rows, out = [], []

    # ---------------- LightGBM 网格（并行：每个任务 threads = 总线程 / jobs）
    from joblib import Parallel, delayed
    thr = max(4, C.THREADS // args.jobs)
    grid = [dict(num_leaves=nl, learning_rate=lr, min_data_in_leaf=md, num_threads=thr)
            for nl in (31, 127, 511) for lr in (0.02, 0.05) for md in (20, 100, 500)]
    grid += [dict(num_leaves=nl, learning_rate=0.02, min_data_in_leaf=md, num_threads=thr)
             for nl, md in ((511, 1000), (511, 2000), (1023, 500), (1023, 1000), (1023, 2000))]

    def run_lgb(p, feats, data, seed=C.SEED):
        q = dict(p, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        pr, it = M.fit_lgb(data, feats, "ly3", "lgb", q)
        return pr, it

    res = Parallel(n_jobs=args.jobs, backend="loky")(delayed(run_lgb)(p, L4, s) for p in grid)
    for p, (pr, it) in zip(grid, res):
        m = male(pr, s)
        rows.append({"model": "LightGBM", **{k: v for k, v in p.items() if k != "num_threads"}, "text": "pca64", "iters": it,
                     "val_MALE": m["val"]})
    best_lgb = min(zip(grid, res), key=lambda z: male(z[1][0], s)["val"])[0]
    log(f"LightGBM 网格完成，选定 {best_lgb}")
    use256, feats, data = False, L4, s
    seeds_lgb = Parallel(n_jobs=3, backend="loky")(delayed(run_lgb)(dict(best_lgb, num_threads=C.THREADS // 3), feats, data, sd)
                                                   for sd in (C.SEED, C.SEED + 1, C.SEED + 2))
    pl = pd.concat([x[0].set_index(["paper_id", "split"]).pred for x in seeds_lgb], axis=1).mean(1).rename("pred").reset_index()
    out.append(pl.assign(model="L4 +竞争（调参）"))

    # ---------------- MLP 网格（GPU，顺序）
    mgrid = [(w, dp, dr, lr) for w in (512, 1024) for dp in (2, 3) for dr in (0.1, 0.2, 0.3) for lr in (3e-4, 1e-3)]
    best_m = None
    for w, dp, dr, lr in mgrid:
        pr, v = fit_mlp(s, X, L4, args.gpu, w, dp, dr, lr, C.SEED)
        mv = male(pr, s)["val"]
        rows.append({"model": "MLP", "width": w, "depth": dp, "dropout": dr, "lr": lr, "val_MALE": mv})
        log(f"MLP w={w} d={dp} drop={dr} lr={lr}: val MALE {mv:.4f}")
        if best_m is None or mv < best_m[0]:
            best_m = (mv, (w, dp, dr, lr))
    w, dp, dr, lr = best_m[1]
    log(f"MLP 选定 {best_m[1]}（val MALE {best_m[0]:.4f}），训练 5 个种子")
    ps = [fit_mlp(s, X, L4, args.gpu, w, dp, dr, lr, C.SEED + k)[0].set_index(["paper_id", "split"]).pred for k in range(5)]
    out.append(pd.concat(ps, axis=1).mean(1).rename("pred").reset_index().assign(model="MLP-文本（调参）"))

    po = pd.concat(out, ignore_index=True)[["paper_id", "split", "model", "pred"]]
    po.to_parquet(os.path.join(C.RES_DIR, "preds_tuned.parquet"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(C.RES_DIR, "05_tuning.csv"), index=False)
    fin = po.merge(s[["paper_id", "ly3"]], on="paper_id").groupby(["model", "split"]).apply(lambda g: (g.pred - g.ly3).abs().mean()).unstack()
    md = ["# 实验 1 · 05 基线调参（修改计划 A5）\n", f"LightGBM 选定：{best_lgb}；文本 = {'PCA-256' if use256 else 'PCA-64'}；3 个种子平均。",
          f"MLP 选定：宽度 {w}、{dp} 层、dropout {dr}、学习率 {lr}；5 个种子平均。\n", "## 调参后（MALE）\n", fin.round(4).to_markdown(), "\n",
          "## 全部组合（验证集）\n", tab.round(4).to_markdown(index=False)]
    open(os.path.join(C.RES_DIR, "05_tuning.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:6]))
    log(f"05 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
