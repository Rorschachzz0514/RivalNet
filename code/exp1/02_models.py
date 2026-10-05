"""
【02】实验 1 · 傻模型与学习模型（预先登记见 实验1说明.md 第 3 节）

用途
  冷启动（目标 log1p(y3)）：
    D0 全局延续；D1-方向 / D1-子课题 / D1-方向+子课题（LightGBM，只用组层面历史特征 → 组内预测相同）；
    D1*-方向 / D1*-子课题（测试组真实均值，上帝视角；主版本为留一均值，另存含自身的版本）；D3（D1-方向+子课题 + N(0, 0.3²) 噪声）；
    L1 元数据；L2 + 热度；L3 + 文本（SPECTER2 PCA-64，只在训练集拟合）；L4 + 竞争；MLP-文本（GPU）。
  发表 1 年后（目标 log1p(y2 + y3a)）：
    D2-方向 / D2-子课题 = log1p(y1) + 该组 Y−2 队列 [log1p(y2+y3a) − log1p(y1)] 的平均；L-after = L4 特征 + log1p(y1) + log1p(y0)。
  LightGBM：L2 损失，num_leaves 127，学习率 0.05，最多 5000 轮，验证集早停 200 轮，超参数固定。
  MLP：768 维向量 + 标准化数值特征（缺失填 0 并加缺失指示）+ 类别独热，3×512，dropout 0.2，AdamW，验证集早停。

输入
  config_exp1.SAMPLES、config_exp1.EMB

输出（results/）
  preds_cold.parquet      paper_id, split（val/test）, model, pred（log1p 空间）
  preds_after1y.parquet   同上（目标 log1p(y2 + y3a)）
  02_importance.csv       各 LightGBM 模型的特征重要性（gain）
  02_models.md            各模型在验证 / 测试集上的 MALE 速览与最佳轮数

用法
  python 02_models.py [--gpu 0] [--test]      --test：每个切分只抽 5,000 篇，快速检查流程
"""
import argparse
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp1 as C

warnings.filterwarnings("ignore")


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def prepare(test=False):
    s = pd.read_parquet(C.SAMPLES)
    if test:
        s = s.groupby("split", group_keys=False).sample(n=5000, random_state=0).reset_index(drop=True)
    sl = lambda a, b: np.log((s[a] + 1) / (s[b] + 1))
    s["g_topic_inflow"], s["g_topic_supply"] = sl("topic_inflow_Y", "topic_inflow_Ym2"), sl("topic_supply_Y", "topic_supply_Ym2")
    s["g_sub_inflow"], s["g_sub_supply"] = sl("sub_inflow_Y", "sub_inflow_Ym2"), sl("sub_supply_Y", "sub_supply_Ym2")
    for c in ["any_oa", "has_abstract", "has_funding", "has_preprint_at_T"]:
        s[c] = s[c].astype(float)
    for c in C.CATEG:
        s[c] = s[c].astype("category")
    s["ly3"] = np.log1p(s.y3)
    s["ly1"], s["ly0"] = np.log1p(s.y1), np.log1p(s.y0)
    s["ly23"] = np.log1p(s.y2 + s.y3a)
    # SPECTER2 主成分（只在训练集拟合）
    from sklearn.decomposition import PCA
    E = np.load(C.EMB, mmap_mode="r")
    X = np.asarray(E[s.emb_row.to_numpy()], dtype=np.float32)
    tr = (s.split == "train").to_numpy()
    pca = PCA(n_components=C.N_PCA, svd_solver="randomized", random_state=C.SEED).fit(X[tr])
    P = pca.transform(X)
    for i in range(C.N_PCA):
        s[f"pc{i}"] = P[:, i].astype(np.float32)
    log(f"PCA-{C.N_PCA} 解释方差 {pca.explained_variance_ratio_.sum():.3f}")
    return s, X


def fit_lgb(s, feats, target, name, params=None, imp=None):
    p = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=C.THREADS, verbose=-1, seed=C.SEED)
    p.update(params or {})
    tr, va = s[s.split == "train"], s[s.split == "val"]
    dtr = lgb.Dataset(tr[feats], tr[target], categorical_feature=[c for c in feats if c in C.CATEG], free_raw_data=False)
    dva = lgb.Dataset(va[feats], va[target], reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
    if imp is not None:
        imp.append(pd.DataFrame({"model": name, "feature": feats, "gain": m.feature_importance("gain")}))
    ev = s[s.split.isin(["val", "test"])]
    return pd.DataFrame({"paper_id": ev.paper_id, "split": ev.split, "model": name, "pred": m.predict(ev[feats], num_iteration=m.best_iteration)}), m.best_iteration


def fit_mlp(s, X, feats, target, name, gpu):
    import torch
    torch.manual_seed(C.SEED)
    dev = torch.device(f"cuda:{gpu}")
    num = [c for c in feats if c not in C.CATEG]
    tr = (s.split == "train").to_numpy()
    A = s[num].astype(np.float32)
    miss = [c for c in num if A[c].isna().any()]
    M = A[miss].isna().astype(np.float32).add_suffix("_na")
    mu, sd = A[tr].mean(), A[tr].std().replace(0, 1)
    A = ((A - mu) / sd).fillna(0).clip(-10, 10)
    O = pd.get_dummies(s[C.CATEG], dtype=np.float32)
    Xn = np.hstack([X / np.linalg.norm(X, axis=1, keepdims=True), A.to_numpy(), M.to_numpy(), O.to_numpy()]).astype(np.float32)
    y = s[target].to_numpy(np.float32)
    T = lambda a: torch.from_numpy(a).to(dev)
    Xt, yt = T(Xn), T(y)
    itr, iva = np.where(tr)[0], np.where((s.split == "val").to_numpy())[0]
    net = torch.nn.Sequential(torch.nn.Linear(Xn.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(100):
        net.train()
        perm = itr[np.random.default_rng(ep).permutation(len(itr))]
        for b in range(0, len(perm), 1024):
            i = T(perm[b:b + 1024])
            loss = torch.nn.functional.mse_loss(net(Xt[i]).squeeze(1), yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(Xt[T(iva)]).squeeze(1), yt[T(iva)]).item()
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
    log(f"{name}: {ep + 1} 轮，验证 MSE {best:.4f}")
    return pd.DataFrame({"paper_id": s.paper_id.iloc[ev].to_numpy(), "split": s.split.iloc[ev].to_numpy(), "model": name, "pred": pr}), ep + 1


def group_shift(s, key, lag, diff_col):
    """组 g、年份 Y 的值 = 该组 Y−lag 队列 diff_col 的平均（训练外的组用全局平均）。"""
    src = s[["Y", key, diff_col]].copy()
    src["Yt"] = src.Y + lag
    gm = src.groupby([key, "Yt"])[diff_col].mean().rename("gm")
    am = src.groupby("Yt")[diff_col].mean().rename("am")
    out = s[[key, "Y"]].merge(gm, left_on=[key, "Y"], right_index=True, how="left").merge(am, left_on="Y", right_index=True, how="left")
    return out.gm.fillna(out.am).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(C.RES_DIR, exist_ok=True)
    sfx = "_test" if args.test else ""
    s, X = prepare(args.test)
    log(f"样本 {len(s):,}")
    PCS = [f"pc{i}" for i in range(C.N_PCA)]
    preds, info, imp = [], [], []
    ev = s[s.split.isin(["val", "test"])]

    # ---------- 冷启动：傻模型
    d0 = s.groupby("Y").ly3.mean()
    preds.append(pd.DataFrame({"paper_id": ev.paper_id, "split": ev.split, "model": "D0 全局延续",
                               "pred": (ev.Y - 3).map(d0).to_numpy()}))
    grp = {"D1-方向": C.HEAT_TOPIC, "D1-子课题": C.HEAT_SUB, "D1-方向+子课题": C.HEAT_TOPIC + C.HEAT_SUB}
    for name, feats in grp.items():
        p, it = fit_lgb(s, feats, "ly3", name, dict(num_leaves=31, min_data_in_leaf=500), imp)
        preds.append(p)
        info.append((name, it))
        if name == "D1-方向+子课题":
            rng = np.random.default_rng(C.SEED)
            preds.append(p.assign(model="D3 热度 + 随机扰动", pred=p.pred + rng.normal(0, 0.3, len(p))))
    # 上帝视角：组内真实均值。主版本用"留一"均值（同组其他论文的平均，避免把自己的答案算进去）；含自身的版本作对照
    for name, key in (("D1*-方向（上帝视角）", "topic"), ("D1*-子课题（上帝视角）", "eval_c2000")):
        g = ev.groupby(["split", key]).ly3
        sm, n = g.transform("sum"), g.transform("count")
        loo = ((sm - ev.ly3) / (n - 1)).where(n > 1, ev.groupby("split").ly3.transform("mean"))
        preds.append(pd.DataFrame({"paper_id": ev.paper_id, "split": ev.split, "model": name, "pred": loo.to_numpy()}))
        preds.append(pd.DataFrame({"paper_id": ev.paper_id, "split": ev.split, "model": name.replace("）", "，含自身）"),
                                   "pred": g.transform("mean").to_numpy()}))
    log("傻模型完成")

    # ---------- 冷启动：学习模型
    L1 = C.META
    L2 = L1 + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2
    L3 = L2 + PCS
    L4 = L3 + C.COMP
    for name, feats in (("L1 元数据", L1), ("L2 元数据+热度", L2), ("L3 +文本", L3), ("L4 +竞争", L4)):
        p, it = fit_lgb(s, feats, "ly3", name, imp=imp)
        preds.append(p)
        info.append((name, it))
        log(f"{name} 完成（{it} 轮）")
    p, ep = fit_mlp(s, X, L4, "ly3", "MLP-文本", args.gpu)
    preds.append(p)
    info.append(("MLP-文本", ep))
    pc = pd.concat(preds, ignore_index=True)
    pc.to_parquet(os.path.join(C.RES_DIR, f"preds_cold{sfx}.parquet"), index=False)

    # ---------- 发表 1 年后
    pa = []
    s["dlt"] = s.ly23 - s.ly1
    for name, key in (("D2-方向", "topic"), ("D2-子课题", "ct2000")):
        sh = group_shift(s, key, 2, "dlt")
        pr = s.ly1.to_numpy() + sh
        m = s.split.isin(["val", "test"]).to_numpy()
        pa.append(pd.DataFrame({"paper_id": s.paper_id[m], "split": s.split[m], "model": name, "pred": pr[m]}))
    p, it = fit_lgb(s, L4 + ["ly1", "ly0"], "ly23", "L-after（L4 + 第一年被引）", imp=imp)
    pa.append(p)
    info.append(("L-after", it))
    pa = pd.concat(pa, ignore_index=True)
    pa.to_parquet(os.path.join(C.RES_DIR, f"preds_after1y{sfx}.parquet"), index=False)
    pd.concat(imp).to_csv(os.path.join(C.RES_DIR, f"02_importance{sfx}.csv"), index=False)

    # ---------- 速览
    def male(pred, target):
        d = pred.merge(s[["paper_id", target]], on="paper_id")
        return d.assign(e=(d.pred - d[target]).abs()).groupby(["model", "split"]).e.mean().unstack()
    md = ["# 实验 1 · 02 模型速览（MALE，log1p 空间）\n", "## 冷启动（y3）\n", male(pc, "ly3").round(4).to_markdown(), "\n",
          "## 发表 1 年后（y2 + y3a）\n", male(pa, "ly23").round(4).to_markdown(), "\n",
          "## 训练轮数\n", pd.DataFrame(info, columns=["model", "best_iter_or_epochs"]).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, f"02_models{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    log(f"02 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
