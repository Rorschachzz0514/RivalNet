"""
【17】修改计划 A1 · kNN 检索基线（回应"对手模块只是检索增强"）

用途
  对每篇验证 / 测试论文，用 SPECTER2 向量（余弦）在检索池中找最相似的 k 篇论文，
  用它们真实 log(1+y) 的（等权或相似度 softmax 加权）平均作为预测。
    检索池（与 RivalNet 一致）：A 池 = 训练集（2017–2019）；B 池 = 训练集 + 验证集（2017–2020）。
    k ∈ {5, 10, 20, 50, 100, 200, 500, 1000}、加权方式 / 温度只在验证集上（A 池）选；测试集只评价一次。
    预先规定的终版 kNN = A 池与 B 池预测的平均（对应 RivalNet 终版 = A 组与 B 组集成的平均）。
  两种被引口径：all（全部来源被引，主表）、dp（语料内被引，DPPDCC 口径表）。
  检索增强 MLP（口径 all）：在验证集上用最小二乘拟合 pred = a·MLP + b·kNN + c（只有 3 个系数），测试集只评价一次；
    MLP = 主表中的 MLP 基线（实验 1 的 MLP-文本；若有修改计划 A5 的调参版本则用调参版本），kNN = 上面选定的 kNN（验证集用 A 池，测试集用 A、B 池平均）。
  注意：检索到的是"过去论文的最终被引"，与 RivalNet 的对手（同期论文、只用 T 时可见信息）不同。

输出
  /path/to/mpcc/baselines/knn/preds_{all,dp}.parquet（paper_id, split, seed, pred；seed=0）
  /path/to/mpcc/baselines/knn_mlp/preds_all.parquet（检索增强 MLP）
  /path/to/mpcc/baselines/knn/selection.csv（验证集上的全部组合）、README.md

用法
  CUDA_VISIBLE_DEVICES=4 python 17_knn_baseline.py
"""
import os
import sys
from importlib import util

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
OUT = "/path/to/mpcc/baselines/knn"
KS = [5, 10, 20, 50, 100, 200, 500, 1000]
TEMPS = [None, 10.0, 20.0, 50.0]          # None = 等权；否则 softmax(temp · cos)


def topk_search(Q, P, kmax, bs=4096):
    """Q (nq, 768)、P (np, 768) 已归一化 → (相似度, 下标)，各 (nq, kmax)"""
    sims, inds = [], []
    for b in range(0, len(Q), bs):
        s = Q[b:b + bs] @ P.T
        v, i = s.topk(kmax, dim=1)
        sims.append(v.float())
        inds.append(i)
    return torch.cat(sims), torch.cat(inds)


def predict(sim, ind, ly_pool, k, temp):
    s, i = sim[:, :k], ind[:, :k]
    yv = ly_pool[i]
    if temp is None:
        return yv.mean(1)
    w = torch.softmax(temp * s, 1)
    return (w * yv).sum(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    meta = pd.read_parquet(os.path.join(C.TENSORS + "_v6", "meta.parquet"))
    E = torch.from_numpy(np.load(C.EMB)).to(dev)
    X = torch.nn.functional.normalize(E[torch.from_numpy(meta.emb_row.to_numpy()).to(dev)].float(), dim=1).half()
    del E
    split = meta.split.to_numpy()
    s_all = pd.read_parquet(C.SAMPLES, columns=["paper_id", "y3", "eval_c2000"]).set_index("paper_id").reindex(meta.paper_id)
    targets = {"all": s_all.y3.to_numpy(np.float64),
               "dp": np.load(os.path.join(C.TENSORS + "_v6dp", "y.npy")).sum(1).astype(np.float64)}
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    pools = {"A": tr, "B": np.concatenate([tr, va])}
    kmax = max(KS)
    print("检索 …", flush=True)
    srch = {("val", "A"): topk_search(X[va], X[tr], kmax),
            ("test", "A"): topk_search(X[te], X[tr], kmax),
            ("test", "B"): topk_search(X[te], X[pools["B"]], kmax)}
    sel_rows, readme = [], ["# kNN 检索基线（修改计划 A1）\n", __doc__.split("输出")[0].split("用途")[1].strip(), ""]
    for tg, yraw in targets.items():
        ly = torch.from_numpy(np.log1p(yraw)).to(dev)
        # ---- 验证集选 k / 温度
        best = None
        sv, iv = srch[("val", "A")]
        for k in KS:
            for temp in TEMPS:
                p = predict(sv, iv, ly[torch.from_numpy(tr).to(dev)], k, temp).cpu().numpy()
                d = meta.iloc[va][["paper_id", "topic", "eval_c2000"]].assign(pred=p, y=yraw[va], ly=np.log1p(yraw[va]))
                m = EV.metrics(d)
                sel_rows.append({"target": tg, "k": k, "temp": temp if temp else "uniform", "val_MALE": m["MALE"],
                                 "val_rho_sub": m["Spearman_子课题内"]})
                if best is None or m["MALE"] < best[0]:
                    best = (m["MALE"], k, temp, m["Spearman_子课题内"])
        _, k, temp, rho = best
        print(f"[{tg}] 验证集选定 k={k}, temp={temp or 'uniform'}：MALE={best[0]:.4f}, ρ_sub={rho:.4f}", flush=True)
        readme.append(f"- 口径 {tg}：验证集选定 k = {k}，加权 = {temp or '等权'}（验证 MALE {best[0]:.4f}，ρ_sub {rho:.4f}）")
        # ---- 预测：验证集（A 池）、测试集（A、B 池平均）
        pv = predict(sv, iv, ly[torch.from_numpy(tr).to(dev)], k, temp).cpu().numpy()
        pa = predict(*srch[("test", "A")], ly[torch.from_numpy(pools["A"]).to(dev)], k, temp).cpu().numpy()
        pb = predict(*srch[("test", "B")], ly[torch.from_numpy(pools["B"]).to(dev)], k, temp).cpu().numpy()
        out = pd.concat([pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[va], "split": "val", "seed": 0, "pred": pv}),
                         pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[te], "split": "test", "seed": 0, "pred": (pa + pb) / 2})])
        out.to_parquet(os.path.join(OUT, f"preds_{tg}.parquet"), index=False)
        # 变体（只存档，不进主表）
        pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[te], "pred_A": pa, "pred_B": pb}).to_parquet(
            os.path.join(OUT, f"variants_{tg}.parquet"), index=False)
        for lab, p in (("A 池", pa), ("B 池", pb), ("A+B 平均（终版）", (pa + pb) / 2)):
            d = meta.iloc[te][["paper_id", "topic", "eval_c2000"]].assign(pred=p, y=yraw[te], ly=np.log1p(yraw[te]))
            m = EV.metrics(d)
            line = f"  测试 {lab}：MALE={m['MALE']:.4f}, RMSLE={m['RMSLE']:.4f}, ρ_sub={m['Spearman_子课题内']:.4f}, NDCG={m['NDCG@10_子课题内']:.4f}, ρ_all={m['Spearman_全体']:.4f}"
            print(f"[{tg}]" + line, flush=True)
            readme.append(line)
        if tg == "all":                                    # 检索增强 MLP：验证集线性叠加
            tf = os.path.join(C.EXP1_RES, "preds_tuned.parquet")    # 主表中的 MLP = 修改计划 A5 调参后的版本（若已有）
            tl = os.path.join(C.EXP1_RES, "preds_main_mlp_log.parquet")   # A4 诊断后：计数特征取 log1p（验证集更好，主表改用）
            if os.path.exists(tl):
                pc, mname = pd.read_parquet(tl), "MLP-文本（调参，计数取对数）"
            elif os.path.exists(tf):
                pc, mname = pd.read_parquet(tf), "MLP-文本（调参）"
            else:
                pc, mname = pd.read_parquet(os.path.join(C.EXP1_RES, "preds_cold.parquet")), "MLP-文本"
            print(f"检索增强 MLP 使用：{mname}", flush=True)
            mlp = pc[pc.model == mname].set_index(["split", "paper_id"]).pred
            mv = mlp.loc["val"].reindex(meta.paper_id.to_numpy()[va]).to_numpy()
            mt = mlp.loc["test"].reindex(meta.paper_id.to_numpy()[te]).to_numpy()
            assert not (np.isnan(mv).any() or np.isnan(mt).any()), "MLP 预测缺失"
            Xv = np.stack([mv, pv, np.ones_like(pv)], 1)
            coef = np.linalg.lstsq(Xv, np.log1p(yraw[va]), rcond=None)[0]
            ps_v, ps_t = Xv @ coef, np.stack([mt, (pa + pb) / 2, np.ones_like(mt)], 1) @ coef
            os.makedirs(OUT + "_mlp", exist_ok=True)
            pd.concat([pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[va], "split": "val", "seed": 0, "pred": ps_v}),
                       pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[te], "split": "test", "seed": 0, "pred": ps_t})]
                      ).to_parquet(os.path.join(OUT + "_mlp", "preds_all.parquet"), index=False)
            for lab, p, idx_ in (("验证（拟合用）", ps_v, va), ("测试", ps_t, te), ("测试·MLP 单独", mt, te)):
                d = meta.iloc[idx_][["paper_id", "topic", "eval_c2000"]].assign(pred=p, y=yraw[idx_], ly=np.log1p(yraw[idx_]))
                m = EV.metrics(d)
                line = (f"  检索增强 MLP {lab}：系数 a={coef[0]:.3f}, b={coef[1]:.3f}, c={coef[2]:.3f}；MALE={m['MALE']:.4f}, "
                        f"RMSLE={m['RMSLE']:.4f}, ρ_sub={m['Spearman_子课题内']:.4f}, NDCG={m['NDCG@10_子课题内']:.4f}, ρ_all={m['Spearman_全体']:.4f}")
                print(line, flush=True)
                readme.append(line)
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "selection.csv"), index=False)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(readme) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
