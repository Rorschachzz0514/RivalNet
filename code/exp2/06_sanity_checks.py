"""
【06】实验 2 · 代码自查：确保模型结果的好坏来自模型本身，而不是代码错误

检查项
  C1 标签对齐     张量 y.npy 的每一行 = 样本表中同一 paper_id 的 (y1, y2, y3a)
  C2 特征对齐     x_paper 中 n_authors 列反标准化后 = log1p(样本表 n_authors)；参考文献特征列同理（v6 张量）
  C3 对手对齐     随机抽 2,000 篇：由向量重算焦点与对手的余弦相似度，与 r_feat 中（反标准化后的）相似度一致；
                  对手行号对应的 paper_id = rivals.parquet 中记录的 comp_id
  C4 切分隔离     训练用的格子全部来自训练集；每个格子内论文的 split 一致；标准化统计量只用训练集
  C5 指标一致     用保存的验证集预测独立重算 MALE，= summary.json 中的 best_val_MALE（直接输出模型，无随机性）
  C6 标签打乱     把训练集的被引在训练集内部随机打乱后训练：若无泄漏，验证 MALE 应接近"只猜平均水平"的傻模型（≈ 1.0）
  C7 可复现       同一配置同一种子训练两次，验证 MALE 相差 < 0.003（GPU 浮点运算有极小的非确定性）
  C8 选项生效     随机遮掉对手（rdrop）只在训练时生效：同一模型在评价模式下两次前向结果完全相同

输出
  results/06_sanity.md

用法
  python 06_sanity_checks.py --gpu 0
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

PY = "/path/to/conda/envs/mpcc/bin/python"
TD = C.TENSORS + "_v6"
L = lambda n: np.load(os.path.join(TD, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    res = []
    add = lambda k, ok, info: res.append({"检查": k, "通过": "✅" if ok else "❌", "说明": info})
    meta = pd.read_parquet(os.path.join(TD, "meta.parquet"))
    s = pd.read_parquet(C.SAMPLES).set_index("paper_id").reindex(meta.paper_id)

    # C1
    y = L("y.npy")
    ok = np.array_equal(y, s[["y1", "y2", "y3a"]].to_numpy(np.float32))
    add("C1 标签对齐", ok, f"{len(y):,} 行逐一比较")

    # C2
    names = json.load(open(os.path.join(TD, "feature_names.json")))["paper"]
    xp = L("x_paper.npy")
    tr = (meta.split == "train").to_numpy()
    j = names.index("n_authors")
    raw = np.log1p(s.n_authors.to_numpy(np.float64))
    back = xp[:, j] * raw[tr].std(ddof=1) + raw[tr].mean()
    ok2 = np.allclose(back[np.abs(xp[:, j]) < 7.9], raw[np.abs(xp[:, j]) < 7.9], atol=1e-3)
    rf_ = pd.read_parquet(os.path.join(os.path.dirname(C.SAMPLES), "ref_feats.parquet")).set_index("paper_id").reindex(meta.paper_id)
    j2 = names.index("ref_lc_mean")
    rr = rf_.ref_lc_mean.to_numpy(np.float64)
    m = ~np.isnan(rr)
    back2 = xp[m, j2] * np.nanstd(rr[tr], ddof=1) + np.nanmean(rr[tr])
    ok3 = np.allclose(back2[np.abs(xp[m, j2]) < 7.9], rr[m][np.abs(xp[m, j2]) < 7.9], atol=1e-3)
    add("C2 特征对齐", ok2 and ok3, "n_authors 与 ref_lc_mean 反标准化后与样本表一致")

    # C3
    E = np.load(C.EMB, mmap_mode="r")
    rrow, rft = L("r_row.npy"), L("r_feat.npy")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(meta), 2000, replace=False)
    em = meta.emb_row.to_numpy()
    a = np.asarray(E[em[pick]], dtype=np.float32)
    b = np.asarray(E[rrow[pick].ravel()], dtype=np.float32).reshape(2000, -1, 768)
    cos = np.einsum("nd,nkd->nk", a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=2, keepdims=True))
    rv = pd.read_parquet(C.RIVALS)
    sim_raw = rv.sim.to_numpy()
    s_mu, s_sd = None, None
    # r_feat 第 0 列 = (sim − μ_train) / σ_train；用同一篇论文的原始 sim 求回 μ、σ 后比较
    rvp = rv.set_index(["focal_id", "rank"]).sim
    raw_sim = np.stack([rvp.loc[pid].to_numpy() for pid in meta.paper_id.to_numpy()[pick]])
    # 由训练集全部对手的原始相似度求 μ、σ，直接反算标准化值（之前用 np.polyfit 拟合，数值条件差、斜率错误，导致误判）
    trp = set(meta.paper_id[tr])
    tr_sim = rv[rv.focal_id.isin(trp)].sim.to_numpy(np.float64)
    expect = (raw_sim - tr_sim.mean()) / tr_sim.std(ddof=1)
    std_err = np.abs(expect - rft[pick, :, 0].astype(np.float64)).max()
    ok4 = np.abs(cos - raw_sim).max() < 0.01 and std_err < 0.01
    emi = pd.read_parquet(C.EMB_META, columns=["paper_id", "row"]).set_index("row").paper_id
    comp = rv.set_index(["focal_id", "rank"]).comp_id
    ok5 = all(emi.loc[rrow[i, kk]] == comp.loc[(meta.paper_id.iloc[i], kk + 1)] for i in pick[:200] for kk in range(50))
    add("C3 对手对齐", ok4 and ok5, f"重算余弦与存储相似度最大差 {np.abs(cos - raw_sim).max():.4f}；标准化相似度最大差 {std_err:.4f}（float16）；对手编号 1 万对逐一核对")

    # C4
    csplit, ptr = L("cell_split.npy"), L("cell_ptr.npy")
    same = all(meta.split.iloc[ptr[c]:ptr[c + 1]].nunique() == 1 and meta.split.iloc[ptr[c]] == csplit[c] for c in range(len(csplit)))
    n_num = len(C.PAPER_NUM) + 8                                   # 数值特征列（其后是缺失指示与独热）
    trm = xp[tr, :n_num].mean(0)
    ok6 = same and np.abs(trm).max() < 0.05
    add("C4 切分隔离", ok6, f"{len(csplit):,} 个格子内 split 一致；训练集标准化后的均值最大偏离 {np.abs(trm).max():.3f}")

    # C5
    rows = []
    for sd in (1, 2):
        d = f"runs/AS_r01_0_s{sd}"
        if os.path.exists(os.path.join(C.HERE, d, "preds.parquet")):
            p = pd.read_parquet(os.path.join(C.HERE, d, "preds.parquet"))
            p = p[p.split == "val"].merge(s.reset_index()[["paper_id", "y3"]], on="paper_id")
            mm = np.abs(p.pred - np.log1p(p.y3)).mean()
            rows.append((mm, json.load(open(os.path.join(C.HERE, d, "summary.json")))["best_val_MALE"]))
    ok7 = rows and all(abs(a_ - b_) < 1e-4 for a_, b_ in rows)
    add("C5 指标一致", bool(ok7), "；".join(f"重算 {a_:.5f} vs 记录 {b_:.5f}" for a_, b_ in rows))

    # C6 + C7 + C8：调用 02 的训练（标签打乱 / 重复训练）
    os.makedirs(os.path.join(C.HERE, "autosearch", "configs"), exist_ok=True)
    base = json.load(open(os.path.join(C.HERE, "autosearch", "configs", "AS_r01_0.json")))
    json.dump({**base, "shuffle_train_y": True}, open(os.path.join(C.HERE, "autosearch", "configs", "SANITY_shuffle.json"), "w"))
    json.dump(base, open(os.path.join(C.HERE, "autosearch", "configs", "SANITY_rep.json"), "w"))
    json.dump({**base, "rdrop": 0.3}, open(os.path.join(C.HERE, "autosearch", "configs", "SANITY_rdrop.json"), "w"))
    for name in ("SANITY_shuffle", "SANITY_rep", "SANITY_rdrop"):
        subprocess.run(["rm", "-rf", os.path.join(C.HERE, "runs", f"{name}_s1")])
        subprocess.run([PY, "-u", os.path.join(C.HERE, "02_mpcnet.py"), "--configs", name, "--seeds", "1", "--gpus", args.gpu],
                       cwd=C.HERE, capture_output=True)
    sv = lambda n: json.load(open(os.path.join(C.HERE, "runs", f"{n}_s1", "summary.json")))["best_val_MALE"]
    sh, rep = sv("SANITY_shuffle"), sv("SANITY_rep")
    orig = json.load(open(os.path.join(C.HERE, "runs", "AS_r01_0_s1", "summary.json")))["best_val_MALE"]
    add("C6 标签打乱", sh > 0.95, f"打乱训练标签后验证 MALE = {sh:.4f}（正常训练 {orig:.4f}；只猜平均水平约 1.0）")
    add("C7 可复现", abs(rep - orig) < 0.003, f"同配置同种子两次：{orig:.4f} vs {rep:.4f}")
    pr = pd.read_parquet(os.path.join(C.HERE, "runs", "SANITY_rdrop_s1", "preds.parquet"))
    pr2 = os.path.join(C.HERE, "runs", "SANITY_rdrop_s1", "preds_eval_twice.parquet")
    ok8 = os.path.exists(pr2) and np.allclose(pd.read_parquet(pr2).pred.to_numpy(), pr[pr.split == "test"].pred.to_numpy())
    add("C8 选项生效", ok8, "rdrop = 0.3 的模型在评价模式下两次预测完全相同（遮挡只在训练时生效）")

    md = "# 实验 2 代码自查\n\n" + pd.DataFrame(res).to_markdown(index=False) + "\n"
    os.makedirs(C.RES_DIR, exist_ok=True)
    open(os.path.join(C.RES_DIR, "06_sanity.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
