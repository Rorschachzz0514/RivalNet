"""
【01】实验 2 · 组装训练张量：论文特征、竞争集合、格子（训练期子课题 × 年份）特征与目标

用途
  一次性把样本表与竞争集合整理成定长 numpy 数组，训练时整块放进 GPU。
    论文顺序：按 (split, 格子) 排序，同一格子的论文连续存放，便于按格子组批。
    论文特征：config_exp2.PAPER_NUM（计数类取 log1p）用训练集均值 / 标准差标准化，缺失填 0 并加缺失指示；PAPER_CAT 独热。
    竞争集合：每篇 50 个对手的向量行号 + 关系特征（相似度、年份差独热、天数差 / 365 及缺失指示、log1p 对手被引、
              渠道独热、共同作者、文献耦合、焦点引用对手、对手引用焦点）。
    格子：同一训练期子课题(ct2000) × 年份的全部样本论文。格子特征 = log1p(格内论文数) + 子课题 / 方向历史（取格内均值，
          计数类取 log1p）+ 增长率，用训练格子标准化；格子真实总被引（逐年）作为需求塔目标。
  所有特征都来自 T 时可见的信息（见 data_pipeline【30】【31】）。

输入
  config_exp2.SAMPLES、config_exp2.RIVALS

输出（config_exp2.TENSORS）
  meta.parquet       paper_id, split, cell, Y, emb_row（按张量顺序）
  x_paper.npy        (N, d_p) float32
  r_row.npy          (N, 50) int64：对手在 emb_sample.npy 中的行号
  r_feat.npy         (N, 50, d_r) float16
  r_rank.npy, r_coupled.npy   (N, 50)：名次、是否文献耦合（消融 A17 用）
  y.npy              (N, 3) float32：第 1、2、3 年被引
  cell_x.npy         (C, d_c) float32；cell_tot.npy (C, 3)；cell_split.npy (C,)；cell_ptr.npy (C+1,)：格子在论文数组中的起止
  feature_names.json

用法
  python 01_build_tensors.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C


INST_COLS = ["inst_cites_max", "inst_cites_mean", "inst_npap_max", "inst_npap_mean", "inst_cpp_max", "inst_cpp_mean",
             "auth_cpp_max", "auth_cpp_mean"]
REF_COLS = ["ref_found", "ref_lc_mean", "ref_lc_max", "ref_lc_p75", "ref_share_100", "ref_share_recent", "ref_age_mean", "ref_share_same_sf"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", action="store_true", help="v6：加参考文献质量特征（【33】）与渠道 / 方向编号，写到 TENSORS_v6")
    ap.add_argument("--inst", action="store_true", help="v7：再加机构历史与作者篇均被引（【34】）")
    ap.add_argument("--rivals", default=None, help="竞争集合文件名（默认 config 的 rivals.parquet）")
    ap.add_argument("--k", type=int, default=C.K_RIVALS)
    ap.add_argument("--suffix", default=None, help="张量目录后缀（默认 v6 → _v6）")
    ap.add_argument("--ytarget", default="all", help="all = 全部来源被引（主实验）；dppdcc = DPPDCC 口径的语料内被引（与 DPPDCC 比较用）")
    args = ap.parse_args()
    v6 = args.v6 or args.inst
    t0 = time.time()
    OUTD = C.TENSORS + (args.suffix if args.suffix is not None else ("_v6" if v6 else ""))
    os.makedirs(OUTD, exist_ok=True)
    s = pd.read_parquet(C.SAMPLES)
    if v6:
        rf_ = pd.read_parquet(os.path.join(os.path.dirname(C.SAMPLES), "ref_feats.parquet"))
        s = s.merge(rf_, on="paper_id", how="left")
        s["ref_found"] = np.log1p(s.ref_found)
    if args.inst:
        s = s.merge(pd.read_parquet(os.path.join(os.path.dirname(C.SAMPLES), "inst_feats.parquet")), on="paper_id", how="left")
    s["cell"] = s.ct2000.astype(np.int64) * 10000 + s.Y
    order = {"train": 0, "val": 1, "test": 2}
    s = s.sort_values(["split", "cell", "paper_id"], key=lambda c: c.map(order) if c.name == "split" else c).reset_index(drop=True)
    tr = (s.split == "train").to_numpy()

    # ---------- 论文特征
    pnum = C.PAPER_NUM + (REF_COLS if v6 else []) + (INST_COLS if args.inst else [])
    P = s[pnum].astype(np.float64).copy()
    for c in C.LOG_COLS:
        P[c] = np.log1p(P[c].clip(lower=0))
    miss = [c for c in pnum if P[c].isna().any()]
    M = P[miss].isna().astype(np.float32).add_suffix("_na")
    mu, sd = P[tr].mean(), P[tr].std().replace(0, 1)
    P = ((P - mu) / sd).fillna(0).clip(-8, 8).astype(np.float32)
    O = pd.get_dummies(s[C.PAPER_CAT].astype(str), dtype=np.float32)
    xp = np.hstack([P.to_numpy(), M.to_numpy(), O.to_numpy()]).astype(np.float32)
    names_p = list(P.columns) + list(M.columns) + list(O.columns)

    # ---------- 竞争集合
    r = pd.read_parquet(os.path.join(os.path.dirname(C.RIVALS), args.rivals) if args.rivals else C.RIVALS)
    em = pd.read_parquet(C.EMB_META, columns=["paper_id", "row"]).set_index("paper_id").row
    pos = pd.Series(np.arange(len(s)), index=s.paper_id)
    r["i"] = pos.reindex(r.focal_id).to_numpy()
    assert r.i.notna().all()
    r = r.sort_values(["i", "rank"])
    N, K = len(s), args.k
    assert len(r) == N * K
    r_row = em.reindex(r.comp_id).to_numpy().reshape(N, K).astype(np.int64)
    feats = [r.sim.to_numpy(np.float32)]
    for d in (-2, -1, 0):
        feats.append((r.dyear == d).to_numpy(np.float32))
    dd = r.ddays.to_numpy(np.float64)
    feats += [np.nan_to_num(dd / 365.0).astype(np.float32), np.isnan(dd).astype(np.float32),
              np.log1p(r.rival_cum.to_numpy(np.float32))]
    for v in C.RIVAL_VENUES:
        feats.append((r.rival_venue == v).to_numpy(np.float32))
    for c in ("shared_auth", "coupled", "focal_cites", "rival_cites"):
        feats.append(r[c].to_numpy(np.float32))
    r_feat = np.stack(feats, axis=1).reshape(N, K, -1)
    sim_mu, sim_sd = r_feat[tr, :, 0].mean(), r_feat[tr, :, 0].std()
    r_feat[:, :, 0] = (r_feat[:, :, 0] - sim_mu) / sim_sd
    names_r = ["sim_std", "dyear_m2", "dyear_m1", "dyear_0", "ddays_yr", "ddays_na", "log_rival_cum"] + \
              [f"venue_{v}" for v in C.RIVAL_VENUES] + ["shared_auth", "coupled", "focal_cites", "rival_cites"]

    # ---------- 格子
    g = s.groupby("cell", sort=False)
    cells = pd.DataFrame({"cell": g.size().index, "n": g.size().to_numpy()})
    cells["start"] = np.r_[0, np.cumsum(cells.n.to_numpy())[:-1]]
    assert (s.cell.to_numpy()[cells.start.to_numpy()] == cells.cell.to_numpy()).all()
    cx = g[C.CELL_LOG + C.CELL_RAW].mean().reindex(cells.cell)
    for c in C.CELL_LOG:
        cx[c] = np.log1p(cx[c].clip(lower=0))
    cx["log_n_cell"] = np.log1p(cells.n.to_numpy())
    cx["g_sub_inflow"] = cx.sub_inflow_Y - cx.sub_inflow_Ym2
    cx["g_sub_supply"] = cx.sub_supply_Y - cx.sub_supply_Ym2
    cx["g_topic_inflow"] = cx.topic_inflow_Y - cx.topic_inflow_Ym2
    cx["n_per_sub_supply"] = cx.log_n_cell - cx.sub_supply_Y
    csplit = g.split.first().reindex(cells.cell).to_numpy()
    ctr = csplit == "train"
    cmiss = cx.isna().astype(np.float32).add_suffix("_na")
    cmiss = cmiss.loc[:, cmiss.any()]
    cmu, csd = cx[ctr].mean(), cx[ctr].std().replace(0, 1)
    cxs = ((cx - cmu) / csd).fillna(0).clip(-8, 8)
    cell_x = np.hstack([cxs.to_numpy(np.float32), cmiss.to_numpy(np.float32)])
    y = s[["y1", "y2", "y3a"]].to_numpy(np.float32)
    if args.ytarget == "dppdcc":
        # DPPDCC 的标签口径：只数被语料（v2 全部样本论文，2016–2024）内论文引用的次数，按施引论文的首次公开年份分年
        cy = json.load(open("/path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/sample_cite_year_dict.json"))
        y = np.zeros((len(s), 3), np.float32)
        for i, (pid, Y) in enumerate(zip(s.paper_id.astype(str), s.Y)):
            d = cy.get(pid, {})
            for k in range(3):
                y[i, k] = float(d.get(str(Y + 1 + k), 0))
    cell_tot = np.add.reduceat(y, cells.start.to_numpy(), axis=0)

    D = lambda n: os.path.join(OUTD, n)
    if v6:
        # 渠道 / 方向 / 第二方向的整数编号：只给训练集中出现 ≥ 20 次的值编号，其余（含训练集外的新值）= 0（未知）
        cats = {}
        for c in ("source_at_T", "topic", "topic2"):
            vc = s.loc[tr, c].value_counts()
            keep = vc[vc >= 20].index
            mp = pd.Series(np.arange(1, len(keep) + 1), index=keep)
            np.save(D(f"cat_{c}.npy"), s[c].map(mp).fillna(0).astype(np.int64).to_numpy())
            cats[c] = int(len(keep) + 1)
        json.dump(cats, open(D("cat_sizes.json"), "w"))
    s[["paper_id", "split", "cell", "Y", "emb_row", "topic", "eval_c2000"]].to_parquet(D("meta.parquet"), index=False)
    np.save(D("x_paper.npy"), xp)
    np.save(D("r_row.npy"), r_row)
    np.save(D("r_feat.npy"), r_feat.astype(np.float16))
    np.save(D("r_rank.npy"), r["rank"].to_numpy().reshape(N, K).astype(np.int8))
    np.save(D("r_coupled.npy"), r.coupled.to_numpy().reshape(N, K))
    np.save(D("y.npy"), y)
    np.save(D("cell_x.npy"), cell_x)
    np.save(D("cell_tot.npy"), cell_tot.astype(np.float32))
    np.save(D("cell_split.npy"), csplit.astype(str))
    np.save(D("cell_ptr.npy"), np.r_[cells.start.to_numpy(), N].astype(np.int64))
    json.dump({"paper": names_p, "rival": names_r, "cell": list(cxs.columns) + list(cmiss.columns)},
              open(D("feature_names.json"), "w"), ensure_ascii=False, indent=1)
    print(f"N = {N:,}, 格子 {len(cells):,}（训练 {int(ctr.sum()):,}），论文特征 {xp.shape[1]}，对手特征 {r_feat.shape[2]}，"
          f"格子特征 {cell_x.shape[1]}；格子大小中位数 {int(cells.n.median())}，最大 {int(cells.n.max())}（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
