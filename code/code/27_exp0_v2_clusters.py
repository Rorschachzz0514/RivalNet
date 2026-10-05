"""
【27】实验 0 v2：用 SPECTER2 向量把样本论文聚成"子课题"，并统计每个子课题每年的供给与需求          阶段 2：实验 0

用途
  第一版实验 0 只能控制到"77 个 AI 方向 × 年份"，同一方向内的热门子课题会同时带来"对手多"和"被引多"，
  把竞争效应掩盖成正相关。这里把样本论文聚成更细的子课题（K = 1000 / 2000 / 5000，主分析用 2000，
  约为 OpenAlex 方向粒度的 26 倍），供实验 0 v2 在"子课题 × 年份"内比较、并在子课题层面重做宏观检验。
  方法：球面 k-means（向量 L2 归一化后用内积），GPU 上运行；k-means++ 式的随机加权初始化（固定种子），
  迭代至分配变化 < 0.1% 或 200 轮；空簇用离自己中心最远的点重新播种。
  注意：聚类用到了 2016–2024 全部样本论文（包括焦点论文之后的年份）。实验 0 是描述 / 计量分析，可以这样做；
  实验 2 的预测若要用子课题，需只用训练期论文重新聚类，避免用到未来信息。
  每个子课题 × 年份（2016–2024）统计：
    n_papers      该年首次公开的 v2 样本论文数（供给）
    refs_made     这些论文的参考文献总数（"施引方规模"需求口径：有多少条参考文献要写）
    inflow        该年子课题内全部样本论文（任何年份发表的）收到的被引之和（"流入量"需求口径）

输入（subsets/exp0_v2/）
  emb_meta.parquet（【26】，含原向量矩阵的行号）、../exp0/emb_sample.npy、papers.parquet、paper_year.parquet

输出（subsets/exp0_v2/）
  clusters.parquet        paper_id, c1000, c2000, c5000，以及到所属簇中心的相似度 sim_c2000
  cluster_year.parquet    K, cluster, year, n_papers, refs_made, inflow
  _clusters_summary.json  各 K 的迭代次数、簇大小分布、平均簇内相似度

用法
  python 27_exp0_v2_clusters.py --gpu 0

运行记录
  2026-10-02  建立
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

V2 = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
SRC = os.path.join(C.SUBSET_DIR, C.EXP0)
KS = [1000, 2000, 5000]


def kmeans(X, k, seed, iters=200, chunk=65536):
    import torch
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = X.shape[0]
    # k-means++ 式初始化（分批近似：每次加 k/10 个中心，按到最近中心的距离加权抽样）
    cent = X[torch.randint(0, n, (1,), generator=g, device=X.device)]
    best = torch.full((n,), -2.0, device=X.device)
    step = max(1, k // 10)
    while cent.shape[0] < k:
        for s in range(0, n, chunk):
            best[s:s + chunk] = torch.maximum(best[s:s + chunk], (X[s:s + chunk] @ cent[-step:].T).max(1).values.float())
        w = (1 - best).clamp(min=1e-6)
        idx = torch.multinomial(w / w.sum(), min(step, k - cent.shape[0]), replacement=False, generator=g)
        cent = torch.cat([cent, X[idx]])
    assign = torch.full((n,), -1, dtype=torch.long, device=X.device)
    for it in range(iters):
        new = torch.empty_like(assign)
        sims = torch.empty(n, device=X.device)
        for s in range(0, n, chunk):
            v, i = (X[s:s + chunk] @ cent.T).max(1)
            new[s:s + chunk], sims[s:s + chunk] = i, v.float()
        changed = (new != assign).float().mean().item()
        assign = new
        sums = torch.zeros(k, X.shape[1], device=X.device, dtype=torch.float32).index_add_(0, assign, X.float())
        cnt = torch.bincount(assign, minlength=k)
        empty = (cnt == 0).nonzero().flatten()
        if len(empty):                                          # 空簇：用离中心最远的点重新播种
            far = torch.argsort(sims)[:len(empty)]
            sums[empty] = X[far].float()
            cnt[empty] = 1
        cent = torch.nn.functional.normalize(sums / cnt[:, None], dim=1).to(X.dtype)
        if changed < 1e-3:
            break
    return assign.cpu().numpy(), sims.cpu().numpy(), it + 1, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import duckdb
    t0 = time.time()
    meta = pd.read_parquet(os.path.join(V2, "emb_meta.parquet"))
    E = np.load(os.path.join(SRC, "emb_sample.npy"), mmap_mode="r")
    X = torch.from_numpy(np.ascontiguousarray(E[meta.row.to_numpy()])).cuda()
    X = torch.nn.functional.normalize(X.float(), dim=1).half()
    print(f"向量 {tuple(X.shape)}（{time.time() - t0:.0f}s）", flush=True)
    out = pd.DataFrame({"paper_id": meta.paper_id})
    summ = {}
    for k in KS:
        a, s, it, ch = kmeans(X, k, seed=2026 + k)
        out[f"c{k}"] = a.astype(np.int32)
        if k == 2000:
            out["sim_c2000"] = s.astype(np.float32)
        sz = np.bincount(a, minlength=k)
        summ[f"K{k}"] = {"iters": it, "last_change": ch, "size_min": int(sz.min()), "size_median": float(np.median(sz)),
                         "size_max": int(sz.max()), "mean_sim_to_center": float(s.mean())}
        print(f"K={k}: {it} 轮，最后变化 {ch:.4%}，簇大小 {sz.min()}–{np.median(sz):.0f}–{sz.max()}，"
              f"平均到中心相似度 {s.mean():.3f}（{time.time() - t0:.0f}s）", flush=True)
    out.to_parquet(os.path.join(V2, "clusters.parquet"), index=False)

    con = duckdb.connect()
    con.execute("SET threads=96; SET memory_limit='100GB'")
    con.register("cl", out)
    P = os.path.join(V2, "papers.parquet")
    PY = os.path.join(V2, "paper_year.parquet")
    parts = []
    for k in KS:
        parts.append(con.execute(f"""
            with sup as (select c.c{k} as cluster, p.first_public_year as year, count(*) as n_papers, sum(p.n_refs) as refs_made
                         from cl as c join read_parquet('{P}') as p using (paper_id) group by 1, 2),
                 inf as (select c.c{k} as cluster, y.year, sum(y.cites) as inflow
                         from cl as c join read_parquet('{PY}') as y using (paper_id) where y.year between 2016 and 2024 group by 1, 2),
                 grid as (select cluster, year from (select distinct c{k} as cluster from cl) cross join range(2016, 2025) t(year))
            select {k} as K, g.cluster, g.year, coalesce(s.n_papers, 0) as n_papers, coalesce(s.refs_made, 0) as refs_made,
                   coalesce(i.inflow, 0) as inflow
            from grid as g left join sup as s using (cluster, year) left join inf as i using (cluster, year)""").df())
    cy = pd.concat(parts, ignore_index=True)
    cy.to_parquet(os.path.join(V2, "cluster_year.parquet"), index=False)
    assert cy.groupby("K").n_papers.sum().eq(len(out)).all(), "簇 × 年份的论文数之和应等于样本数"
    summ["seconds"] = round(time.time() - t0)
    json.dump(summ, open(os.path.join(V2, "_clusters_summary.json"), "w"), indent=1)
    print(f"CLUSTERS DONE {summ}", flush=True)


if __name__ == "__main__":
    main()
