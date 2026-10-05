"""
【29】实验 0 v2：导出每篇焦点论文在 Y−1～Y+1 年的全部高相似论文对（SPECTER2 ≥ 0.95，不截断）   阶段 2：实验 0

用途
  【28】只给出计数。"浪潮位置"分析需要知道当年 / 后一年的相似论文里，哪些引用了焦点论文
  （引用了它的是"跟随者"，数量受焦点论文自身质量影响；没引用它的才是独立的"同期 / 后来者"，可以作为子课题热度的代理）。
  本脚本逐对导出：焦点论文 × 候选（全部 v2 样本论文）中，首次公开年份差 d ∈ {−1, 0, +1}、相似度 ≥ 0.95 的论文对，
  并与 v2 引用边对照，标记双方是否互相引用。

输入（subsets/exp0_v2/）
  emb_meta.parquet、citations.parquet；../exp0/emb_sample.npy

输出（subsets/exp0_v2/）
  sim_pairs_95.parquet   focal_id, comp_id, sim（float16 计算后转 float32）, dyear（候选年份 − 焦点年份）,
                         comp_cites_focal, focal_cites_comp
  _sim_pairs_parts/      各 GPU 分块

用法
  python 29_exp0_v2_sim_pairs.py --gpus 0,1,2,3,4,5,7

运行记录
  2026-10-02  建立
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

V2 = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
SRC = os.path.join(C.SUBSET_DIR, C.EXP0)
PARTS = os.path.join(V2, "_sim_pairs_parts")
TAU = 0.95


def worker(gpu, part, nparts, batch):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    m = pd.read_parquet(os.path.join(V2, "emb_meta.parquet"))
    m = m.iloc[np.argsort(m.first_public_year.to_numpy(), kind="stable")].reset_index(drop=True)
    E = np.load(os.path.join(SRC, "emb_sample.npy"), mmap_mode="r")
    X = torch.from_numpy(np.ascontiguousarray(E[m.row.to_numpy()])).cuda()
    yr = m.first_public_year.to_numpy()
    ys = {y: int(np.searchsorted(yr, y, "left")) for y in range(2014, 2028)}
    ids = m.paper_id.to_numpy()
    focal = np.where(m.is_focal.to_numpy())[0]
    mine = np.array_split(focal, nparts)[part]
    out = {"focal_id": [], "comp_id": [], "sim": [], "dyear": []}
    t0 = time.time()
    for y in sorted(set(yr[mine])):
        fy = mine[yr[mine] == y]
        lo, hi = ys[y - 1], ys[y + 2]
        Ec = X[lo:hi]
        for s in range(0, len(fy), batch):
            q = fy[s:s + batch]
            sim = X[torch.from_numpy(q).cuda()] @ Ec.T
            r, c = torch.nonzero(sim >= TAU, as_tuple=True)
            v = sim[r, c].float().cpu().numpy()
            r, c = r.cpu().numpy(), c.cpu().numpy() + lo
            keep = q[r] != c
            out["focal_id"].append(ids[q[r[keep]]])
            out["comp_id"].append(ids[c[keep]])
            out["sim"].append(v[keep])
            out["dyear"].append((yr[c[keep]] - y).astype(np.int8))
        print(f"[gpu{gpu}] year {y} done ({time.time() - t0:.0f}s)", flush=True)
    tbl = pa.table({k: np.concatenate(v) for k, v in out.items()})
    pq.write_table(tbl, os.path.join(PARTS, f"part{part:02d}.parquet"), compression="zstd")
    print(f"[gpu{gpu}] DONE {tbl.num_rows:,} pairs ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,7")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    for f in glob.glob(os.path.join(PARTS, "*.parquet")):
        os.remove(f)
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(g, i, len(gpus), args.batch)) for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        sys.exit("子进程失败")
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=96; SET memory_limit='200GB'; SET temp_directory='/path/to/mpcc/tmp'")
    cit = os.path.join(V2, "citations.parquet")
    con.execute(f"""copy (
        select p.*, (c1.citing_id is not null) as comp_cites_focal, (c2.citing_id is not null) as focal_cites_comp
        from read_parquet('{PARTS}/*.parquet') as p
        left join (select distinct citing_id, cited_id from read_parquet('{cit}')) as c1 on c1.citing_id = p.comp_id and c1.cited_id = p.focal_id
        left join (select distinct citing_id, cited_id from read_parquet('{cit}')) as c2 on c2.citing_id = p.focal_id and c2.cited_id = p.comp_id
    ) to '{os.path.join(V2, 'sim_pairs_95.parquet')}' (format parquet, compression zstd)""")
    n = con.execute(f"select count(*), count(distinct focal_id) from read_parquet('{os.path.join(V2, 'sim_pairs_95.parquet')}')").fetchone()
    # 自查：与【28】的计数一致
    chk = con.execute(f"""
        with c as (select focal_id, count(*) filter (where dyear = -1) as m1, count(*) filter (where dyear = 0) as y0,
                          count(*) filter (where dyear = 1) as p1
                   from read_parquet('{os.path.join(V2, 'sim_pairs_95.parquet')}') group by 1)
        select count(*) from read_parquet('{os.path.join(V2, 'sim_counts_v2.parquet')}') as s left join c using (focal_id)
        where coalesce(c.m1, 0) <> s."n_ym1_ge0.95" or coalesce(c.y0, 0) <> s."n_y0_ge0.95" or coalesce(c.p1, 0) <> s."n_yp1_ge0.95" """).fetchone()[0]
    print(f"SIM PAIRS DONE: {n[0]:,} pairs, {n[1]:,} focal; 与【28】计数不一致的焦点论文 {chk:,} 篇", flush=True)


if __name__ == "__main__":
    main()
