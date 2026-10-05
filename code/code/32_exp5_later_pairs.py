"""
【32】实验 5-2（强对手事件）：2017–2019 年焦点论文与其后 1–2 年出现的高相似论文对（SPECTER2 ≥ 0.95，不截断）   阶段 4：因果验证

用途
  事件研究需要知道：焦点论文 i（首次公开年份 Y）发表后，Y+1、Y+2 年是否出现了和它高度相似的"强对手" k。
  本脚本逐对导出 i × k（k 为任意 v2 样本论文，首次公开年份 ∈ {Y+1, Y+2}，相似度 ≥ 0.95），并标记：
    comp_cites_focal  k 的参考文献里有 i（k 是"跟随者 / 客户"，不算对手）
    shared_auth       有共同作者
    k_age1_cites      k 在首次公开后第一个完整年份的被引
    k_age1_pctl       上一项在"同一主方向、同一首次公开年份"全部样本论文中的百分位（0–1）
  焦点论文 = v2 焦点论文中 2017–2019 年的全部论文（实验脚本再按主分析集筛选）。

输入（subsets/exp0_v2/）
  emb_meta、papers、paper_year、citations；../exp0/emb_sample.npy

输出（subsets/exp0_v2/）
  later_pairs_95.parquet   focal_id, comp_id, sim, dyear(1/2), comp_cites_focal, shared_auth, k_age1_cites, k_age1_pctl

用法
  python 32_exp5_later_pairs.py --gpus 0,1,2

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
PARTS = os.path.join(V2, "_later_pairs_parts")
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
    focal = np.where(m.is_focal.to_numpy() & (yr <= 2019))[0]
    mine = np.array_split(focal, nparts)[part]
    out = {"focal_id": [], "comp_id": [], "sim": [], "dyear": []}
    t0 = time.time()
    for y in sorted(set(yr[mine])):
        fy = mine[yr[mine] == y]
        lo, hi = ys[y + 1], ys[y + 3]
        Ec = X[lo:hi]
        for s in range(0, len(fy), batch):
            q = fy[s:s + batch]
            sim = X[torch.from_numpy(q).cuda()] @ Ec.T
            r, c = torch.nonzero(sim >= TAU, as_tuple=True)
            v = sim[r, c].float().cpu().numpy()
            r, c = r.cpu().numpy(), c.cpu().numpy() + lo
            out["focal_id"].append(ids[q[r]])
            out["comp_id"].append(ids[c])
            out["sim"].append(v)
            out["dyear"].append((yr[c] - y).astype(np.int8))
        print(f"[gpu{gpu}] year {y} done ({time.time() - t0:.0f}s)", flush=True)
    pq.write_table(pa.table({k: np.concatenate(v) for k, v in out.items()}), os.path.join(PARTS, f"part{part:02d}.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
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
    V = lambda n: os.path.join(V2, n)
    con.execute(f"""create table p as select paper_id, pt_topic, first_public_year, author_ids from read_parquet('{V('papers.parquet')}') where is_sample""")
    con.execute(f"""create table a1 as
        select p.paper_id, p.pt_topic, p.first_public_year, coalesce(y.cites, 0) as age1
        from p left join read_parquet('{V('paper_year.parquet')}') as y on y.paper_id = p.paper_id and y.year = p.first_public_year + 1""")
    con.execute("""create table pct as select paper_id, age1, percent_rank() over (partition by pt_topic, first_public_year order by age1) as pctl from a1""")
    con.execute(f"""copy (
        select l.*, (c.citing_id is not null) as comp_cites_focal,
               len(list_intersect(coalesce(pf.author_ids, []), coalesce(pk.author_ids, []))) > 0 as shared_auth,
               pc.age1 as k_age1_cites, pc.pctl as k_age1_pctl
        from read_parquet('{PARTS}/*.parquet') as l
        join p as pf on pf.paper_id = l.focal_id join p as pk on pk.paper_id = l.comp_id
        left join (select distinct citing_id, cited_id from read_parquet('{V('citations.parquet')}')) as c
          on c.citing_id = l.comp_id and c.cited_id = l.focal_id
        left join pct as pc on pc.paper_id = l.comp_id
    ) to '{V('later_pairs_95.parquet')}' (format parquet, compression zstd)""")
    n = con.execute(f"""select count(*), count(distinct focal_id), avg(comp_cites_focal::int), avg((k_age1_pctl >= 0.95)::int)
                       from read_parquet('{V('later_pairs_95.parquet')}')""").fetchone()
    print(f"LATER PAIRS DONE: {n[0]:,} pairs, {n[1]:,} focal; comp cites focal {n[2]:.3f}; strong (top 5%) {n[3]:.3f}", flush=True)


if __name__ == "__main__":
    main()
