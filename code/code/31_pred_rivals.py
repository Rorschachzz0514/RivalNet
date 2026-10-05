"""
【31】实验 2 的竞争集合（T5 扩展）：每篇样本论文在预测时间点可见的最相似 50 篇论文及其关系特征     阶段 3：预测实验

用途
  预测时间点 T = 首次公开年份 Y 的年底。候选 = Y−2～Y 年首次公开的全部 v2 样本论文（T 时都已可见），不含自身。
  GPU 上按 SPECTER2 余弦相似度取前 50 名（top-k），再补充关系特征（全部截至 T）：
    sim             相似度
    dyear           对手年份 − 焦点年份（−2, −1, 0）
    ddays           对手日期 − 焦点日期（双方日期都精确时；否则缺失）
    rival_cum       对手截至 Y 年的累计被引（对手强度；同年对手大多为 0）
    rival_venue     对手的发表渠道类别（T 时：正式版发表年份 > Y 记为 arxiv / preprint_other）
    shared_auth     与焦点论文有共同作者
    coupled         文献耦合（共享参考文献 ≥ 3 且 Jaccard ≥ 5%，来自 pairs；pairs 只覆盖焦点论文，其余记 false）
    focal_cites     焦点论文的参考文献里有这个对手（T 时可知）
    rival_cites     对手的参考文献里有焦点论文（对手在 T 前发表，T 时可知）
  输出为长表（一行一对），实验 2 的数据加载器按 focal_id 组装成定长集合（不足 50 补空并掩码）。

输入
  subsets/exp0_v2：emb_meta、papers、citations、pairs；subsets/pred_v2/samples.parquet；../exp0/emb_sample.npy

输出（subsets/pred_v2/）
  rivals.parquet   focal_id, rank（1–50）, comp_id, sim, dyear, ddays, rival_cum, rival_venue, shared_auth, coupled,
                   focal_cites, rival_cites
  _rivals_parts/   各 GPU 分块

用法
  python 31_pred_rivals.py --gpus 3,4,5
  python 31_pred_rivals.py --gpus 3,4,5 --k 100 --back 3 --out rivals_k100.parquet     v7：前三年内最相似 100 篇

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
OUT = os.path.join(C.SUBSET_DIR, C.PRED_V2)
PARTS = os.path.join(OUT, "_rivals_parts")
K = 50


def worker(gpu, part, nparts, batch, K=50, back=2):
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
    ys = {y: int(np.searchsorted(yr, y, "left")) for y in range(2013, 2028)}
    ids = m.paper_id.to_numpy()
    pos = pd.Series(np.arange(len(m)), index=ids)
    samp = pd.read_parquet(os.path.join(OUT, "samples.parquet"), columns=["paper_id"])
    focal = np.sort(pos.loc[samp.paper_id].to_numpy())
    mine = np.array_split(focal, nparts)[part]
    out = {"focal_id": [], "rank": [], "comp_id": [], "sim": [], "dyear": []}
    t0 = time.time()
    for y in sorted(set(yr[mine])):
        fy = mine[yr[mine] == y]
        lo, hi = ys[y - back], ys[y + 1]
        Ec = X[lo:hi]
        for s in range(0, len(fy), batch):
            q = fy[s:s + batch]
            qt = torch.from_numpy(q).cuda()
            sim = (X[qt] @ Ec.T).float()
            self_col = qt - lo
            sim[torch.arange(len(q), device="cuda"), self_col] = -2.0                 # 去掉自身
            v, i = torch.topk(sim, K, dim=1)
            i = i.cpu().numpy() + lo
            out["focal_id"].append(np.repeat(ids[q], K))
            out["rank"].append(np.tile(np.arange(1, K + 1, dtype=np.int8), len(q)))
            out["comp_id"].append(ids[i.ravel()])
            out["sim"].append(v.cpu().numpy().ravel().astype(np.float32))
            out["dyear"].append((yr[i.ravel()] - y).astype(np.int8))
        print(f"[gpu{gpu}] year {y}: done ({time.time() - t0:.0f}s)", flush=True)
    pq.write_table(pa.table({k: np.concatenate(v) for k, v in out.items()}), os.path.join(PARTS, f"part{part:02d}.parquet"))
    print(f"[gpu{gpu}] DONE ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="3,4,5")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--k", type=int, default=50, help="每篇论文的对手数")
    ap.add_argument("--back", type=int, default=2, help="候选向前看几年（Y−back～Y）")
    ap.add_argument("--out", default="rivals.parquet")
    args = ap.parse_args()
    global K
    K = args.k
    os.makedirs(PARTS, exist_ok=True)
    for f in glob.glob(os.path.join(PARTS, "*.parquet")):
        os.remove(f)
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(g, i, len(gpus), args.batch, args.k, args.back)) for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        sys.exit("子进程失败")
    import duckdb
    con = duckdb.connect()
    con.execute("SET threads=128; SET memory_limit='300GB'; SET temp_directory='/path/to/mpcc/tmp'; SET preserve_insertion_order=false")
    V = lambda n: os.path.join(V2, n)
    con.execute(f"create table r as select * from read_parquet('{PARTS}/*.parquet')")
    con.execute(f"create table f as select paper_id, Y from read_parquet('{os.path.join(OUT, 'samples.parquet')}')")
    con.execute(f"""create table p as select paper_id, first_public_year, first_public_date::date as d, date_precision, author_ids,
                       case when coalesce(published_year, 9999) <= first_public_year then venue_class
                            when arxiv_id is not null or venue_class = 'arxiv' then 'arxiv' else 'preprint_other' end as venue_T
                    from read_parquet('{V('papers.parquet')}') where is_sample""")
    con.execute(f"""create table cum as select y.paper_id, f.Y, sum(y.cites) as cum
                    from read_parquet('{V('paper_year.parquet')}') as y join (select distinct Y from f) as f on y.year <= f.Y group by 1, 2""")
    con.execute(f"create table cit as select distinct citing_id, cited_id from read_parquet('{V('citations.parquet')}')")
    con.execute(f"""create table cp as select distinct focal_id, comp_id from read_parquet('{V('pairs.parquet')}') where found_by_s3""")
    con.execute(f"""copy (
        select r.focal_id, r.rank, r.comp_id, r.sim, r.dyear,
               case when pf.date_precision <> 'year' and pc.date_precision <> 'year' then pc.d - pf.d end as ddays,
               coalesce(cu.cum, 0) as rival_cum, pc.venue_T as rival_venue,
               len(list_intersect(coalesce(pf.author_ids, []), coalesce(pc.author_ids, []))) > 0 as shared_auth,
               (cp.focal_id is not null) as coupled,
               (c1.citing_id is not null) as focal_cites, (c2.citing_id is not null) as rival_cites
        from r join f on f.paper_id = r.focal_id join p as pf on pf.paper_id = r.focal_id join p as pc on pc.paper_id = r.comp_id
        left join cum as cu on cu.paper_id = r.comp_id and cu.Y = f.Y
        left join cp on cp.focal_id = r.focal_id and cp.comp_id = r.comp_id
        left join cit as c1 on c1.citing_id = r.focal_id and c1.cited_id = r.comp_id
        left join cit as c2 on c2.citing_id = r.comp_id and c2.cited_id = r.focal_id
        order by r.focal_id, r.rank) to '{os.path.join(OUT, args.out)}' (format parquet, compression zstd)""")
    n = con.execute(f"select count(*), count(distinct focal_id), avg(sim), avg(coupled::int), avg(rival_cites::int) from read_parquet('{os.path.join(OUT, args.out)}')").fetchone()
    n_f = con.execute("select count(*) from f").fetchone()[0]
    assert n[1] == n_f and n[0] == n_f * K, f"竞争集合行数不对: {n} vs {n_f} × {K}"
    print(f"RIVALS DONE: {n[0]:,} rows, {n[1]:,} focal, mean sim {n[2]:.3f}, coupled {n[3]:.3f}, rival_cites {n[4]:.3f}", flush=True)


if __name__ == "__main__":
    main()
