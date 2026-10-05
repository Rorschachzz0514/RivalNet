"""
【22】实验 0 子集（三）：内容相似对手 S2（SPECTER2 向量，GPU 近邻搜索）              阶段 2：实验 0

用途
  每篇焦点论文与首次公开年份相差 ≤1 年的样本论文比较 SPECTER2 余弦相似度，保留最相似的 K=100 篇；
  同时抽样随机论文对校准相似度分布（SPECTER2 余弦普遍偏高，"相似"的阈值要相对于随机对的分布来定）。

输入
  subsets/exp0/papers.parquet、union/specter2/*.parquet

输出（/path/to/mpcc/subsets/exp0/）
  emb_sample.npy / emb_meta.parquet   样本论文的归一化向量矩阵（float16）及其顺序
  _knn_parts/y{年}_p{块}.parquet      各 GPU 的分块结果
  pairs_knn.parquet                   focal_id、comp_id、sim、rank（1–100）
  _sim_calibration.json               随机同时期对、随机同方向同时期对的相似度分位数

用法
  python 22_exp0_pairs_knn.py --gpus 0,1,2,3,4,5,6,7 --k 100

运行记录
  2026-10-02  运行中（见数据说明第 7 节）
  原文件名：02c_pairs_knn.py
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.SUBSET_DIR, C.EXP0)
EMB_DIR = os.path.join(C.UNION_DIR, "specter2")


def build_matrix():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    p = pq.read_table(os.path.join(OUT, "papers.parquet"),
                      columns=["paper_id", "first_public_year", "pt_topic", "is_sample", "is_focal"]).to_pandas()
    p = p[p.is_sample].reset_index(drop=True)
    want = set(p.paper_id.tolist())
    ids, vecs = [], []
    for f in sorted(glob.glob(os.path.join(EMB_DIR, "*.parquet"))):
        t = pq.read_table(f, columns=["id", "emb"])
        idc = t.column("id").to_numpy()
        m = pd.Index(idc).isin(want)
        if m.any():
            sub = t.filter(pa_mask(m))
            ids.append(sub.column("id").to_numpy())
            vecs.append(np.asarray(sub.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                                   dtype=np.float16).reshape(-1, 768))
    ids = np.concatenate(ids)
    vecs = np.concatenate(vecs).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    order = pd.Series(np.arange(len(ids)), index=ids)
    missing = len(want) - len(ids)
    p = p[p.paper_id.isin(ids)].reset_index(drop=True)
    vecs = vecs[order[p.paper_id].to_numpy()]
    np.save(os.path.join(OUT, "emb_sample.npy"), vecs.astype(np.float16))
    p.to_parquet(os.path.join(OUT, "emb_meta.parquet"), index=False)
    print(f"向量矩阵: {len(p):,} 篇样本论文 (缺向量 {missing})", flush=True)
    return missing


def pa_mask(m):
    import pyarrow as pa
    return pa.array(m)


def knn_worker(gpu, jobs, k):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    meta = pd.read_parquet(os.path.join(OUT, "emb_meta.parquet"))
    E = torch.from_numpy(np.load(os.path.join(OUT, "emb_sample.npy"), mmap_mode="r")[:]).cuda()  # float16
    year = meta.first_public_year.to_numpy()
    pid = meta.paper_id.to_numpy()
    for (yy, part, nparts) in jobs:
        outp = os.path.join(OUT, "_knn_parts", f"y{yy}_p{part:02d}.parquet")
        if os.path.exists(outp):                      # 断点续跑: 已完成的块跳过
            continue
        fidx = np.where((year == yy) & meta.is_focal.to_numpy())[0]
        fidx = np.array_split(fidx, nparts)[part]
        cidx = np.where((year >= yy - 1) & (year <= yy + 1))[0]
        Ct = E[torch.from_numpy(cidx).cuda()]
        rows_f, rows_c, rows_s, rows_r = [], [], [], []
        for s in range(0, len(fidx), 2048):
            q = fidx[s:s + 2048]
            sim = E[torch.from_numpy(q).cuda()] @ Ct.T                      # (q, cand) float16
            # 排除自身
            self_pos = torch.from_numpy(np.searchsorted(cidx, q)).cuda()
            sim[torch.arange(len(q), device="cuda"), self_pos] = -2
            v, i = torch.topk(sim, k, dim=1)              # 直接在 float16 上取 top-k, 省显存
            rows_f.append(np.repeat(pid[q], k))
            rows_c.append(pid[cidx[i.cpu().numpy().reshape(-1)]])
            rows_s.append(v.float().cpu().numpy().reshape(-1))
            rows_r.append(np.tile(np.arange(1, k + 1, dtype=np.int16), len(q)))
        t = pa.table({"focal_id": np.concatenate(rows_f), "comp_id": np.concatenate(rows_c),
                      "sim": np.concatenate(rows_s).astype(np.float32), "rank": np.concatenate(rows_r)})
        pq.write_table(t, outp + ".tmp", compression="zstd")
        os.replace(outp + ".tmp", outp)
        print(f"[gpu{gpu}] year {yy} part {part}: {len(fidx):,} focal x {len(cidx):,} candidates", flush=True)


def calibrate():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    meta = pd.read_parquet(os.path.join(OUT, "emb_meta.parquet"))
    E = np.load(os.path.join(OUT, "emb_sample.npy"), mmap_mode="r")
    year, topic = meta.first_public_year.to_numpy(), meta.pt_topic.to_numpy()
    focal = np.where(meta.is_focal.to_numpy())[0]
    out = {}
    # 同时期随机对 (年份相差 <= 1)
    a = rng.choice(focal, 1_000_000)
    b = rng.integers(0, len(meta), 1_000_000)
    keep = np.abs(year[a] - year[b]) <= 1
    a, b = a[keep], b[keep]
    s = np.einsum("ij,ij->i", E[a].astype(np.float32), E[b].astype(np.float32))
    out["random_same_period"] = {f"p{q}": float(np.percentile(s, q)) for q in (50, 90, 95, 99, 99.9)}
    # 同方向同时期随机对
    df = pd.DataFrame({"i": np.arange(len(meta)), "t": topic, "y": year})
    pairs = []
    for (t, y), g in df[df.i.isin(focal)].groupby(["t", "y"]):
        pool = df[(df.t == t) & (df.y.between(y - 1, y + 1))].i.to_numpy()
        if len(pool) < 2:
            continue
        n = min(200, len(g))
        pairs.append(np.stack([rng.choice(g.i.to_numpy(), n), rng.choice(pool, n)], 1))
    pr = np.concatenate(pairs)
    pr = pr[pr[:, 0] != pr[:, 1]]
    s2 = np.einsum("ij,ij->i", E[pr[:, 0]].astype(np.float32), E[pr[:, 1]].astype(np.float32))
    out["random_same_topic_period"] = {f"p{q}": float(np.percentile(s2, q)) for q in (50, 90, 95, 99, 99.9)}
    out["n_pairs"] = {"same_period": int(len(s)), "same_topic_period": int(len(s2))}
    json.dump(out, open(os.path.join(OUT, "_sim_calibration.json"), "w"), indent=1)
    print("相似度校准:", json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, "emb_sample.npy")):
        build_matrix()
    print(f"[{time.time() - t0:.0f}s] 矩阵就绪", flush=True)
    os.makedirs(os.path.join(OUT, "_knn_parts"), exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    nparts = 8
    jobs = [(yy, p, nparts) for yy in range(2017, 2022) for p in range(nparts)]
    per = [jobs[i::len(gpus)] for i in range(len(gpus))]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=knn_worker, args=(g, per[i], args.k)) for i, g in enumerate(gpus)]
    for pr in procs:
        pr.start()
    calibrate()                                   # CPU 上与 GPU 并行
    for pr in procs:
        pr.join()
    import pyarrow.parquet as pq
    import pyarrow as pa
    parts = sorted(glob.glob(os.path.join(OUT, "_knn_parts", "*.parquet")))
    if len(parts) != len(jobs):
        sys.exit(f"KNN 分块缺失: {len(parts)}/{len(jobs)}")
    pq.write_table(pa.concat_tables([pq.read_table(f) for f in parts]), os.path.join(OUT, "pairs_knn.parquet"),
                   compression="zstd")
    print(f"[{time.time() - t0:.0f}s] KNN DONE: {len(parts)} 块", flush=True)


if __name__ == "__main__":
    main()
