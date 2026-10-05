"""
【28】实验 0 v2：每篇焦点论文的相似论文计数——按日历年份分档（不受日期精度影响）+ 按天窗口（只用日期精确的论文）   阶段 2：实验 0

用途
  第一版【24】按"首次公开日期 ±365 天"计数，但 v1 中 38% 的日期是 1 月 1 日占位值，前后顺序不可靠。v2 改为：
    按年份分档（主）：与焦点论文首次公开年份 Y 相差 d = −2, −1, 0, +1, +2 年的样本论文中，SPECTER2 相似度 ≥ τ 的个数。
                      "前一年的相似论文数"（d = −1）不依赖日期精度，是 v2 的主竞争变量。
    按天窗口（稳健）：焦点论文与对手的日期都精确（到日或到月）时，发表前 / 后 365 天内相似度 ≥ τ 的个数；
                      焦点论文日期不精确时记为缺失。
  候选 = 全部 v2 样本论文（含预印本，不含无效条目；补合并后的代表记录），焦点 = 全部 v2 焦点论文。不含自身。
  阈值 τ = 0.90, 0.91, …, 0.98。GPU 并行：焦点按年份分组，每批只与 Y−2～Y+2 年的候选做矩阵乘法。

输入（subsets/exp0_v2/）
  emb_meta.parquet（【26】）、../exp0/emb_sample.npy

输出（subsets/exp0_v2/）
  sim_counts_v2.parquet   focal_id；n_cand_y{d}；n_y{d}_ge{τ}（d ∈ m2, m1, 0, p1, p2）；
                          n_cand_pb365 / n_cand_pa365、n_pb365_ge{τ} / n_pa365_ge{τ}（发表前 / 后 365 天，仅日期精确者，否则缺失）
  _sim_counts_v2_parts/   各 GPU 分块

用法
  python 28_exp0_v2_sim_counts.py --gpus 0,1,2,3,4,5,7

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
PARTS = os.path.join(V2, "_sim_counts_v2_parts")
TAUS = [round(0.90 + 0.01 * i, 2) for i in range(9)]
DS = [-2, -1, 0, 1, 2]
DN = {-2: "m2", -1: "m1", 0: "0", 1: "p1", 2: "p2"}
WIN = 365


def load_meta():
    import numpy as np
    import pandas as pd
    m = pd.read_parquet(os.path.join(V2, "emb_meta.parquet"))
    m["day"] = (pd.to_datetime(m.first_public_date) - pd.Timestamp("2000-01-01")).dt.days.astype(np.int32)
    return m


def worker(gpu, part, nparts, batch):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    m = load_meta()
    order = np.argsort(m.first_public_year.to_numpy(), kind="stable")
    m = m.iloc[order].reset_index(drop=True)                                    # 候选按年份排序
    E = np.load(os.path.join(SRC, "emb_sample.npy"), mmap_mode="r")
    X = torch.from_numpy(np.ascontiguousarray(E[m.row.to_numpy()])).cuda()
    yr = m.first_public_year.to_numpy()
    ystart = {y: int(np.searchsorted(yr, y, "left")) for y in range(2014, 2028)}
    dayt = torch.from_numpy(m.day.to_numpy()).cuda()
    prec = torch.from_numpy(m.date_precise.to_numpy()).cuda()
    th = torch.tensor(TAUS, device="cuda", dtype=torch.float16)
    focal = np.where(m.is_focal.to_numpy())[0]
    mine = np.array_split(focal, nparts)[part]
    cols = {"focal_id": []}
    for d in DS:
        cols[f"n_cand_y{DN[d]}"] = []
        for t in TAUS:
            cols[f"n_y{DN[d]}_ge{t:.2f}"] = []
    for k in ("pb365", "pa365"):
        cols[f"n_cand_{k}"] = []
        for t in TAUS:
            cols[f"n_{k}_ge{t:.2f}"] = []
    t0 = time.time()
    done = 0
    for y in sorted(set(yr[mine])):
        fy = mine[yr[mine] == y]
        lo, hi = ystart[y - 2], ystart[y + 3]
        segs = [(ystart[y + d] - lo, ystart[y + d + 1] - lo) for d in DS]
        cand = torch.arange(lo, hi, device="cuda")
        Ec = X[lo:hi]
        for s in range(0, len(fy), batch):
            q = torch.from_numpy(fy[s:s + batch]).cuda()
            sim = X[q] @ Ec.T                                                    # (b, c) float16
            notself = cand[None, :] != q[:, None]
            cols["focal_id"].append(m.paper_id.to_numpy()[fy[s:s + batch]])
            ges = [(sim >= th[j]) & notself for j in range(len(TAUS))]
            for d, (a, b) in zip(DS, segs):
                cols[f"n_cand_y{DN[d]}"].append(notself[:, a:b].sum(1).cpu().numpy())
                for j, t in enumerate(TAUS):
                    cols[f"n_y{DN[d]}_ge{t:.2f}"].append(ges[j][:, a:b].sum(1).cpu().numpy())
            dd = dayt[cand][None, :] - dayt[q][:, None]
            ok = prec[cand][None, :] & prec[q][:, None] & notself & (dd.abs() <= WIN)
            fprec = prec[q].cpu().numpy()
            for k, msk in (("pb365", ok & (dd < 0)), ("pa365", ok & (dd >= 0))):
                c = msk.sum(1).cpu().numpy().astype(np.float32)
                c[~fprec] = np.nan
                cols[f"n_cand_{k}"].append(c)
                for j, t in enumerate(TAUS):
                    v = (ges[j] & msk).sum(1).cpu().numpy().astype(np.float32)
                    v[~fprec] = np.nan
                    cols[f"n_{k}_ge{t:.2f}"].append(v)
            done += len(q)
        print(f"[gpu{gpu}] year {y}: {done:,}/{len(mine):,} ({time.time() - t0:.0f}s)", flush=True)
    tbl = {}
    for k, v in cols.items():
        a = np.concatenate(v)
        tbl[k] = a if k == "focal_id" else (a.astype(np.float32) if "365" in k else a.astype(np.int32))
    pq.write_table(pa.table(tbl), os.path.join(PARTS, f"part{part:02d}.parquet"), compression="zstd")
    print(f"[gpu{gpu}] DONE {len(mine):,} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,7")
    ap.add_argument("--batch", type=int, default=256)
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
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(os.path.join(PARTS, "*.parquet")))
    if len(parts) != len(gpus) or any(p.exitcode != 0 for p in procs):
        sys.exit(f"分块缺失或子进程失败: {len(parts)}/{len(gpus)}")
    t = pa.concat_tables([pq.read_table(f) for f in parts])
    m = load_meta()
    assert t.num_rows == int(m.is_focal.sum()), "行数应等于 v2 焦点论文数"
    ids = t.column("focal_id").to_numpy()
    assert len(np.unique(ids)) == len(ids), "focal_id 重复"
    pq.write_table(t, os.path.join(V2, "sim_counts_v2.parquet"), compression="zstd")
    print(f"SIM COUNTS V2 DONE: {t.num_rows:,} focal, {t.num_columns} columns", flush=True)


if __name__ == "__main__":
    main()
