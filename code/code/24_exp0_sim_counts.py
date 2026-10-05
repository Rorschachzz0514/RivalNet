"""
【24】实验 0 子集（五）：每篇焦点论文的"相似论文计数"（全量比较，不截断）            阶段 2：实验 0

用途
  【22】只保留每篇焦点论文最相似的 100 篇。实验 0 的核心变量是"发表前撞题对手有多少篇"，在拥挤的方向里
  真正相似的论文可能超过 100 篇，截断会恰好压低竞争最激烈的论文的对手数，造成偏差。
  本脚本让每篇焦点论文与首次公开日期相差 ≤365 天的全部样本论文比较 SPECTER2 相似度（不截断），
  按一组相似度阈值统计个数，并区分"对手更早/更晚"和"是否同一主方向"。阈值最终取哪个，
  由【22】的随机对校准分布决定，这里把整组阈值都算好，之后不必重算。

输入
  subsets/exp0/emb_sample.npy + emb_meta.parquet（【22】生成的样本论文向量矩阵）
  subsets/exp0/papers.parquet（first_public_date）

输出（/path/to/mpcc/subsets/exp0/）
  sim_counts.parquet   每篇焦点论文一行：
    focal_id
    n_cand_before / n_cand_after                     时间窗内全部候选论文数（更早 / 更晚，不含自身）
    n_ge{阈值}_before / n_ge{阈值}_after              相似度 ≥ 阈值的候选数，阈值 0.85, 0.86, …, 0.98
    n_ge{阈值}_before_same / n_ge{阈值}_after_same    其中与焦点论文同一主方向的个数
  _sim_counts_parts/   各 GPU 的分块结果

机制
  8 张 GPU 各取一部分焦点论文；候选按首次公开日期排序，每批焦点论文只与日期窗口内的候选做矩阵乘法；
  float16 计算，阈值比较在 GPU 上完成，只回传计数

用法
  python 24_exp0_sim_counts.py --gpus 0,1,2,3,4,5,6,7
  python 24_exp0_sim_counts.py --keep <ids.parquet> --out <目录>   只保留 ids.parquet（paper_id 列）中的论文，
                                                                  焦点与候选都限于这些论文；供【25】的子样本变体使用

运行记录
  2026-10-02  8 张 GPU，每张 10–25 秒，391,439 篇焦点论文 × 59 列（14 个阈值 × 前/后 × 全部/同方向 + 候选数）
  2026-10-02  增加 --keep / --out（默认行为不变），供【25】子样本变体重算
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.SUBSET_DIR, C.EXP0)
PARTS = os.path.join(OUT, "_sim_counts_parts")
THRESHOLDS = [round(0.85 + 0.01 * i, 2) for i in range(14)]     # 0.85 ... 0.98
WINDOW_DAYS = 365


def load_meta(keep_path=None):
    import numpy as np
    import pandas as pd
    meta = pd.read_parquet(os.path.join(OUT, "emb_meta.parquet"))
    meta["row"] = np.arange(len(meta))                                                  # 在 emb_sample.npy 中的行号
    if keep_path:
        keep = pd.read_parquet(keep_path, columns=["paper_id"])
        meta = meta[meta.paper_id.isin(keep.paper_id)].reset_index(drop=True)
    p = pd.read_parquet(os.path.join(OUT, "papers.parquet"), columns=["paper_id", "first_public_date"])
    meta = meta.merge(p, on="paper_id", how="left")
    meta["day"] = (pd.to_datetime(meta.first_public_date, errors="coerce") - pd.Timestamp("2000-01-01")).dt.days
    meta["day"] = meta["day"].fillna((meta.first_public_year - 2000) * 365.25 + 182).astype(np.int32)
    return meta


def worker(gpu, part, nparts, batch, keep_path=None, parts_dir=PARTS):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    meta = load_meta(keep_path)
    E = np.load(os.path.join(OUT, "emb_sample.npy"), mmap_mode="r")
    E = torch.from_numpy(np.ascontiguousarray(E[meta.row.to_numpy()])).cuda()           # (N, 768) float16
    day = meta.day.to_numpy()
    order = np.argsort(day, kind="stable")                                              # 候选按日期排序
    day_sorted = day[order]
    topic = torch.from_numpy(meta.pt_topic.fillna(-1).astype(np.int64).to_numpy()).cuda()
    dayt = torch.from_numpy(day.astype(np.int32)).cuda()                               # int32 省显存
    focal = np.where(meta.is_focal.to_numpy())[0]
    focal = focal[np.argsort(day[focal], kind="stable")]                                # 焦点也按日期排, 窗口更紧
    mine = np.array_split(focal, nparts)[part]
    th = torch.tensor(THRESHOLDS, device="cuda", dtype=torch.float16)
    cols = {"focal_id": [], "n_cand_before": [], "n_cand_after": []}
    for t in THRESHOLDS:
        for k in ("before", "after", "before_same", "after_same"):
            cols[f"n_ge{t:.2f}_{k}"] = []
    t0 = time.time()
    for s in range(0, len(mine), batch):
        q = mine[s:s + batch]
        lo = np.searchsorted(day_sorted, day[q].min() - WINDOW_DAYS, "left")
        hi = np.searchsorted(day_sorted, day[q].max() + WINDOW_DAYS, "right")
        cand = torch.from_numpy(order[lo:hi]).cuda()
        qi = torch.from_numpy(q).cuda()
        sim = E[qi] @ E[cand].T                                                         # (b, c) float16
        dd = dayt[cand][None, :] - dayt[qi][:, None]                                    # 候选日期 - 焦点日期
        inwin = dd.abs() <= WINDOW_DAYS
        notself = cand[None, :] != qi[:, None]
        before = inwin & notself & (dd < 0)
        after = inwin & notself & (dd >= 0)
        same = topic[cand][None, :] == topic[qi][:, None]
        cols["focal_id"].append(meta.paper_id.to_numpy()[q])
        cols["n_cand_before"].append(before.sum(1).cpu().numpy())
        cols["n_cand_after"].append(after.sum(1).cpu().numpy())
        for j, t in enumerate(THRESHOLDS):
            ge = sim >= th[j]
            cols[f"n_ge{t:.2f}_before"].append((ge & before).sum(1).cpu().numpy())
            cols[f"n_ge{t:.2f}_after"].append((ge & after).sum(1).cpu().numpy())
            cols[f"n_ge{t:.2f}_before_same"].append((ge & before & same).sum(1).cpu().numpy())
            cols[f"n_ge{t:.2f}_after_same"].append((ge & after & same).sum(1).cpu().numpy())
        if (s // batch) % 20 == 0:
            print(f"[gpu{gpu}] {s + len(q):,}/{len(mine):,} focal, cand window {hi - lo:,}, "
                  f"{(s + len(q)) / (time.time() - t0):.0f} focal/s", flush=True)
    tbl = pa.table({k: (np.concatenate(v).astype(np.int32) if k != "focal_id" else np.concatenate(v))
                    for k, v in cols.items()})
    pq.write_table(tbl, os.path.join(parts_dir, f"part{part:02d}.parquet"), compression="zstd")
    print(f"[gpu{gpu}] DONE {len(mine):,} focal in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--keep", default=None, help="只保留这些论文（parquet，paper_id 列）")
    ap.add_argument("--out", default=OUT, help="sim_counts.parquet 的输出目录")
    args = ap.parse_args()
    parts_dir = os.path.join(args.out, "_sim_counts_parts")
    os.makedirs(parts_dir, exist_ok=True)
    for f in glob.glob(os.path.join(parts_dir, "*.parquet")):
        os.remove(f)
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(g, i, len(gpus), args.batch, args.keep, parts_dir))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))
    if len(parts) != len(gpus):
        sys.exit(f"分块缺失: {len(parts)}/{len(gpus)}")
    t = pa.concat_tables([pq.read_table(f) for f in parts])
    pq.write_table(t, os.path.join(args.out, "sim_counts.parquet"), compression="zstd")
    print(f"SIM COUNTS DONE: {t.num_rows:,} focal papers, {t.num_columns} columns", flush=True)


if __name__ == "__main__":
    main()
