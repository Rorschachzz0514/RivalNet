"""
【23】实验 0 子集（四）：合并 S2 与 S3，补全每一对的属性，生成 T5 对手表               阶段 2：实验 0

输入（subsets/exp0/）
  pairs_knn.parquet（【22】）、pairs_coupling.parquet（【21】）、papers.parquet、citations.parquet、
  emb_sample.npy + emb_meta.parquet

输出（/path/to/mpcc/subsets/exp0/）
  pairs.parquet，每行一对（焦点论文，对手）：
    focal_id, comp_id
    found_by_s2, knn_rank      在焦点论文的 SPECTER2 近邻前 100 名中（及名次）
    found_by_s3                满足文献耦合规则
    sem_sim                    SPECTER2 余弦相似度（所有对都有；只被 S3 找到的对用向量矩阵补算）
    n_shared_refs, jaccard     共享参考文献数与重叠比例（所有对都有）
    days_after_focal           对手首次公开日期 − 焦点论文首次公开日期（负数 = 对手更早）
    same_topic, comp_topic     是否同一主方向；对手的主方向
    focal_cites_comp, comp_cites_focal, is_strict   互引关系（来自 T2）；is_strict = 互不引用（严格替代品候选）
  _pairs_summary.json
  S1（同方向同期）不生成配对，由 topic_year 表计数

用法
  python 23_exp0_pairs_merge.py [--threads 160]

运行记录
  2026-10-02  第 1 次因列名 strict 是 DuckDB 保留字报错（只做了语法检查、没做冒烟测试），改名 is_strict 并给所有别名加 AS；
              第 2 次成功，140 秒：43,632,704 对（S2 39,143,900、S3 6,266,842、两者都是 1,778,038），覆盖全部 391,439 篇焦点论文；
              平均相似度 0.927、同方向 63.8%、互不引用 98.2%、对手更早 43.1%、相似度缺失 0
  原文件名：02d_pairs_merge.py（新写）
"""
import argparse
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.SUBSET_DIR, C.EXP0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=160)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    con = duckdb.connect(os.path.join(OUT, "_merge.duckdb"))
    con.execute(f"SET threads={args.threads}; SET memory_limit='400GB'; SET temp_directory='{C.TMP_DIR}'; "
                "SET preserve_insertion_order=false")
    f = lambda n: os.path.join(OUT, n)

    # 1. 合并两种来源
    con.execute(f"""create or replace table pr as
        select coalesce(k.focal_id, c.focal_id) focal_id, coalesce(k.comp_id, c.comp_id) comp_id,
               k.focal_id is not null found_by_s2, k.rank knn_rank, k.sim sim_knn,
               c.focal_id is not null found_by_s3
        from read_parquet('{f("pairs_knn.parquet")}') k
        full join read_parquet('{f("pairs_coupling.parquet")}') c on k.focal_id = c.focal_id and k.comp_id = c.comp_id""")
    n = con.execute("select count(*), sum(found_by_s2::int), sum(found_by_s3::int), sum((found_by_s2 and found_by_s3)::int) from pr").fetchone()
    log(f"合并: {n[0]:,} 对 (S2 {n[1]:,}, S3 {n[2]:,}, 两者都是 {n[3]:,})")

    # 2. 只被 S3 找到的对补算 SPECTER2 相似度 (用 02c 的向量矩阵)
    meta = pd.read_parquet(f("emb_meta.parquet"), columns=["paper_id"])
    E = np.load(f("emb_sample.npy"), mmap_mode="r")
    pos = pd.Series(np.arange(len(meta)), index=meta.paper_id.to_numpy())
    need = con.execute("select focal_id, comp_id from pr where sim_knn is null").df()
    sims = np.empty(len(need), dtype=np.float32)
    a, b = pos[need.focal_id].to_numpy(), pos[need.comp_id].to_numpy()
    for s in range(0, len(need), 1_000_000):
        sims[s:s + 1_000_000] = np.einsum("ij,ij->i", E[a[s:s + 1_000_000]].astype(np.float32),
                                          E[b[s:s + 1_000_000]].astype(np.float32))
    need["sim_extra"] = sims
    con.register("need_df", need)
    log(f"补算相似度 {len(need):,} 对")

    # 3. 其余属性
    con.execute(f"""create or replace table pp as
        select paper_id, first_public_date, pt_topic, refs, n_refs from read_parquet('{f("papers.parquet")}') where is_sample""")
    con.execute(f"""create or replace table cit as
        select citing_id, cited_id from read_parquet('{f("citations.parquet")}')""")
    con.execute(f"""copy (
        select p.focal_id, p.comp_id, p.found_by_s2, p.knn_rank, p.found_by_s3,
               coalesce(p.sim_knn, x.sim_extra) as sem_sim,
               len(list_intersect(a.refs, b.refs)) as n_shared_refs,
               len(list_intersect(a.refs, b.refs)) / nullif(a.n_refs + b.n_refs - len(list_intersect(a.refs, b.refs)), 0)::double as jaccard,
               date_diff('day', cast(a.first_public_date as date), cast(b.first_public_date as date)) as days_after_focal,
               a.pt_topic = b.pt_topic as same_topic, b.pt_topic as comp_topic,
               fc.citing_id is not null as focal_cites_comp, cf.citing_id is not null as comp_cites_focal,
               fc.citing_id is null and cf.citing_id is null as is_strict
        from pr p
        join pp a on a.paper_id = p.focal_id
        join pp b on b.paper_id = p.comp_id
        left join need_df x on x.focal_id = p.focal_id and x.comp_id = p.comp_id
        left join cit fc on fc.citing_id = p.focal_id and fc.cited_id = p.comp_id
        left join cit cf on cf.citing_id = p.comp_id and cf.cited_id = p.focal_id
    ) to '{f("pairs.parquet")}' (format parquet, compression zstd)""")
    s = con.execute(f"""select count(*) n, count(distinct focal_id) n_focal, avg(sem_sim) mean_sim,
             avg(same_topic::int) same_topic, avg(is_strict::int) is_strict, avg((days_after_focal < 0)::int) comp_earlier,
             sum((sem_sim is null)::int) null_sim
        from read_parquet('{f("pairs.parquet")}')""").df().iloc[0].to_dict()
    s = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in s.items()}
    s["seconds"] = round(time.time() - t0)
    json.dump(s, open(f("_pairs_summary.json"), "w"), indent=1)
    log(f"T5 对手表: {s}")
    print("PAIRS MERGE DONE" if s["null_sim"] == 0 else "PAIRS MERGE DONE (WARNING: null sims)", flush=True)


if __name__ == "__main__":
    main()
