"""
【21】实验 0 子集（二）：文献耦合对手 S3                                         阶段 2：实验 0

用途
  "谁和谁在抢引用"OpenAlex 不提供，需要自己算。S3 定义：焦点论文与首次公开年份相差 ≤1 年的样本论文，
  共享参考文献 ≥3 篇且 Jaccard 重叠 ≥5%（被样本内 500 篇以上论文引用的"万能文献"不参与计算）。

输入
  subsets/exp0/papers.parquet（is_sample、is_focal、first_public_year、refs）

输出（/path/to/mpcc/subsets/exp0/）
  pairs_coupling.parquet          focal_id、comp_id、n_shared_refs、jaccard
  _pairs_coupling_summary.json    排除的万能文献数、对数
  _pairs.duckdb                   中间工作库

用法
  python 21_exp0_pairs_coupling.py [--threads 160]

运行记录
  2026-10-02  15 秒：排除万能文献 2,061 篇；6,266,842 对，覆盖焦点论文 293,574 篇（75%）
  原文件名：02b_pairs_coupling.py
"""
import argparse
import json
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = os.path.join(C.SUBSET_DIR, C.EXP0)
HUB_MAX = 500
MIN_SHARED, MIN_JACCARD = 3, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=160)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    con = duckdb.connect(os.path.join(OUT, "_pairs.duckdb"))
    con.execute(f"SET threads={args.threads}; SET memory_limit='400GB'; SET temp_directory='{C.TMP_DIR}'; "
                "SET preserve_insertion_order=false")
    P = os.path.join(OUT, "papers.parquet")

    con.execute(f"""create or replace table sp as
        select paper_id, first_public_year y, is_focal, n_refs, refs from read_parquet('{P}') where is_sample""")
    con.execute("""create or replace table r as select paper_id, y, is_focal, unnest(refs) as ref from sp""")
    con.execute(f"""create or replace table hub as select ref from r group by 1 having count(*) > {HUB_MAX}""")
    n_hub = con.execute("select count(*) from hub").fetchone()[0]
    con.execute("create or replace table r2 as select * from r anti join hub using (ref)")
    log(f"样本论文 {con.execute('select count(*) from sp').fetchone()[0]:,}; 参考文献条目 "
        f"{con.execute('select count(*) from r').fetchone()[0]:,}; 万能文献 {n_hub:,} 篇已排除")

    # 焦点 x 候选 (年份相差 <= 1), 按共享参考文献计数; 分焦点年份做, 控制中间结果大小
    con.execute("create or replace table pairs (focal_id bigint, comp_id bigint, n_shared_refs integer)")
    for yy in range(2017, 2022):
        con.execute(f"""insert into pairs
            select a.paper_id, b.paper_id, count(*)::integer
            from (select paper_id, ref from r2 where is_focal and y = {yy}) a
            join (select paper_id, ref from r2 where y between {yy - 1} and {yy + 1}) b
              on a.ref = b.ref and a.paper_id <> b.paper_id
            group by 1, 2 having count(*) >= {MIN_SHARED}""")
        log(f"焦点年份 {yy}: 累计 {con.execute('select count(*) from pairs').fetchone()[0]:,} 对 (共享 >= {MIN_SHARED})")
    con.execute(f"""copy (
        select p.focal_id, p.comp_id, p.n_shared_refs,
               p.n_shared_refs / (a.n_refs + b.n_refs - p.n_shared_refs)::double jaccard
        from pairs p join sp a on a.paper_id = p.focal_id join sp b on b.paper_id = p.comp_id
        where p.n_shared_refs / (a.n_refs + b.n_refs - p.n_shared_refs)::double >= {MIN_JACCARD}
    ) to '{OUT}/pairs_coupling.parquet' (format parquet, compression zstd)""")
    n = con.execute(f"select count(*), count(distinct focal_id) from read_parquet('{OUT}/pairs_coupling.parquet')").fetchone()
    log(f"S3 文献耦合对手: {n[0]:,} 对, 覆盖焦点论文 {n[1]:,} 篇")
    json.dump({"hub_refs_excluded": n_hub, "pairs": n[0], "focal_with_pairs": n[1],
               "rule": f"shared>={MIN_SHARED}, jaccard>={MIN_JACCARD}, hub>{HUB_MAX} excluded, |year diff|<=1"},
              open(os.path.join(OUT, "_pairs_coupling_summary.json"), "w"), indent=1)
    print("COUPLING DONE", flush=True)


if __name__ == "__main__":
    main()
