"""
【35】对比方法 DPPDCC（CIKM 2024）的输入：把 v2 AI 语料导出为其官方流程（utilis/s2orc.py）约定的 JSON 格式   阶段 3：预测实验

用途
  DPPDCC 的官方流程从四个文件出发构建按年份变化的异构图（论文—作者—渠道）与被引标签：
    all_info_dict.json      {paper_id: {"year", "title", "authors": [{"authorId"}], "publicationvenueid"}}
    all_ref_dict.json       {paper_id: [被它引用的 paper_id, ...]}（只保留语料内部）
    all_cite_dict.json      {paper_id: [引用它的 paper_id, ...]}
    all_abstract_dict.json  {paper_id: {"abstract"}}
  语料 = 全部 v2 样本论文（2016–2024，含 2022–2024 年的施引论文，DPPDCC 用施引论文的年份统计逐年被引）。
  注意：DPPDCC 的被引标签只统计语料内部的引用（论文自身被 2022–2024 年语料内论文引用的次数），
  与我们主实验的"全部来源被引"不同；比较时 MPC-Net 等也在同一口径上重新训练与评价。

输入
  subsets/<EXP0_V2>/{papers, citations}.parquet
输出
  /path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/all_*.json

用法
  python 35_dppdcc_export.py

运行记录
  2026-10-03  建立
"""
import json
import os
import sys
import time

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

OUT = "/path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/"


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    V2 = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
    con = duckdb.connect()
    con.execute("SET threads=96; SET memory_limit='200GB'")
    p = con.execute(f"""select paper_id, first_public_year, coalesce(title, '') as title, coalesce(abstract, '') as abstract,
                               coalesce(author_ids, []) as author_ids, source_id
                        from read_parquet('{V2}/papers.parquet') where is_sample""").df()
    ids = set(p.paper_id)
    info = {str(r.paper_id): {"year": int(r.first_public_year), "title": r.title,
                              "authors": [{"authorId": str(a)} for a in r.author_ids],
                              "publicationvenueid": (None if pd.isna(r.source_id) else str(int(r.source_id)))}
            for r in p.itertuples()}
    json.dump(info, open(OUT + "all_info_dict.json", "w"))
    json.dump({str(r.paper_id): {"abstract": r.abstract} for r in p.itertuples()}, open(OUT + "all_abstract_dict.json", "w"))
    con.register("pid", p[["paper_id"]])
    e = con.execute(f"""select distinct c.citing_id, c.cited_id from read_parquet('{V2}/citations.parquet') as c
                        where c.citing_id in (select paper_id from pid) and c.cited_id in (select paper_id from pid)
                          and c.citing_id <> c.cited_id""").df()
    ref = e.groupby("citing_id").cited_id.apply(lambda x: [str(v) for v in x]).to_dict()
    cite = e.groupby("cited_id").citing_id.apply(lambda x: [str(v) for v in x]).to_dict()
    json.dump({str(k): v for k, v in ref.items()}, open(OUT + "all_ref_dict.json", "w"))
    json.dump({str(k): v for k, v in cite.items()}, open(OUT + "all_cite_dict.json", "w"))
    print(f"DPPDCC EXPORT DONE: {len(info):,} 篇论文，{len(e):,} 条语料内引用（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
