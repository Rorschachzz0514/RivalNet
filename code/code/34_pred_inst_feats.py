"""
【34】实验 2 v7 · 机构历史与作者篇均被引（预测时间点 T = 首次公开年份 Y 年底之前可知）     阶段 3：预测实验

用途
  对每篇预测样本论文：
    机构历史  论文所有机构在 Y 年之前发表的样本论文数、这些论文截至 Y 年的累计被引；取最大 / 平均
              inst_cites_max, inst_cites_mean, inst_npap_max, inst_npap_mean, inst_cpp_max（机构篇均被引的最大值）
    作者篇均  每位作者 Y 年之前样本论文的篇均被引（截至 Y 年），取最大 / 平均：auth_cpp_max, auth_cpp_mean
  计数类特征取 log1p。只用 ≤ Y 年已发生的被引；机构、作者的历史只算 Y 年之前首次公开的论文。

输入
  subsets/<PRED_V2>/samples.parquet；subsets/<EXP0_V2>/{papers, paper_year}.parquet

输出
  subsets/<PRED_V2>/inst_feats.parquet

用法
  python 34_pred_inst_feats.py

运行记录
  2026-10-03  建立
"""
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def main():
    t0 = time.time()
    P = os.path.join(C.SUBSET_DIR, C.PRED_V2)
    V2 = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
    con = duckdb.connect()
    con.execute("SET threads=128; SET memory_limit='300GB'; SET temp_directory='/path/to/mpcc/tmp'; SET preserve_insertion_order=false")
    con.execute(f"create table p as select paper_id, first_public_year as yr, institution_ids, author_ids from read_parquet('{V2}/papers.parquet') where is_sample")
    con.execute(f"create table f as select paper_id, Y from read_parquet('{P}/samples.parquet')")
    con.execute(f"""create table cum as
        select y.paper_id, t.Yc, sum(y.cites) as cum from read_parquet('{V2}/paper_year.parquet') as y
        cross join (select unnest(range(2017, 2022)) as Yc) as t where y.year <= t.Yc group by 1, 2""")
    out = {}
    for ent, col in (("inst", "institution_ids"), ("auth", "author_ids")):
        con.execute(f"create or replace table ep as select paper_id, unnest({col}) as eid, yr from p where {col} is not null")
        con.execute(f"""create or replace table ey as
            select e.eid, t.Yc, count(*) as n_prior, sum(coalesce(c.cum, 0)) as cites_prior
            from ep as e cross join (select unnest(range(2017, 2022)) as Yc) as t
            left join cum as c on c.paper_id = e.paper_id and c.Yc = t.Yc
            where e.yr < t.Yc group by 1, 2""")
        con.execute(f"""create or replace table fe as
            select f.paper_id, f.Y, unnest(p.{col}) as eid from f join p using (paper_id) where p.{col} is not null""")
        out[ent] = con.execute(f"""
            select fe.paper_id,
                   ln(1 + max(coalesce(ey.cites_prior, 0))) as {ent}_cites_max, ln(1 + avg(coalesce(ey.cites_prior, 0))) as {ent}_cites_mean,
                   ln(1 + max(coalesce(ey.n_prior, 0))) as {ent}_npap_max, ln(1 + avg(coalesce(ey.n_prior, 0))) as {ent}_npap_mean,
                   ln(1 + max(coalesce(ey.cites_prior / nullif(ey.n_prior, 0), 0))) as {ent}_cpp_max,
                   ln(1 + avg(coalesce(ey.cites_prior / nullif(ey.n_prior, 0), 0))) as {ent}_cpp_mean
            from fe left join ey on ey.eid = fe.eid and ey.Yc = fe.Y group by 1""").df()
        print(f"{ent} 完成（{time.time() - t0:.0f}s）", flush=True)
    keep_a = ["paper_id", "auth_cpp_max", "auth_cpp_mean"]
    res = out["inst"].merge(out["auth"][keep_a], on="paper_id", how="outer")
    con.register("res", res)
    con.execute(f"copy res to '{P}/inst_feats.parquet' (format parquet)")
    n_s = con.execute("select count(*) from f").fetchone()[0]
    print(f"INST FEATS DONE: {len(res):,} / {n_s:,} 篇；有机构信息 {res.inst_cites_max.notna().mean():.1%}（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
