"""
【33】实验 2 v6 · 参考文献质量特征（预测时间点 T = 首次公开年份 Y 的年底之前可知）       阶段 3：预测实验

用途
  "站在谁的肩膀上"是论文影响力的常用信号，而且在发表时就已知。对每篇预测样本论文，展开其参考文献（全部 OpenAlex
  作品，不限于样本），在总库 lite 表中查每篇参考文献的发表年份与逐年被引，计算（只用 ≤ Y 年的被引）：
    ref_found          在总库中找到的参考文献数
    ref_lc_mean        参考文献截至 Y 年累计被引的 log1p 平均
    ref_lc_max         同上，最大值
    ref_lc_p75         同上，75 分位
    ref_share_100      截至 Y 年累计被引 ≥ 100 的参考文献比例
    ref_share_recent   发表于 Y−2～Y 年的参考文献比例（追前沿的程度）
    ref_age_mean       参考文献的平均年龄（Y − 参考文献年份）
    ref_share_same_sf  与焦点论文同一子领域的参考文献比例（跨学科程度的反面）
  注意：OpenAlex 的逐年被引只保留 2012 年以后，2012 年以前的被引不计入（对所有论文同等处理）。

输入
  subsets/<PRED_V2>/samples.parquet；subsets/<EXP0_V2>/papers.parquet（refs、pt_subfield）；union/lite/*.parquet

输出
  subsets/<PRED_V2>/ref_feats.parquet   paper_id + 上述 8 列

用法
  python 33_pred_ref_feats.py

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
    LITE = os.path.join(C.ROOT, "union", "lite", "*.parquet")
    con = duckdb.connect()
    con.execute("SET threads=160; SET memory_limit='400GB'; SET temp_directory='/path/to/mpcc/tmp'; SET preserve_insertion_order=false")
    con.execute(f"""create table fr as
        select s.paper_id, s.Y, p.pt_subfield as sf, unnest(p.refs) as ref
        from read_parquet('{os.path.join(P, 'samples.parquet')}') as s
        join read_parquet('{os.path.join(V2, 'papers.parquet')}') as p using (paper_id)""")
    n_pairs = con.execute("select count(*) from fr").fetchone()[0]
    print(f"参考文献对 {n_pairs:,}（{time.time() - t0:.0f}s）", flush=True)
    con.execute("create table rid as select distinct ref as id from fr")
    con.execute(f"""create table lt as
        select l.id, l.year, l.pt_subfield, l.cby_year, l.cby_count
        from read_parquet('{LITE}') as l semi join rid on rid.id = l.id""")
    print(f"在总库中找到参考文献 {con.execute('select count(*) from lt').fetchone()[0]:,} 篇（{time.time() - t0:.0f}s）", flush=True)
    con.execute("""create table rc as
        select fr.paper_id, fr.Y, fr.sf, lt.year as ryear, lt.pt_subfield as rsf,
               coalesce(list_sum(list_transform(list_zip(lt.cby_year, lt.cby_count), x -> case when x[1] <= fr.Y then x[2] else 0 end)), 0) as cum
        from fr join lt on lt.id = fr.ref""")
    con.execute(f"""copy (
        select paper_id, count(*) as ref_found, avg(ln(1 + cum)) as ref_lc_mean, max(ln(1 + cum)) as ref_lc_max,
               quantile_cont(ln(1 + cum), 0.75) as ref_lc_p75, avg((cum >= 100)::int) as ref_share_100,
               avg((ryear between Y - 2 and Y)::int) as ref_share_recent, avg(Y - ryear) as ref_age_mean,
               avg((rsf = sf)::int) as ref_share_same_sf
        from rc where ryear <= Y group by 1
    ) to '{os.path.join(P, 'ref_feats.parquet')}' (format parquet)""")
    r = con.execute(f"""select count(*), avg(ref_found), avg(ref_lc_mean), avg(ref_share_recent)
                        from read_parquet('{os.path.join(P, 'ref_feats.parquet')}')""").fetchone()
    n_s = con.execute(f"select count(*) from read_parquet('{os.path.join(P, 'samples.parquet')}')").fetchone()[0]
    print(f"REF FEATS DONE: {r[0]:,} / {n_s:,} 篇有特征；平均找到 {r[1]:.1f} 篇参考文献，平均 log 被引 {r[2]:.2f}，"
          f"近两年参考文献比例 {r[3]:.2f}（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
