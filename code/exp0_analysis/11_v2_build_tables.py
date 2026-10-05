"""
【11】实验 0 v2：构造焦点论文表与子课题 × 年份表（预先登记见 实验0说明.md 第 7 节）

用途
  把 v2 数据（data_pipeline【26】–【28】）整理成回归可直接使用的宽表。
  焦点论文表包含全部 v2 焦点论文（383,843 篇），用 in_main / in_jc 标记主分析集与"只期刊 + 会议"集。

输入（config_exp0.V2_DATA）
  papers、paper_year、paper_year_in_corpus、sim_counts_v2、clusters、cluster_year、pairs、topic_year

输出（results_v2/）
  focal.parquet         一行一篇焦点论文：
                          paper_id, topic, Y, venue_class, in_main, in_jc, date_precision, pub_month（日期只到年 = 0）
                          y1, y2, y3a, y3, y3_in_corpus
                          c1000, c2000, c5000
                          sim_counts_v2 的全部计数列（n_y{m2,m1,0,p1,p2}_ge{τ}、n_cand_y*、n_pb365_* / n_pa365_*）
                          s3_m1 / s3_m1_strict（前一年首次公开的文献耦合对手数 / 其中互不引用的）
                          n_cited_prior（焦点引用了几篇 Y−1～Y 年的相似论文：S2 ≥ 0.95 或文献耦合）
                          控制变量 n_authors, n_institutions, n_refs, any_oa, has_abstract, has_funding, has_preprint
                          topic2, topic2_score, supply_topic2（第二方向当年 n_rule）, demand_topic2（第二方向 Y+1～Y+3 施引方规模）
                          growth_c2000（子课题 Y+1～Y+3 参考文献总数 ÷ Y−2～Y 参考文献总数，异质性分组用）
  cell_cohort.parquet   K × 子课题 × 年份（2017–2021）：n_cell（焦点论文数）, supply（cluster_year.n_papers）,
                          cohort_y3, demand_followers（Y+1～Y+3 refs_made 之和）, inflow_net（Y+1～Y+3 inflow 之和 − cohort_y3）
  11_describe.md        描述统计

用法
  python 11_v2_build_tables.py
  EXP0_V2_Y=noself EXP0_V2_RES=results_v2_noself python 11_v2_build_tables.py    实验 8 R7：剔除自引的被引口径

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C


def main():
    t0 = time.time()
    os.makedirs(C.V2_RES, exist_ok=True)
    D = lambda n: os.path.join(C.V2_DATA, n)
    R = lambda n: os.path.join(C.V2_RES, n)
    con = duckdb.connect()
    con.execute(f"SET threads={C.THREADS}; SET memory_limit='200GB'; SET temp_directory='{C.TMP_DIR}'")
    q = lambda s: con.execute(s)
    one = lambda s: con.execute(s).fetchone()[0]
    main_v = ", ".join(f"'{v}'" for v in C.V2_MAIN_VENUES)
    jc_v = ", ".join(f"'{v}'" for v in C.V2_JC_VENUES)
    a, b = C.Y_WINDOW

    q(f"create table pv as select * from read_parquet('{D('papers.parquet')}')")
    q(f"""create table f as
        select paper_id, pt_topic as topic, first_public_year as Y, venue_class,
               venue_class in ({main_v}) as in_main, venue_class in ({jc_v}) as in_jc,
               date_precision, case when date_precision = 'year' then 0 else month(first_public_date::date) end as pub_month,
               list_filter(topic_ids, x -> x <> pt_topic)[1] as topic2,
               list_extract(topic_scores, list_position(topic_ids, list_filter(topic_ids, x -> x <> pt_topic)[1])) as topic2_score,
               n_authors, n_institutions, n_refs, coalesce(any_oa, false) as any_oa, has_abstract,
               coalesce(has_funding, false) as has_funding, has_preprint
        from pv where is_focal""")
    q(f"""create table yy as
        select f.paper_id,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 1), 0) as y1,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 2), 0) as y2,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 3), 0) as y3a,
               coalesce(sum(p.cites) filter (where p.year between f.Y + {a} and f.Y + {b}), 0) as y3
        from f left join read_parquet('{D('paper_year.parquet')}') as p on p.paper_id = f.paper_id group by 1""")
    if os.environ.get("EXP0_V2_Y") == "noself":
        # 实验 8 稳健性 R7：被引改由引用边表计算（施引论文来自 OpenAlex 全库、去重），并剔除自引（施引与被引论文作者有交集）
        q(f"""create or replace table yy as
            select f.paper_id,
                   count(*) filter (where c.citing_year = f.Y + 1) as y1, count(*) filter (where c.citing_year = f.Y + 2) as y2,
                   count(*) filter (where c.citing_year = f.Y + 3) as y3a,
                   count(*) filter (where c.citing_year between f.Y + {a} and f.Y + {b}) as y3
            from f left join (select * from read_parquet('{D('citations.parquet')}') where not is_self_cite) as c on c.cited_id = f.paper_id
            group by 1""")
        print("被引口径：引用边表，剔除自引", flush=True)
    q(f"""create table yc as
        select f.paper_id, coalesce(sum(p.cites_in_corpus) filter (where p.year between f.Y + {a} and f.Y + {b}), 0) as y3_in_corpus
        from f left join read_parquet('{D('paper_year_in_corpus.parquet')}') as p on p.paper_id = f.paper_id group by 1""")
    # 文献耦合对手（前一年）与"引用了几篇相似的前人论文"
    q(f"""create table s3 as
        select t.focal_id as paper_id,
               count(*) filter (where t.found_by_s3 and c.first_public_year = f.Y - 1) as s3_m1,
               count(*) filter (where t.found_by_s3 and t.is_strict and c.first_public_year = f.Y - 1) as s3_m1_strict,
               count(*) filter (where t.focal_cites_comp and c.first_public_year between f.Y - 1 and f.Y
                                  and (t.found_by_s3 or t.sem_sim >= {C.TAU})) as n_cited_prior
        from read_parquet('{D('pairs.parquet')}') as t join f on f.paper_id = t.focal_id
        join pv as c on c.paper_id = t.comp_id group by 1""")
    q(f"create table ty as select * from read_parquet('{D('topic_year.parquet')}')")
    q(f"create table cl as select * from read_parquet('{D('clusters.parquet')}')")
    q(f"create table cy as select * from read_parquet('{D('cluster_year.parquet')}')")
    q(f"""create table focal as
        select f.*, yy.y1, yy.y2, yy.y3a, yy.y3, yc.y3_in_corpus, cl.c1000, cl.c2000, cl.c5000,
               sc.* exclude (focal_id),
               coalesce(s3.s3_m1, 0) as s3_m1, coalesce(s3.s3_m1_strict, 0) as s3_m1_strict,
               coalesce(s3.n_cited_prior, 0) as n_cited_prior,
               t2.n_rule as supply_topic2,
               (select sum(t.demand_followers_rule) from ty as t where t.topic_id = f.topic2 and t.year between f.Y + {a} and f.Y + {b}) as demand_topic2,
               (select sum(c.refs_made) from cy as c where c.K = 2000 and c.cluster = cl.c2000 and c.year between f.Y + {a} and f.Y + {b})
                 / nullif((select sum(c.refs_made) from cy as c where c.K = 2000 and c.cluster = cl.c2000 and c.year between f.Y - 2 and f.Y), 0)
                 as growth_c2000
        from f join yy using (paper_id) join yc using (paper_id) join cl using (paper_id)
        join read_parquet('{D('sim_counts_v2.parquet')}') as sc on sc.focal_id = f.paper_id
        left join s3 using (paper_id)
        left join ty as t2 on t2.topic_id = f.topic2 and t2.year = f.Y""")
    n_f, n_focal = one("select count(*) from f"), one("select count(*) from focal")
    assert n_focal == n_f, f"拼表后焦点论文丢失 {n_f} -> {n_focal}"
    q(f"copy focal to '{R('focal.parquet')}' (format parquet)")

    parts = []
    for k in C.V2_KS:
        parts.append(f"""
          select {k} as K, x.cluster, x.Y, x.n_cell, x.cohort_y3, s.n_papers as supply,
                 (select sum(c.refs_made) from cy as c where c.K = {k} and c.cluster = x.cluster and c.year between x.Y + {a} and x.Y + {b}) as demand_followers,
                 (select sum(c.inflow) from cy as c where c.K = {k} and c.cluster = x.cluster and c.year between x.Y + {a} and x.Y + {b}) - x.cohort_y3 as inflow_net
          from (select c{k} as cluster, Y, count(*) as n_cell, sum(y3) as cohort_y3 from focal group by 1, 2) as x
          join cy as s on s.K = {k} and s.cluster = x.cluster and s.year = x.Y""")
    q(f"create table cc as {' union all '.join(parts)}")
    bad = one("select count(*) from cc where supply <> n_cell")
    assert bad == 0, f"{bad} 个格子的供给 ≠ 焦点论文数（2017–2021 的样本论文都应是焦点论文）"
    q(f"copy cc to '{R('cell_cohort.parquet')}' (format parquet)")

    d = lambda s: con.execute(s).df()
    tau = C.V2_TAU
    lines = [f"# 实验 0 v2 · 11 描述统计\n", f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n",
             "## 焦点论文（按集合）\n",
             d(f"""select 'all' as sample, count(*) n, avg(y3) mean_y3, median(y3) med_y3, avg((y3 = 0)::int) zero_y3,
                          avg("n_ym1_ge{tau}") mean_P1, median("n_ym1_ge{tau}") med_P1, avg(("n_ym1_ge{tau}" = 0)::int) zero_P1,
                          avg((date_precision <> 'year')::int) date_precise from focal
                   union all
                   select 'main', count(*), avg(y3), median(y3), avg((y3 = 0)::int), avg("n_ym1_ge{tau}"), median("n_ym1_ge{tau}"),
                          avg(("n_ym1_ge{tau}" = 0)::int), avg((date_precision <> 'year')::int) from focal where in_main
                   union all
                   select 'jc', count(*), avg(y3), median(y3), avg((y3 = 0)::int), avg("n_ym1_ge{tau}"), median("n_ym1_ge{tau}"),
                          avg(("n_ym1_ge{tau}" = 0)::int), avg((date_precision <> 'year')::int) from focal where in_jc""").round(3).to_markdown(index=False), "\n",
             "## 主分析集按年份\n",
             d(f"""select Y, count(*) n, avg(y3) mean_y3, avg("n_ym1_ge{tau}") mean_P1, avg("n_y0_ge{tau}") mean_P0
                   from focal where in_main group by 1 order by 1""").round(3).to_markdown(index=False), "\n",
             "## 按发表渠道\n",
             d(f"""select venue_class, count(*) n, avg(y3) mean_y3, avg("n_ym1_ge{tau}") mean_P1, avg((date_precision <> 'year')::int) date_precise
                   from focal group by 1 order by n desc""").round(3).to_markdown(index=False), "\n",
             "## 子课题 × 年份格子\n",
             d("""select K, count(*) cells, avg(n_cell) mean_n, median(n_cell) med_n, min(n_cell) min_n, max(n_cell) max_n,
                         avg((inflow_net < 0)::int) inflow_net_negative from cc group by 1 order by 1""").round(3).to_markdown(index=False), "\n"]
    open(R("11_describe.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"11 DONE: focal {n_focal:,} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
