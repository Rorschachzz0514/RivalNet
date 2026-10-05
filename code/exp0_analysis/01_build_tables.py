"""
【01】构造实验 0 的两张分析表：焦点论文表、方向 × 年份表

用途
  把实验 0 子集（T1–T5、相似计数）整理成回归直接可用的宽表：每篇焦点论文的因变量、竞争变量、控制变量；
  每个方向 × 年份的总被引、供给、需求。变量定义见 实验0说明.md 第 3 节。

输入（config_exp0.DATA_DIR）
  papers.parquet、paper_year.parquet、paper_year_in_corpus.parquet、topic_year.parquet、
  pairs.parquet、sim_counts.parquet

输出（results/）
  focal.parquet          焦点论文 391,439 行：
                           paper_id, topic, Y, topic2, topic2_score（第二方向）
                           y1, y2, y3a（第 1、2、3 年被引）, y3（第 1–3 年之和）, y3_in_corpus
                           prior_s2_{τ}, prior_s2_{τ}_same（τ = 0.90–0.98，来自全量相似计数）, n_cand_before
                           prior_s3, prior_s3_strict, n_prior_cited
                           控制变量 n_authors, n_institutions, n_refs, any_oa, has_abstract, has_funding,
                                    has_preprint, n_versions
                           supply_topic2（第二方向该年供给 n_rule，含 AI 以外方向）
  topic_cohort.parquet   方向 × 年份 385 行：topic, Y, n_focal, supply, cohort_y3, demand_followers,
                           demand_inflow, demand_inflow_net
  01_describe.md         描述统计

用法
  python 01_build_tables.py           全量
  python 01_build_tables.py --test    只用 2019 年的焦点论文（冒烟测试）

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(C.RES_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={C.THREADS}; SET memory_limit='200GB'; SET temp_directory='{C.TMP_DIR}'")
    D = lambda n: os.path.join(C.DATA_DIR, n)
    y0, y1 = (2019, 2019) if args.test else C.FOCAL_YEARS
    a, b = C.Y_WINDOW

    con.execute(f"""create table f as
        select paper_id, pt_topic as topic, first_public_year as Y, first_public_date,
               list_filter(topic_ids, x -> x <> pt_topic)[1] as topic2,
               list_extract(topic_scores, list_position(topic_ids, list_filter(topic_ids, x -> x <> pt_topic)[1])) as topic2_score,
               n_authors, n_institutions, n_refs, coalesce(any_oa, false) any_oa, has_abstract,
               coalesce(has_funding, false) has_funding, has_preprint, n_versions
        from read_parquet('{D("papers.parquet")}') where is_focal and first_public_year between {y0} and {y1}""")

    # 因变量: 第 1-3 个完整年份的被引
    con.execute(f"""create table yy as
        select f.paper_id,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 1), 0) y1,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 2), 0) y2,
               coalesce(sum(p.cites) filter (where p.year = f.Y + 3), 0) y3a,
               coalesce(sum(p.cites) filter (where p.year between f.Y + {a} and f.Y + {b}), 0) y3
        from f left join read_parquet('{D("paper_year.parquet")}') p on p.paper_id = f.paper_id
        group by f.paper_id""")
    con.execute(f"""create table yc as
        select f.paper_id, coalesce(sum(p.cites_in_corpus) filter (where p.year between f.Y + {a} and f.Y + {b}), 0) y3_in_corpus
        from f left join read_parquet('{D("paper_year_in_corpus.parquet")}') p on p.paper_id = f.paper_id
        group by f.paper_id""")

    # 竞争变量: S2 全量相似计数 (不截断)
    s2cols = ", ".join(f'"n_ge{t:.2f}_before" as prior_s2_{int(round(t * 100))}, '
                       f'"n_ge{t:.2f}_before_same" as prior_s2_{int(round(t * 100))}_same' for t in C.TAU_GRID)
    con.execute(f"""create table s2 as select focal_id paper_id, n_cand_before, {s2cols}
                    from read_parquet('{D("sim_counts.parquet")}')""")
    # S3 与"引用了多少篇发表前的相似论文"
    con.execute(f"""create table s3 as
        select focal_id paper_id,
               count(*) filter (where found_by_s3 and days_after_focal between -{C.PRIOR_DAYS} and -1) prior_s3,
               count(*) filter (where found_by_s3 and is_strict and days_after_focal between -{C.PRIOR_DAYS} and -1) prior_s3_strict,
               count(*) filter (where focal_cites_comp and days_after_focal between -{C.PRIOR_DAYS} and -1
                                  and (found_by_s3 or sem_sim >= {C.TAU})) n_prior_cited
        from read_parquet('{D("pairs.parquet")}') group by 1""")
    con.execute(f"""create table ty as select topic_id, year, n_rule, n_ai_dedup, demand_followers_rule, demand_inflow
                    from read_parquet('{D("topic_year.parquet")}')""")

    con.execute(f"""create table focal as
        select f.*, yy.y1, yy.y2, yy.y3a, yy.y3, yc.y3_in_corpus, s2.* exclude (paper_id),
               coalesce(s3.prior_s3, 0) prior_s3, coalesce(s3.prior_s3_strict, 0) prior_s3_strict,
               coalesce(s3.n_prior_cited, 0) n_prior_cited,
               t2.n_rule supply_topic2
        from f join yy using (paper_id) join yc using (paper_id) join s2 using (paper_id)
        left join s3 using (paper_id)
        left join ty t2 on t2.topic_id = f.topic2 and t2.year = f.Y""")
    n_f = con.execute("select count(*) from f").fetchone()[0]
    n_focal = con.execute("select count(*) from focal").fetchone()[0]
    assert n_focal == n_f, f"焦点论文在拼表后丢失: {n_f} -> {n_focal} (sim_counts 缺失?)"
    con.execute(f"copy focal to '{os.path.join(C.RES_DIR, 'focal.parquet')}' (format parquet)")

    # 方向 x 年份
    con.execute(f"""create table cohort as
        select topic, Y, count(*) n_focal, sum(y3) cohort_y3 from focal group by 1, 2""")
    con.execute(f"""create table tc as
        select c.topic, c.Y, c.n_focal, c.cohort_y3, t0.n_ai_dedup supply,
               (select sum(t.demand_followers_rule) from ty t where t.topic_id = c.topic and t.year between c.Y + {a} and c.Y + {b}) demand_followers,
               (select sum(t.demand_inflow) from ty t where t.topic_id = c.topic and t.year between c.Y + {a} and c.Y + {b}) demand_inflow
        from cohort c left join ty t0 on t0.topic_id = c.topic and t0.year = c.Y""")
    con.execute("alter table tc add column demand_inflow_net double; update tc set demand_inflow_net = demand_inflow - cohort_y3")
    con.execute(f"copy tc to '{os.path.join(C.RES_DIR, 'topic_cohort.parquet')}' (format parquet)")

    # 描述统计
    q = lambda s: con.execute(s).df()
    tau = int(round(C.TAU * 100))
    lines = [f"# 实验 0 · 01 描述统计\n", f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}；焦点年份 {y0}–{y1}\n",
             "## 焦点论文\n", q(f"""select Y, count(*) n, avg(y3) mean_y3, median(y3) median_y3, avg((y3 = 0)::int) zero_y3,
                 avg(prior_s2_{tau}) mean_prior_s2, median(prior_s2_{tau}) median_prior_s2, avg((prior_s2_{tau} = 0)::int) zero_prior_s2,
                 avg(prior_s3) mean_prior_s3, avg(n_cand_before) mean_cand
                 from focal group by 1 order by 1""").round(3).to_markdown(index=False), "\n",
             "## 竞争变量分布（全体）\n", q(f"""select '{tau}' tau, quantile_cont(prior_s2_{tau}, [0.25, 0.5, 0.75, 0.9, 0.99]) q
                 from focal""").to_markdown(index=False), "\n",
             "## 方向 × 年份\n", q("""select count(*) cells, avg(n_focal) mean_n, min(n_focal) min_n, max(n_focal) max_n,
                 avg((supply = n_focal)::int) supply_equals_n_focal, avg((demand_inflow_net < 0)::int) inflow_net_negative
                 from tc""").round(3).to_markdown(index=False), "\n"]
    with open(os.path.join(C.RES_DIR, "01_describe.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"01 DONE: focal {n_focal:,}, cohort cells {con.execute('select count(*) from tc').fetchone()[0]} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
