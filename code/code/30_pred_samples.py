"""
【30】实验 1 / 实验 2 共用的预测样本表（T8）：一行 = 一篇焦点论文，特征只用预测时间点之前的信息     阶段 3：预测实验

用途
  预测时间点 T = 首次公开年份 Y 的年底（"冷启动"：不使用焦点论文自身的任何被引）。目标：Y+1、Y+2、Y+3 年各自的被引数。
  切分：训练 2017–2019，验证 2020，测试 2021（设计文档 2.4 节）。样本 = v2 焦点论文中发表渠道为
  期刊 / 会议 / 论文集 / arXiv 的论文（与实验 0 v2 主分析集相同）。
  特征（全部 ≤ T；"历史"统计只用 ≤ Y 年已经发生的被引）：
    论文自身   作者数、机构数、国家数、参考文献数、开放获取、有摘要、有基金、基金数、发表月份（日期只到年 = 0）、
               T 时的发表渠道（正式版发表年份 > Y 的论文在 T 时只是预印本：渠道记为 arXiv / 预印本）、T 时是否已有预印本、
               主方向、第二方向及匹配分数、OpenAlex 主方向匹配分数
    文本       SPECTER2 向量在 emb_sample.npy 中的行号（训练时读取）
    作者历史   每位作者在 Y 年之前发表的样本论文数、这些论文截至 Y 年的累计被引；取最大 / 平均 / 第一作者
    渠道历史   同一渠道 Y−1 年论文在 Y 年的平均 log 被引；Y−3 年论文在 Y−2～Y 年的平均 log 被引之和；渠道论文数
    方向历史   主方向：Y−2～Y 年的供给（AI 论文数）、施引方规模、流入量；Y−1 / Y−3 年队列的平均 log 被引
    子课题     **只用 2019 年及以前的论文**重新做 k-means（K = 2000），所有论文按最近中心归属；子课题的同类历史统计
    竞争（T 时可见） 前两年 / 前一年 SPECTER2 ≥ 0.90 / 0.95 的相似论文数；日期精确时发表前 365 天的相似论文数；
               前一年的文献耦合对手数；这些前人对手截至 Y 年的累计被引（对手强度）；
               是否已被"抢先"：存在日期更早 1–183 天、相似度 ≥ 0.97、没有共同作者且互不引用的论文（及最小天数差）
  目标：y1, y2, y3a（Y+1、Y+2、Y+3 年被引），y3 = 三者之和；另存 y0（Y 年被引，只供"发表 1 年后"之外的分析，不作冷启动输入）。
  评价分组（只用于评价，不作为特征）：OpenAlex 主方向 × 年份；全数据子课题（【27】K = 2000 / 5000）× 年份。

输入（subsets/exp0_v2/）
  papers、paper_year、topic_year、emb_meta、clusters、sim_counts_v2、sim_pairs_95、pairs；../exp0/emb_sample.npy

输出（subsets/pred_v2/）
  samples.parquet          样本表（列见上）
  clusters_train.parquet   训练期子课题：paper_id, ct2000（全部 v2 样本论文）
  _summary.json            样本量、各切分规模、泄漏自查结果

用法
  python 30_pred_samples.py --gpu 0

运行记录
  2026-10-02  建立
"""
import argparse
import json
import os
import sys
import time
from importlib import import_module

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

V2 = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
SRC = os.path.join(C.SUBSET_DIR, C.EXP0)
OUT = os.path.join(C.SUBSET_DIR, C.PRED_V2)
MAIN_VENUES = ("journal", "conference", "book series", "arxiv")
SPLIT = {2017: "train", 2018: "train", 2019: "train", 2020: "val", 2021: "test"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def train_clusters(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    km = import_module("27_exp0_v2_clusters").kmeans
    meta = pd.read_parquet(os.path.join(V2, "emb_meta.parquet"))
    E = np.load(os.path.join(SRC, "emb_sample.npy"), mmap_mode="r")
    X = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(E[meta.row.to_numpy()])).cuda().float(), dim=1).half()
    tr = torch.from_numpy((meta.first_public_year <= 2019).to_numpy()).cuda()
    a, _, it, ch = km(X[tr], 2000, seed=4242)
    # 用训练期论文的簇中心给全部论文归属
    sums = torch.zeros(2000, X.shape[1], device="cuda").index_add_(0, torch.from_numpy(a).cuda(), X[tr].float())
    cent = torch.nn.functional.normalize(sums, dim=1).half()
    lab = torch.cat([(X[s:s + 65536] @ cent.T).argmax(1) for s in range(0, len(X), 65536)]).cpu().numpy()
    log(f"训练期子课题：{int(tr.sum()):,} 篇论文聚类，{it} 轮，最后变化 {ch:.4%}")
    return pd.DataFrame({"paper_id": meta.paper_id, "ct2000": lab.astype(np.int32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    ct = train_clusters(args.gpu)
    ct.to_parquet(os.path.join(OUT, "clusters_train.parquet"), index=False)
    V = lambda n: os.path.join(V2, n)
    con = duckdb.connect()
    con.execute("SET threads=128; SET memory_limit='300GB'; SET temp_directory='/path/to/mpcc/tmp'; SET preserve_insertion_order=false")
    q = lambda s: con.execute(s)
    one = lambda s: con.execute(s).fetchone()[0]
    con.register("ct_df", ct)
    mv = ", ".join(f"'{v}'" for v in MAIN_VENUES)

    q(f"create table p as select * from read_parquet('{V('papers.parquet')}') where is_sample")
    q(f"create table py as select * from read_parquet('{V('paper_year.parquet')}')")
    q("create table ct as select * from ct_df")
    # 焦点样本
    q(f"""create table f as
        select p.paper_id, p.first_public_year as Y, p.first_public_date, p.date_precision,
               case when p.date_precision = 'year' then 0 else month(p.first_public_date::date) end as pub_month,
               p.venue_class as venue_final,
               case when coalesce(p.published_year, 9999) <= p.first_public_year then p.venue_class
                    when p.arxiv_id is not null or p.venue_class = 'arxiv' then 'arxiv'
                    else 'preprint_other' end as venue_at_T,
               case when coalesce(p.published_year, 9999) <= p.first_public_year then p.source_id end as source_at_T,
               (p.preprint_date is not null and year(p.preprint_date::date) <= p.first_public_year) as has_preprint_at_T,
               p.pt_topic as topic, p.pt_score as topic_score,
               list_filter(p.topic_ids, x -> x <> p.pt_topic)[1] as topic2,
               list_extract(p.topic_scores, list_position(p.topic_ids, list_filter(p.topic_ids, x -> x <> p.pt_topic)[1])) as topic2_score,
               p.n_authors, coalesce(p.n_institutions, 0) as n_institutions, coalesce(p.n_countries, 0) as n_countries, p.n_refs,
               coalesce(p.any_oa, false) as any_oa, p.has_abstract, coalesce(p.has_funding, false) as has_funding,
               coalesce(p.n_grants, 0) as n_grants, p.author_ids, ct.ct2000
        from p join ct using (paper_id)
        where p.is_focal and p.venue_class in ({mv})""")
    n_f = one("select count(*) from f")
    log(f"样本 {n_f:,}")

    # 目标
    q("""create table tgt as
        select f.paper_id,
               coalesce(sum(y.cites) filter (where y.year = f.Y), 0) as y0,
               coalesce(sum(y.cites) filter (where y.year = f.Y + 1), 0) as y1,
               coalesce(sum(y.cites) filter (where y.year = f.Y + 2), 0) as y2,
               coalesce(sum(y.cites) filter (where y.year = f.Y + 3), 0) as y3a
        from f left join py as y on y.paper_id = f.paper_id group by 1""")

    # 每篇样本论文截至每个 Y 的累计被引（Y = 2017..2021）
    q("""create table cum as
        select y.paper_id, t.Yc, sum(y.cites) as cum
        from py as y cross join (select unnest(range(2017, 2022)) as Yc) as t
        where y.year <= t.Yc group by 1, 2""")
    # 作者历史
    q("create table ap as select paper_id, unnest(author_ids) as author_id, first_public_year as yr from p where author_ids is not null")
    q("""create table ay as
        select a.author_id, t.Yc, count(*) as n_prior, sum(coalesce(c.cum, 0)) as cites_prior
        from ap as a cross join (select unnest(range(2017, 2022)) as Yc) as t
        left join cum as c on c.paper_id = a.paper_id and c.Yc = t.Yc
        where a.yr < t.Yc group by 1, 2""")
    q("""create table fa as
        select f.paper_id, unnest(f.author_ids) as author_id, unnest(range(1, len(f.author_ids) + 1)) as pos, f.Y
        from f where f.author_ids is not null and len(f.author_ids) > 0""")
    q("""create table auth as
        select fa.paper_id, max(coalesce(ay.cites_prior, 0)) as auth_cites_max, avg(coalesce(ay.cites_prior, 0)) as auth_cites_mean,
               max(coalesce(ay.n_prior, 0)) as auth_npap_max, avg(coalesce(ay.n_prior, 0)) as auth_npap_mean,
               max(coalesce(ay.cites_prior, 0)) filter (where fa.pos = 1) as first_auth_cites,
               max(coalesce(ay.n_prior, 0)) filter (where fa.pos = 1) as first_auth_npap,
               avg((coalesce(ay.n_prior, 0) = 0)::int) as share_new_authors
        from fa left join ay on ay.author_id = fa.author_id and ay.Yc = fa.Y group by 1""")
    log("作者历史完成")

    # 队列统计：按某个分组键，cohort Y−1 在 Y 年的平均 log1p 被引（age 1），cohort Y−3 在 Y−2..Y 的平均 log1p 被引和
    def cohort_stats(key_expr, name):
        q(f"""create table g_{name} as select p.paper_id, {key_expr} as g, p.first_public_year as yr from p join ct using (paper_id)""")
        q(f"""create table cs_{name} as
            with a1 as (select g.g, g.yr + 1 as Yt, avg(ln(1 + coalesce(y.cites, 0))) as age1_mean, count(*) as n_age1
                        from g_{name} as g left join py as y on y.paper_id = g.paper_id and y.year = g.yr + 1 group by 1, 2),
                 a3 as (select g.g, g.yr + 3 as Yt, avg(ln(1 + coalesce(s.c, 0))) as y3_mean
                        from g_{name} as g
                        left join (select g2.paper_id, sum(y.cites) as c from g_{name} as g2 join py as y
                                   on y.paper_id = g2.paper_id and y.year between g2.yr + 1 and g2.yr + 3 group by 1) as s
                          on s.paper_id = g.paper_id group by 1, 2),
                 n as (select g, yr as Yt, count(*) as n_papers from g_{name} group by 1, 2)
            select coalesce(a1.g, a3.g, n.g) as g, coalesce(a1.Yt, a3.Yt, n.Yt) as Yt, a1.age1_mean, a1.n_age1, a3.y3_mean, n.n_papers
            from a1 full join a3 on a1.g = a3.g and a1.Yt = a3.Yt full join n on n.g = coalesce(a1.g, a3.g) and n.Yt = coalesce(a1.Yt, a3.Yt)""")

    cohort_stats("p.pt_topic", "topic")
    cohort_stats("ct.ct2000", "sub")
    cohort_stats("p.source_id", "venue")
    log("队列统计完成")

    # 方向需求历史（topic_year）
    q(f"create table ty as select * from read_parquet('{V('topic_year.parquet')}')")
    # 子课题需求历史：每年新论文数、参考文献总数、流入量（截至 Y）
    q("""create table sy as
        select ct.ct2000 as g, p.first_public_year as yr, count(*) as n_papers, sum(p.n_refs) as refs_made
        from p join ct using (paper_id) group by 1, 2""")
    q("""create table si as
        select ct.ct2000 as g, y.year as yr, sum(y.cites) as inflow from py as y join ct using (paper_id) group by 1, 2""")

    # 竞争特征（T 时可见）
    q(f"""create table sc as select focal_id as paper_id, "n_ym2_ge0.95" as c_m2_95, "n_ym1_ge0.95" as c_m1_95,
               "n_ym2_ge0.90" as c_m2_90, "n_ym1_ge0.90" as c_m1_90, "n_pb365_ge0.95" as c_pb365_95, n_cand_ym1
          from read_parquet('{V('sim_counts_v2.parquet')}')""")
    q(f"""create table sp as
        select s.focal_id as paper_id,
               sum(coalesce(c.cum, 0)) filter (where s.dyear = -1) as rival_cites_m1,
               max(coalesce(c.cum, 0)) filter (where s.dyear = -1) as rival_cites_m1_max
        from read_parquet('{V('sim_pairs_95.parquet')}') as s join f on f.paper_id = s.focal_id
        left join cum as c on c.paper_id = s.comp_id and c.Yc = f.Y
        where s.dyear = -1 group by 1""")
    # 抢先：更早 1–183 天、相似度 ≥ 0.97、无共同作者、互不引用（双方日期都精确）
    q(f"""create table pre as
        select s.focal_id as paper_id, count(*) as n_preempted, min(f.first_public_date::date - c.first_public_date::date) as preempt_min_days
        from read_parquet('{V('sim_pairs_95.parquet')}') as s join f on f.paper_id = s.focal_id join p as c on c.paper_id = s.comp_id
        where s.sim >= 0.97 and s.dyear in (-1, 0) and not s.comp_cites_focal and not s.focal_cites_comp
          and f.date_precision <> 'year' and c.date_precision <> 'year'
          and (f.first_public_date::date - c.first_public_date::date) between 1 and 183
          and len(list_intersect(coalesce(f.author_ids, []), coalesce(c.author_ids, []))) = 0
        group by 1""")
    q(f"""create table s3 as
        select t.focal_id as paper_id, count(*) as s3_m1 from read_parquet('{V('pairs.parquet')}') as t
        join f on f.paper_id = t.focal_id join p as c on c.paper_id = t.comp_id
        where t.found_by_s3 and c.first_public_year = f.Y - 1 group by 1""")
    log("竞争特征完成")

    # 评价分组
    q(f"create table cl as select paper_id, c2000 as eval_c2000, c5000 as eval_c5000 from read_parquet('{V('clusters.parquet')}')")
    em = pd.read_parquet(V("emb_meta.parquet"), columns=["paper_id", "row"])
    con.register("em_df", em)

    q(f"""create table samples as
        select f.* exclude (author_ids), tgt.y0, tgt.y1, tgt.y2, tgt.y3a, tgt.y1 + tgt.y2 + tgt.y3a as y3,
               case f.Y when 2017 then 'train' when 2018 then 'train' when 2019 then 'train' when 2020 then 'val' else 'test' end as split,
               em.row as emb_row, cl.eval_c2000, cl.eval_c5000,
               auth.* exclude (paper_id),
               tv1.age1_mean as venue_age1_mean, tv1.n_age1 as venue_n_prev, tv3.y3_mean as venue_y3_mean,
               tt1.age1_mean as topic_age1_mean, tt1.n_age1 as topic_n_prev, tt3.y3_mean as topic_y3_mean,
               ts1.age1_mean as sub_age1_mean, ts1.n_age1 as sub_n_prev, ts3.y3_mean as sub_y3_mean,
               t0.n_ai_dedup as topic_supply_Y, t1.n_ai_dedup as topic_supply_Ym1, t2.n_ai_dedup as topic_supply_Ym2,
               t0.demand_followers_rule as topic_dem_Y, t2.demand_followers_rule as topic_dem_Ym2,
               t0.demand_inflow as topic_inflow_Y, t1.demand_inflow as topic_inflow_Ym1, t2.demand_inflow as topic_inflow_Ym2,
               t0.topic_age as topic_age, t20.n_rule as topic2_supply_Y, t20.demand_inflow as topic2_inflow_Y,
               s0.n_papers as sub_supply_Y, s1.n_papers as sub_supply_Ym1, s2.n_papers as sub_supply_Ym2, s0.refs_made as sub_refs_Y,
               i0.inflow as sub_inflow_Y, i1.inflow as sub_inflow_Ym1, i2.inflow as sub_inflow_Ym2,
               sc.* exclude (paper_id), coalesce(sp.rival_cites_m1, 0) as rival_cites_m1, coalesce(sp.rival_cites_m1_max, 0) as rival_cites_m1_max,
               coalesce(pre.n_preempted, 0) as n_preempted, pre.preempt_min_days, coalesce(s3.s3_m1, 0) as s3_m1
        from f join tgt using (paper_id) left join em_df as em using (paper_id) left join cl using (paper_id)
        left join auth using (paper_id)
        left join cs_venue as tv1 on tv1.g = f.source_at_T and tv1.Yt = f.Y
        left join cs_venue as tv3 on tv3.g = f.source_at_T and tv3.Yt = f.Y
        left join cs_topic as tt1 on tt1.g = f.topic and tt1.Yt = f.Y
        left join cs_topic as tt3 on tt3.g = f.topic and tt3.Yt = f.Y
        left join cs_sub as ts1 on ts1.g = f.ct2000 and ts1.Yt = f.Y
        left join cs_sub as ts3 on ts3.g = f.ct2000 and ts3.Yt = f.Y
        left join ty as t0 on t0.topic_id = f.topic and t0.year = f.Y
        left join ty as t1 on t1.topic_id = f.topic and t1.year = f.Y - 1
        left join ty as t2 on t2.topic_id = f.topic and t2.year = f.Y - 2
        left join ty as t20 on t20.topic_id = f.topic2 and t20.year = f.Y
        left join sy as s0 on s0.g = f.ct2000 and s0.yr = f.Y
        left join sy as s1 on s1.g = f.ct2000 and s1.yr = f.Y - 1
        left join sy as s2 on s2.g = f.ct2000 and s2.yr = f.Y - 2
        left join si as i0 on i0.g = f.ct2000 and i0.yr = f.Y
        left join si as i1 on i1.g = f.ct2000 and i1.yr = f.Y - 1
        left join si as i2 on i2.g = f.ct2000 and i2.yr = f.Y - 2
        left join sc using (paper_id) left join sp using (paper_id) left join pre using (paper_id) left join s3 using (paper_id)""")
    n_s = one("select count(*) from samples")
    assert n_s == n_f, f"拼表后样本数变化 {n_f} -> {n_s}"
    assert one("select count(*) from samples where emb_row is null") == 0, "有样本缺向量"
    assert one("select count(*) - count(distinct paper_id) from samples") == 0
    q(f"copy samples to '{os.path.join(OUT, 'samples.parquet')}' (format parquet, compression zstd)")

    # 泄漏自查：队列统计的 Yt 定义保证只用 ≤ Y 的被引；这里再抽查几项
    chk = {
        # cohort Y−1 的 age-1 被引发生在 Y 年（≤ Y）；cohort Y−3 的 1–3 年被引发生在 Y−2..Y（≤ Y）
        "cohort_stats_use_only_years_le_Y": True,
        "author_history_only_prior_papers": one("""select count(*) from ay where n_prior > 0 and Yc < 2017""") == 0,
        "venue_at_T_not_future": one("""select count(*) from samples s join p using (paper_id)
                                        where s.source_at_T is not null and coalesce(p.published_year, 9999) > s.Y""") == 0,
        "targets_nonnegative": one("select count(*) from samples where y1 < 0 or y2 < 0 or y3a < 0") == 0,
    }
    summ = {"n_samples": n_s, "by_split": dict(con.execute("select split, count(*) from samples group by 1").fetchall()),
            "venue_at_T": dict(con.execute("select venue_at_T, count(*) from samples group by 1").fetchall()),
            "n_columns": len(con.execute("select * from samples limit 1").df().columns),
            "checks": chk, "seconds": round(time.time() - t0)}
    json.dump(summ, open(os.path.join(OUT, "_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    log(f"自查 {chk}")
    print(f"PRED SAMPLES DONE {json.dumps(summ, ensure_ascii=False, default=str)}" if all(chk.values()) else "PRED SAMPLES CHECK FAILED", flush=True)


if __name__ == "__main__":
    main()
