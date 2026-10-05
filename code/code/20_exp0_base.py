"""
【20】实验 0 子集（一）：AI 论文样本、预印本与正式版合并、T1/T2/T3/T4                阶段 2：实验 0

用途
  从总库切出实验 0 的 AI 论文样本（规则 R-a～R-g，数据说明第 4 节），合并同一篇论文的不同版本，生成论文表、
  逐年被引、引用边、方向-年份表。

规则
  候选：主方向子领域 1702；2016–2024；类型 article/book-chapter/preprint；非撤稿/paratext；有标题
  版本合并：DOI 相同 / arXiv 编号相同 / 标题相同（≥20 字符）且有共同作者，并要求一方是预印本、同标题记录 ≤5 条、
           年份相差 ≤3、不是不同的 arXiv 编号；连通分量归组；代表记录 article > book-chapter > preprint
  样本 is_sample：合并后参考文献 ≥5 篇；焦点 is_focal：样本中首次公开 2017–2021

输入
  union/detail、union/lite

输出（/path/to/mpcc/subsets/exp0/，本地副本 MPCC/data/exp0/）
  versions.parquet              版本 → 合并后论文（1,357,124 → 1,227,247）
  papers.parquet                T1：合并后论文 1,227,247 篇（样本 732,846、焦点 391,439）
  paper_year.parquet            T3：逐年被引（各版本相加）1,870,432 行
  paper_year_in_corpus.parquet  样本内被引（由 T2 统计）
  citations.parquet             T2：引用边 10,721,097 条（施引方来自全库，映射到合并后论文并去重，含自引标记）
  topic_year.parquet            T4：全部 4,516 个方向 × 年份的供给/需求（含跨域加权）692,942 行
  _summary.json                 各步骤计数与 6 项校验
  _work.duckdb                  中间工作库（不下载）

用法
  python 20_exp0_base.py [--threads 160]      全量（约 2 分钟）
  python 20_exp0_base.py --test               冒烟测试：只读少量分片，输出到 subsets/exp0_test/

运行记录
  2026-10-02  第 1 次：标题+作者合并过宽，把同标题的连载专栏/期刊目录/数据集系列串成 84 个版本 → 收紧规则；
              第 2 次：同一组作者两篇同名不同论文被合并 → 增加"不同 arXiv 编号不合并"；
              第 3 次（最终）：最大一组 10 个版本，抽查均为同一论文的多个版本；6 项校验全部通过
  原文件名：02_build_exp0.py
"""
import argparse
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

DETAIL = os.path.join(C.UNION_DIR, "detail", "*.parquet")
LITE = os.path.join(C.UNION_DIR, "lite", "*.parquet")
OUT = os.path.join(C.SUBSET_DIR, C.EXP0)
TYPES = ("article", "book-chapter", "preprint")
TYPE_RANK = "case type when 'article' then 1 when 'book-chapter' then 2 else 3 end"
summary = {}
t00 = time.time()


def log(msg):
    print(f"[{time.time() - t00:7.0f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--test", action="store_true", help="冒烟测试: 只读少量分片, 输出到 subsets/exp0_test/")
    args = ap.parse_args()
    global DETAIL, LITE, OUT
    if args.test:
        DETAIL = os.path.join(C.UNION_DIR, "detail", "w01[0-1]_0000[0-1].parquet")
        LITE = os.path.join(C.UNION_DIR, "lite", "w01[0-1]_0000*.parquet")
        OUT = os.path.join(C.SUBSET_DIR, "exp0_test")
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(os.path.join(OUT, "_work.duckdb"))
    con.execute(f"SET threads={args.threads}; SET memory_limit='400GB'; SET temp_directory='{C.TMP_DIR}'; "
                "SET preserve_insertion_order=false")
    q = lambda s: con.execute(s)
    one = lambda s: con.execute(s).fetchone()[0]

    # ------------------------------------------------------------------ 1. 候选论文 (版本层面)
    types = ",".join(f"'{t}'" for t in TYPES)
    q(f"""create or replace table cand as
        select *, lower(regexp_replace(lower(coalesce(title, '')), '[^a-z0-9]', '', 'g')) as title_norm
        from read_parquet('{DETAIL}')
        where pt_subfield = {C.AI_SUBFIELD} and publication_year between {C.DETAIL_YEAR_MIN} and {C.DETAIL_YEAR_MAX}
          and type in ({types}) and not coalesce(is_retracted, false) and not coalesce(is_paratext, false)
          and title is not null and title <> ''""")
    summary["candidates"] = one("select count(*) from cand")
    log(f"候选版本 {summary['candidates']:,}")

    # ------------------------------------------------------------------ 2. 版本合并 (连通分量)
    q("""create or replace table edges as
        with d as (select id, lower(doi) k from cand where doi is not null),
             a as (select id, arxiv_id k from cand where arxiv_id is not null)
        select x.id a, y.id b, 'doi' via from d x join d y on x.k = y.k and x.id < y.id
        union all
        select x.id, y.id, 'arxiv' from a x join a y on x.k = y.k and x.id < y.id
        union all
        -- 标题+作者匹配 (2026-10-02 收紧): 首轮出现把"同标题的连载专栏/期刊目录/数据集系列"串成 84 个版本的过度合并.
        -- 真正的版本对几乎都是"预印本 + 正式版", 因此要求: 其中一方是预印本; 同标题记录不超过 5 条; 年份相差不超过 3 年
        select x.id, y.id, 'title_author' from cand x join cand y
          on x.title_norm = y.title_norm and x.id < y.id
         and length(x.title_norm) >= 20 and list_has_any(x.author_ids, y.author_ids)
         and (x.type = 'preprint' or y.type = 'preprint' or x.arxiv_id is not null or y.arxiv_id is not null)
         and abs(x.publication_year - y.publication_year) <= 3
         and not (x.arxiv_id is not null and y.arxiv_id is not null and x.arxiv_id <> y.arxiv_id)  -- 不同 arXiv 编号必为不同论文
         and x.title_norm in (select title_norm from cand group by 1 having count(*) <= 5)""")
    summary["merge_edges"] = dict(con.execute("select via, count(*) from edges group by 1").fetchall())
    log(f"合并边 {summary['merge_edges']}")
    ids = con.execute("select id from cand order by id").fetchnumpy()["id"]
    e = con.execute("select a, b from edges").fetchnumpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    ai, bi = pos[e["a"]].to_numpy(), pos[e["b"]].to_numpy()
    g = coo_matrix((np.ones(len(ai), dtype=np.int8), (ai, bi)), shape=(len(ids), len(ids)))
    ncomp, lab = connected_components(g, directed=False)
    con.register("comp_df", pd.DataFrame({"member_id": ids, "comp": lab}))
    q("create or replace table comp as select * from comp_df")
    summary["clusters"] = int(ncomp)
    summary["clusters_multi_version"] = one("select count(*) from (select comp from comp group by 1 having count(*) > 1)")
    summary["largest_cluster"] = one("select max(n) from (select count(*) n from comp group by comp)")
    log(f"合并后论文 {ncomp:,} (多版本 {summary['clusters_multi_version']:,}, 最大一组 {summary['largest_cluster']} 个版本)")

    # 代表记录: article > book-chapter > preprint, 同级取被引多者, 再取 id 小者
    q(f"""create or replace table versions as
        select c.member_id, c.comp,
               first_value(c.member_id) over (partition by c.comp order by {TYPE_RANK.replace('type', 'v.type')},
                     v.cited_by_count desc nulls last, c.member_id) as paper_id,
               v.type, v.publication_date, v.arxiv_id
        from comp c join cand v on v.id = c.member_id""")
    q(f"copy (select member_id, paper_id, type, publication_date, arxiv_id from versions) "
      f"to '{OUT}/versions.parquet' (format parquet, compression zstd)")

    # ------------------------------------------------------------------ 3. T1 合并后论文
    q("""create or replace table merged as
        select v.paper_id,
               count(*) n_versions, list(v.member_id order by v.member_id) member_ids, list(v.type) member_types,
               min(coalesce(c.publication_date, cast(c.publication_year as varchar) || '-01-01')) first_public_date,
               bool_or(c.type = 'preprint' or c.arxiv_id is not null) has_preprint,
               min(case when c.type = 'preprint' then c.publication_date end) preprint_date,
               any_value(c.arxiv_id) filter (where c.arxiv_id is not null) arxiv_id_any,
               list_distinct(flatten(list(c.refs))) refs_union,
               list_distinct(flatten(list(c.author_ids))) authors_union,
               any_value(c.abstract) filter (where c.abstract is not null) abstract_any,
               bool_or(c.is_oa) any_oa, max(c.n_grants) n_grants_max
        from versions v join cand c on c.id = v.member_id group by v.paper_id""")
    q(f"""create or replace table papers as
        select m.paper_id, p.doi, coalesce(p.arxiv_id, m.arxiv_id_any) arxiv_id, p.title,
               coalesce(p.abstract, m.abstract_any) abstract, (coalesce(p.abstract, m.abstract_any) is not null) has_abstract,
               p.keywords, p.language, p.type, m.n_versions, m.member_ids, m.member_types,
               m.first_public_date, cast(substr(m.first_public_date, 1, 4) as smallint) first_public_year,
               p.publication_date published_date, p.publication_year published_year,
               m.has_preprint, m.preprint_date,
               case when m.preprint_date is not null and p.publication_date is not null
                    then date_diff('day', cast(m.preprint_date as date), cast(p.publication_date as date)) end preprint_lead_days,
               p.pt_topic, p.pt_subfield, p.pt_field, p.pt_score, p.topic_ids, p.topic_scores, p.topic_subfields, p.topic_fields,
               p.author_ids, m.authors_union, p.n_authors, p.institution_ids, p.institution_types, p.n_institutions,
               p.countries, p.n_countries,
               len(m.refs_union) n_refs, m.refs_union refs,
               p.is_oa, m.any_oa, p.oa_status, p.source_id, p.source_name, p.source_type,
               p.locations_count, p.loc_source_types, p.loc_versions,
               p.n_grants, (m.n_grants_max > 0) has_funding, p.funder_ids,
               p.concept_names, p.concept_scores,
               p.first_page, p.last_page,
               p.fwci, p.cnp_value, p.cnp_top1, p.cnp_top10      -- 含未来信息, 只做评价
        from merged m join cand p on p.id = m.paper_id""")
    q(f"""alter table papers add column is_sample boolean;
         update papers set is_sample = (first_public_year between {C.DETAIL_YEAR_MIN} and {C.DETAIL_YEAR_MAX}
                                        and n_refs >= 5 and title is not null);
         alter table papers add column is_focal boolean;
         update papers set is_focal = is_sample and first_public_year between 2017 and 2021""")
    summary["papers"] = one("select count(*) from papers")
    summary["papers_sample"] = one("select count(*) from papers where is_sample")
    summary["papers_focal"] = one("select count(*) from papers where is_focal")
    summary["focal_by_year"] = dict(con.execute(
        "select first_public_year, count(*) from papers where is_focal group by 1 order by 1").fetchall())
    log(f"T1 论文 {summary['papers']:,}; 样本 {summary['papers_sample']:,}; 焦点 {summary['papers_focal']:,} {summary['focal_by_year']}")

    # ------------------------------------------------------------------ 4. T3 逐年被引 (各版本相加)
    q("""create or replace table paper_year as
        select v.paper_id, y.year, sum(y.cites)::integer cites
        from versions v join (select id, unnest(cby_year) as year, unnest(cby_count) cites from cand) y on y.id = v.member_id
        group by 1, 2""")
    q(f"copy (select * from paper_year order by paper_id, year) to '{OUT}/paper_year.parquet' (format parquet, compression zstd)")
    # 校验汇总正确: 合并后逐年被引之和 = 各版本原始 counts_by_year 之和 (逐篇)
    chk = one("""select count(*) from (select v.paper_id, sum(list_sum(c.cby_count)) m from versions v
                   join cand c on c.id = v.member_id group by 1) a
                 full join (select paper_id, sum(cites) s from paper_year group by 1) b using (paper_id)
                 where coalesce(a.m, 0) <> coalesce(b.s, 0)""")
    summary["check_paper_year_mismatch"] = chk
    # 原始数据的已知不一致 (快照中 counts_by_year 与 cited_by_count 更新不同步), 只记录不修改
    summary["raw_quirk_cby_gt_total"] = one("select count(*) from cand where list_sum(cby_count) > cited_by_count")
    log(f"T3 逐年被引 {one('select count(*) from paper_year'):,} 行; 汇总不符 {chk} 篇 (应为 0); "
        f"原始数据中逐年之和 > 总被引的版本 {summary['raw_quirk_cby_gt_total']} 个 (OpenAlex 自身不一致, 不影响)")

    # ------------------------------------------------------------------ 5. T2 引用边 (施引方来自全库)
    q(f"""create or replace table cites_raw as
        select l.id citing_member, l.year citing_year, l.pt_topic citing_topic, l.pt_subfield citing_subfield,
               v.paper_id cited_id
        from (select id, year, pt_topic, pt_subfield, unnest(refs) as ref from read_parquet('{LITE}')
              where year between {C.DETAIL_YEAR_MIN} and {C.DETAIL_YEAR_MAX} and refs is not null) l
        join versions v on v.member_id = l.ref""")
    summary["cites_raw"] = one("select count(*) from cites_raw")
    log(f"引用边 (版本层面) {summary['cites_raw']:,}")
    # 施引方若是 AI 论文的某个版本, 映射到合并后论文; 去掉同一论文版本间的互引; 去重
    q("""create or replace table citations as
        select coalesce(v.paper_id, r.citing_member) citing_id, min(r.citing_year) citing_year,
               any_value(r.citing_topic) citing_topic, any_value(r.citing_subfield) citing_subfield,
               r.cited_id, bool_or(v.paper_id is not null) citing_is_ai_candidate
        from cites_raw r left join versions v on v.member_id = r.citing_member
        where coalesce(v.paper_id, r.citing_member) <> r.cited_id
        group by 1, r.cited_id""")
    # 自引: 施引论文作者 (全库 lite) 与被引论文 (所有版本) 作者有交集
    q(f"""create or replace table citing_auth as
        select l.id, l.author_ids from read_parquet('{LITE}') l
        semi join (select distinct citing_id id from citations) c on c.id = l.id""")
    q("""create or replace table citations2 as
        select c.*, coalesce(list_has_any(a.author_ids, p.authors_union), false) is_self_cite
        from citations c left join citing_auth a on a.id = c.citing_id join papers p on p.paper_id = c.cited_id""")
    q(f"copy (select * from citations2) to '{OUT}/citations.parquet' (format parquet, compression zstd)")
    summary["citations"] = one("select count(*) from citations2")
    summary["self_cite_share"] = one("select avg(is_self_cite::int) from citations2")
    log(f"T2 引用边 {summary['citations']:,} (自引 {summary['self_cite_share']:.3f})")

    # 被引论文都在 T1 里; 用于 T1 的样本内被引列
    q("""create or replace table in_corpus as
        select cited_id paper_id, citing_year as year, count(*) cites_in_corpus from citations2 group by 1, 2""")
    q(f"copy (select * from in_corpus) to '{OUT}/paper_year_in_corpus.parquet' (format parquet, compression zstd)")

    # ------------------------------------------------------------------ 6. T4 方向 x 年份 (全部方向)
    rule = (f"type in ({types}) and not coalesce(is_retracted, false) and not coalesce(is_paratext, false) "
            f"and coalesce(n_refs, 0) >= 5")
    q(f"""create or replace table ty_supply as
        select pt_topic topic_id, year,
               count(*) n_all, count(*) filter (where {rule}) n_rule,
               sum(n_refs) filter (where {rule}) refs_sum_rule
        from read_parquet('{LITE}') where pt_topic is not null and year is not null group by 1, 2""")
    q(f"""create or replace table ty_supply_w as
        select topic topic_id, year, sum(sc / tot) n_rule_weighted from (
          select unnest(topic_ids) topic, unnest(topic_scores) sc, list_sum(topic_scores) tot, year
          from read_parquet('{LITE}') where {rule} and len(topic_ids) > 0 and list_sum(topic_scores) > 0)
        group by 1, 2""")
    log("T4 供给完成")
    q(f"""create or replace table ty_demand as
        select pt_topic topic_id, year, sum(cnt)::bigint demand_inflow from (
          select pt_topic, unnest(cby_year) as year, unnest(cby_count) cnt from read_parquet('{LITE}')
          where pt_topic is not null and len(cby_year) > 0) group by 1, 2""")
    q(f"""create or replace table ty_demand_w as
        select topic topic_id, year, sum(sc / tot * cnt) demand_inflow_weighted from (
          select topic, sc, tot, unnest(cby_year) as year, unnest(cby_count) cnt from (
            select unnest(topic_ids) topic, unnest(topic_scores) sc, list_sum(topic_scores) tot, cby_year, cby_count
            from read_parquet('{LITE}') where len(topic_ids) > 0 and len(cby_year) > 0 and list_sum(topic_scores) > 0))
        group by 1, 2""")
    log("T4 需求完成")
    q("""create or replace table ty_ai as
        select pt_topic topic_id, first_public_year as year, count(*) n_ai_dedup, sum(n_refs) refs_sum_ai_dedup
        from papers where is_sample group by 1, 2""")
    q("""create or replace table ty_age as select pt_topic topic_id, min(year) first_year
         from read_parquet('""" + LITE + """') where pt_topic is not null and year >= 1900 group by 1""")
    q(f"""copy (
        select coalesce(s.topic_id, d.topic_id) topic_id, coalesce(s.year, d.year) as year,
               s.n_all, s.n_rule, w.n_rule_weighted, a.n_ai_dedup,
               s.refs_sum_rule demand_followers_rule, a.refs_sum_ai_dedup demand_followers_ai_dedup,
               d.demand_inflow, dw.demand_inflow_weighted,
               coalesce(s.year, d.year) - g.first_year topic_age
        from ty_supply s full join ty_demand d on s.topic_id = d.topic_id and s.year = d.year
        left join ty_supply_w w on w.topic_id = coalesce(s.topic_id, d.topic_id) and w.year = coalesce(s.year, d.year)
        left join ty_demand_w dw on dw.topic_id = coalesce(s.topic_id, d.topic_id) and dw.year = coalesce(s.year, d.year)
        left join ty_ai a on a.topic_id = coalesce(s.topic_id, d.topic_id) and a.year = coalesce(s.year, d.year)
        left join ty_age g on g.topic_id = coalesce(s.topic_id, d.topic_id)
        order by 1, 2) to '{OUT}/topic_year.parquet' (format parquet, compression zstd)""")
    summary["topic_year_rows"] = one(f"select count(*) from read_parquet('{OUT}/topic_year.parquet')")
    summary["topics"] = one(f"select count(distinct topic_id) from read_parquet('{OUT}/topic_year.parquet')")
    log(f"T4 {summary['topic_year_rows']:,} 行, {summary['topics']} 个方向")

    # ------------------------------------------------------------------ 7. 写出 T1 + 校验
    q(f"""copy (select p.*, coalesce(t.tot, 0) cited_by_count_merged from papers p
              left join (select paper_id, sum(cites) tot from paper_year group by 1) t using (paper_id))
          to '{OUT}/papers.parquet' (format parquet, compression zstd)""")
    checks = {
        "versions_unique": one("select count(*) = count(distinct member_id) from versions"),
        "papers_unique": one("select count(*) = count(distinct paper_id) from papers"),
        "every_member_mapped": one("select count(*) from cand anti join versions v on v.member_id = cand.id") == 0,
        "citations_cited_in_papers": one("select count(*) from citations2 anti join papers p on p.paper_id = citations2.cited_id") == 0,
        "no_self_version_cites": one("select count(*) from citations2 where citing_id = cited_id") == 0,
        "paper_year_matches_versions": summary["check_paper_year_mismatch"] == 0,
    }
    summary["checks"] = checks
    summary["seconds"] = round(time.time() - t00)
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    log(f"校验 {checks}")
    print("EXP0 BASE DONE" if all(checks.values()) else "EXP0 BASE DONE WITH CHECK FAILURES", flush=True)


if __name__ == "__main__":
    main()
