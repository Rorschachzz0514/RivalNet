"""
【15】总库质量报告：规模、字段覆盖率、AI 论文的摘要/方向/arXiv/类型分布           阶段 1：总库

输入
  union/detail、union/lite（参数可指定其他目录，如测试输出）

输出
  标准输出（日志 /path/to/mpcc/logs/check_scan.log）；要点已写入 数据说明 6.3 节

用法
  python 15_check_union_report.py [总库目录]

运行记录
  2026-10-02  AI 子领域 1,502,914 篇：有摘要 67.5%、≥2 个方向 83.8%、有 arXiv 编号 22.4%、参考文献 ≥5 篇 53.0%；
              类型 article 92.1 万 / preprint 28.7 万 / book-chapter 15.1 万，另有 dataset 等非研究论文类型
  原文件名：check_scan.py
"""
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

d = sys.argv[1] if len(sys.argv) > 1 else C.UNION_DIR
F, A = f"'{d}/detail/*.parquet'", f"'{d}/lite/*.parquet'"
c = duckdb.connect()
c.execute(f"SET temp_directory='{C.TMP_DIR}'")


def q(title, sql):
    print(f"\n== {title}")
    print(c.sql(sql).df().to_string(index=False))


q("AI 子领域 (1702) 字段覆盖率", f"""
  select count(*) n, avg((abstract is not null)::int) has_abstract, avg((len(topic_ids) >= 2)::int) ge2_topics,
         avg((arxiv_id is not null)::int) has_arxiv, avg((doi is not null)::int) has_doi,
         avg(n_refs) mean_refs, avg((n_refs >= 5)::int) refs_ge5, avg((len(refs) = n_refs)::int) refs_list_complete,
         avg((len(cby_year) > 0)::int) has_cby
  from {F} where pt_subfield = {C.AI_SUBFIELD}""")
q("AI 论文类型", f"select type, count(*) n from {F} where pt_subfield = {C.AI_SUBFIELD} group by 1 order by 2 desc")
q("AI 论文按年份 (含摘要比例、article 且参考文献>=5 的数量)", f"""
  select publication_year y, count(*) n, avg((abstract is not null)::int) has_abstract,
         sum((type = 'article' and n_refs >= 5)::int) article_ref5
  from {F} where pt_subfield = {C.AI_SUBFIELD} group by 1 order by 1""")
q("detail 各领域", f"select pt_field, count(*) n from {F} group by 1 order by 2 desc")
q("AI 样例", f"""select title, arxiv_id, source_name, publication_date, topic_ids, cby_year[1:3] cy, cby_count[1:3] cc
  from {F} where pt_subfield = {C.AI_SUBFIELD} and arxiv_id is not null limit 3""")
q("lite 概况", f"""select count(*) n, avg((year is null)::int) no_year, avg((pt_topic is null)::int) no_topic,
  avg((refs is not null)::int) has_refs_list, count(distinct pt_topic) n_topics from {A}""")
q("detail 新增字段覆盖率 (按领域)", f"""
  select pt_field, count(*) n, avg((len(concept_ids) > 0)::int) has_concepts, avg((len(mesh_names) > 0)::int) has_mesh,
         avg((len(institution_types) > 0)::int) has_inst_type, avg((list_contains(institution_types, 'company'))::int) any_company
  from {F} group by 1 order by 2 desc""")
q("lite 作者列表覆盖 (按年代)", f"""
  select case when year < 2000 then '<2000' when year < 2015 then '2000-2014' else '>=2015' end era,
         count(*) n, avg((len(author_ids) > 0)::int) has_authors, avg((refs is not null)::int) has_refs_list
  from {A} group by 1 order by 1""")
