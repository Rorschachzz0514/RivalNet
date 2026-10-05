"""
【26】实验 0 第二版数据（v2）：剔除无效条目、补合并漏掉的重复记录、标记日期精度、标记发表渠道          阶段 2：实验 0

用途
  实验 0 第一版发现三个数据问题（见 Experiment_0/实验0结果解读_不同版本数据.md 与 08_duplicates.md）：
    (1) 无效条目：书的目录、references、introduction 等附属章节，或标题过短 / 为空，不是真正的论文；
    (2) 漏合并的同一篇论文：R-d 只合并"标题完全相同"的版本，改了标题的版本（arXiv 版改题投会议）漏掉；
    (3) 日期精度：38% 的焦点论文首次公开日期是 1 月 1 日——OpenAlex 只知道年份时的占位值，
        "发表前 365 天"之类按天计算的量对这些论文不可靠。
  本脚本在实验 0 子集上修正，生成 subsets/exp0_v2/（原 subsets/exp0 不改动）：
    无效条目   规范化标题（小写、去标点，保留各语种文字）< 10 字符，或整题是 references / bibliography / index /
               preface / contents / introduction / conclusion / appendix / editorial / erratum 等 → 从全部表中剔除
    补合并     候选：pairs 中 SPECTER2 相似度 ≥ 0.95 的论文对 + 样本内规范化标题完全相同（同题 ≤ 5 条）的论文对；
               条件：有共同作者，首次公开年份相差 ≤ 3，两题中不同的词里没有数字 / 罗马数字（防止把"Part IV"和"Part V"
               这类系列论文并在一起），且 [作者重合度 ≥ 0.5 且标题 Jaro-Winkler ≥ 0.9] 或 [标题完全相同]；
               用连通分量归组，组大小 > 6 的整组放弃合并（防止串联）
               代表记录：渠道 期刊/会议 > 论文集 > arXiv > 其他仓储 > 电子书平台 > 无渠道，再按 article > book-chapter
               > preprint，再按被引多者；首次公开年份取最早；日期取该年份内最早的精确日期（没有则为该年 1 月 1 日、
               标记为不精确）；逐年被引相加；引用边两端重映射、去重、去掉自环
    日期精度   OpenAlex 只知道年份时把日期填成 1 月 1 日（arXiv 记录也大量如此）。只在精确日期中取最早：
               版本日期不是 1 月 1 日 → 精确到日；新式 arXiv 编号 YYMM.NNNNN → 精确到月（当月 15 日）；
               都没有 → 只精确到年（记为 1 月 1 日）。date_precision = day / month / year，date_precise = 非 year
    发表渠道   venue_class = journal / conference / book series / arxiv / repository / ebook / none
               （arXiv：代表记录的渠道是 arXiv，或有 arXiv 编号且没有期刊/会议/论文集渠道）
  样本 / 焦点的定义与第一版相同（样本：2016–2024、参考文献 ≥5、有标题；焦点：样本中 2017–2021），再去掉无效条目。
  焦点论文的"主分析集"（is_focal_main）由实验脚本按 venue_class 选择，这里只给出标记。

输入（subsets/exp0/）
  papers、versions、citations、paper_year、paper_year_in_corpus、topic_year、pairs、emb_meta（+ emb_sample.npy）

输出（subsets/exp0_v2/）
  papers.parquet                 v2 论文表：第一版的列 + date_precision、date_precise
                                 + venue_class + n_merged_extra（本次补合并进来的记录数）
  merge_map.parquet              old_id → new_id（被补合并或保留的全部论文；无效条目不在其中）
  junk.parquet                   被剔除的无效条目（paper_id、title）
  dup_edges.parquet              补合并用到的论文对及判定指标（供核查）
  versions / citations / paper_year / paper_year_in_corpus / pairs .parquet   按 merge_map 重映射
  topic_year.parquet             n_ai_dedup 按 v2 样本重算（原值保留为 n_ai_dedup_v1），其余列不变
  emb_meta.parquet               v2 样本论文 → 原 emb_sample.npy 的行号（row 列），向量不复制
  _summary.json                  各项计数与自查结果

用法
  python 26_exp0_v2_clean.py

运行记录
  2026-10-02  建立
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

SRC = os.path.join(C.SUBSET_DIR, C.EXP0)
OUT = os.path.join(C.SUBSET_DIR, C.EXP0_V2)
NORM = r"trim(regexp_replace(lower(coalesce(title, '')), '[^\p{L}\p{N}]+', ' ', 'g'))"
JUNK = ("^(references?|bibliography|index|author index|subject index|preface|foreword|contents|table of contents|"
        "front matter|back matter|frontmatter|backmatter|editorial|introduction|conclusions?|appendix.*|glossary|"
        "abbreviations|acknowledg.*|title page|copyright|list of (figures|tables)|notes|about the authors?|"
        "editorial board|cover|erratum|corrigendum|abstracts?|withdrawn|retracted|front cover|back cover|"
        "other topics|background|summary|bibliographie|literaturverzeichnis|inhaltsverzeichnis|vorwort|"
        "einleitung|index of authors|keyword index|contributors|list of contributors|"
        "(chapter|part|section) [0-9ivx]+)$")
ROMAN_NUM = r"^([0-9]+|i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3})$"
MAX_GROUP = 6


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    S = lambda n: os.path.join(SRC, n)
    O = lambda n: os.path.join(OUT, n)
    con = duckdb.connect()
    con.execute("SET threads=128; SET memory_limit='300GB'; SET temp_directory='/path/to/mpcc/tmp'; SET preserve_insertion_order=false")
    q = lambda s: con.execute(s)
    one = lambda s: con.execute(s).fetchone()[0]
    summ = {}

    q(f"create table p as select *, {NORM} as tnorm from read_parquet('{S('papers.parquet')}')")
    summ["papers_v1"] = one("select count(*) from p")

    # ---------------- 1. 无效条目
    q(f"""create table junk as select paper_id, title, is_sample, is_focal from p
          where length(tnorm) < 10 or regexp_matches(tnorm, '{JUNK}')""")
    summ["junk_all"] = one("select count(*) from junk")
    summ["junk_sample"] = one("select count(*) from junk where is_sample")
    summ["junk_focal"] = one("select count(*) from junk where is_focal")
    q(f"copy (select paper_id, title, is_sample, is_focal from junk) to '{O('junk.parquet')}' (format parquet)")
    q("create table pk as select * from p where paper_id not in (select paper_id from junk)")
    log(f"无效条目 {summ['junk_all']:,}（样本 {summ['junk_sample']:,}，焦点 {summ['junk_focal']:,}）")

    # ---------------- 2. 补合并候选
    q(f"""create table cand as
        select least(focal_id, comp_id) as a, greatest(focal_id, comp_id) as b from read_parquet('{S('pairs.parquet')}')
        where sem_sim >= 0.95
        union
        select x.paper_id as a, y.paper_id as b from
          (select paper_id, tnorm from pk where is_sample and tnorm in
             (select tnorm from pk where is_sample group by 1 having count(*) between 2 and 5)) as x
          join (select paper_id, tnorm from pk where is_sample) as y on x.tnorm = y.tnorm and x.paper_id < y.paper_id""")
    q(f"""create table edges as
        select * from (
          select c.a, c.b, x.tnorm as ta, y.tnorm as tb,
                 jaro_winkler_similarity(x.tnorm, y.tnorm) as title_sim,
                 (x.tnorm = y.tnorm) as same_title,
                 len(list_intersect(coalesce(x.author_ids, []), coalesce(y.author_ids, []))) as n_shared,
                 len(list_intersect(coalesce(x.author_ids, []), coalesce(y.author_ids, [])))
                   / greatest(len(list_distinct(list_concat(coalesce(x.author_ids, []), coalesce(y.author_ids, [])))), 1) as auth_jacc,
                 abs(x.first_public_year - y.first_public_year) as dyear,
                 len(list_filter(list_concat(
                       list_filter(string_split(x.tnorm, ' '), w -> not list_contains(string_split(y.tnorm, ' '), w)),
                       list_filter(string_split(y.tnorm, ' '), w -> not list_contains(string_split(x.tnorm, ' '), w))),
                     w -> regexp_matches(w, '{ROMAN_NUM}'))) as n_num_diff
          from cand as c join pk as x on x.paper_id = c.a join pk as y on y.paper_id = c.b
          where x.is_sample and y.is_sample)
        where n_shared > 0 and dyear <= 3 and n_num_diff = 0
          and ((auth_jacc >= 0.5 and title_sim >= 0.9) or same_title)""")
    q(f"copy edges to '{O('dup_edges.parquet')}' (format parquet)")
    ed = con.execute("select a, b from edges").df()
    summ["dup_candidate_pairs"] = one("select count(*) from cand")
    summ["dup_edges"] = len(ed)
    uf = UF()
    for a, b in zip(ed.a, ed.b):
        uf.union(a, b)
    comp = pd.DataFrame({"paper_id": list(uf.p.keys())})
    comp["root"] = comp.paper_id.map(uf.find)
    size = comp.groupby("root").paper_id.transform("size")
    summ["dup_groups_too_large_dropped"] = int(comp[size > MAX_GROUP].root.nunique())
    comp = comp[size <= MAX_GROUP]
    summ["dup_groups"] = int(comp.root.nunique())
    summ["dup_papers_in_groups"] = len(comp)
    con.register("comp_df", comp)
    log(f"补合并：候选 {summ['dup_candidate_pairs']:,} 对，满足条件 {len(ed):,} 对 → {summ['dup_groups']:,} 组 "
        f"{len(comp):,} 篇（放弃过大的组 {summ['dup_groups_too_large_dropped']}）")

    # 代表记录
    q("""create table vc as select paper_id,
           case when source_type in ('journal', 'conference') then source_type
                when source_type = 'book series' then 'book series'
                when source_name ilike 'arxiv%' or (arxiv_id is not null and coalesce(source_type, '') not in ('journal', 'conference', 'book series')) then 'arxiv'
                when source_type = 'repository' then 'repository'
                when source_type = 'ebook platform' then 'ebook'
                else 'none' end as venue_class
         from pk""")
    q("""create table grp as
        select c.paper_id, c.root, row_number() over (partition by c.root order by
                 case v.venue_class when 'journal' then 0 when 'conference' then 0 when 'book series' then 1
                      when 'arxiv' then 2 when 'repository' then 3 when 'ebook' then 4 else 5 end,
                 case p.type when 'article' then 0 when 'book-chapter' then 1 else 2 end,
                 p.cited_by_count_merged desc, p.paper_id) as rk
        from comp_df as c join pk as p using (paper_id) join vc as v using (paper_id)""")
    q("""create table mm as
        select g.paper_id as old_id, r.paper_id as new_id from grp as g join grp as r on r.root = g.root and r.rk = 1
        union all
        select paper_id as old_id, paper_id as new_id from pk where paper_id not in (select paper_id from grp)""")
    assert one("select count(*) - count(distinct old_id) from mm") == 0
    q(f"copy mm to '{O('merge_map.parquet')}' (format parquet)")

    # ---------------- 3. v2 论文表
    q("""create table agg as
        select m.new_id as paper_id, min(p.first_public_year) as y_min, count(*) - 1 as n_merged_extra,
               bool_or(p.has_preprint) as has_preprint, bool_or(p.any_oa) as any_oa, bool_or(p.has_funding) as has_funding,
               sum(p.n_versions) as n_versions, max(p.n_refs) as n_refs_max, sum(p.cited_by_count_merged) as cbc,
               bool_or(p.is_sample) as any_sample
        from mm as m join pk as p on p.paper_id = m.old_id group by 1""")
    # 首次公开日期：OpenAlex 只知道年份时把日期填成 1 月 1 日（arXiv 记录也大量如此），第一版取各版本最小日期时
    # 会被这类占位值覆盖。这里只在"精确日期"中取最早：版本日期不是 1 月 1 日 → 精确到日；
    # 新式 arXiv 编号 YYMM.NNNNN 的前四位是提交年月 → 精确到月（取当月 15 日）。最早的精确日期必须落在最早年份内，
    # 否则只知道年份（日期记为该年 1 月 1 日，精度 = 年）。
    q(f"""create table vv as select m.new_id as paper_id, v.publication_date::date as d, v.arxiv_id
          from read_parquet('{S('versions.parquet')}') as v join mm as m on m.old_id = v.paper_id""")
    q("""create table dcand as
        select paper_id, d, 0 as prec from vv where d is not null and strftime(d, '%m-%d') <> '01-01'
        union all
        select paper_id, make_date(2000 + cast(substr(arxiv_id, 1, 2) as int), cast(substr(arxiv_id, 3, 2) as int), 15) as d, 1 as prec
        from vv where regexp_matches(coalesce(arxiv_id, ''), '^[0-9]{4}[.][0-9]{4,5}')
          and cast(substr(arxiv_id, 3, 2) as int) between 1 and 12""")
    q("""create table dt as
        select paper_id, d, case prec when 0 then 'day' else 'month' end as precision from (
          select c.paper_id, c.d, c.prec, row_number() over (partition by c.paper_id order by c.d, c.prec) as rn
          from dcand as c join agg as a on a.paper_id = c.paper_id where year(c.d) = a.y_min) where rn = 1""")
    q("""create table p2 as
        select p.* exclude (first_public_year, first_public_date, has_preprint, any_oa, has_funding, n_versions,
                            cited_by_count_merged, is_sample, is_focal, tnorm),
               a.y_min as first_public_year,
               coalesce(dt.d, make_date(a.y_min, 1, 1)) as first_public_date,
               coalesce(dt.precision, 'year') as date_precision, (dt.d is not null) as date_precise,
               a.has_preprint, a.any_oa, a.has_funding, a.n_versions, a.cbc as cited_by_count_merged,
               a.n_merged_extra, v.venue_class
        from agg as a join pk as p on p.paper_id = a.paper_id join vc as v on v.paper_id = a.paper_id
        left join dt on dt.paper_id = a.paper_id""")
    # 样本 / 焦点（定义同第一版）
    q("""alter table p2 add column is_sample boolean; alter table p2 add column is_focal boolean;
         update p2 set is_sample = (first_public_year between 2016 and 2024 and n_refs >= 5 and title is not null);
         update p2 set is_focal = is_sample and first_public_year between 2017 and 2021""")
    q(f"copy p2 to '{O('papers.parquet')}' (format parquet, compression zstd)")
    for k, s in {"papers_v2": "select count(*) from p2", "sample_v2": "select count(*) from p2 where is_sample",
                 "focal_v2": "select count(*) from p2 where is_focal",
                 "focal_v2_date_precise": "select avg(date_precise::int) from p2 where is_focal"}.items():
        summ[k] = one(s)
    summ["focal_v2_date_precision"] = dict(con.execute("select date_precision, count(*) from p2 where is_focal group by 1").fetchall())
    summ["focal_v2_by_venue"] = dict(con.execute("select venue_class, count(*) from p2 where is_focal group by 1 order by 2 desc").fetchall())
    log(f"v2 论文 {summ['papers_v2']:,}，样本 {summ['sample_v2']:,}，焦点 {summ['focal_v2']:,}；渠道 {summ['focal_v2_by_venue']}")

    # ---------------- 4. 重映射其余表
    q(f"""copy (select v.member_id, m.new_id as paper_id, v.type, v.publication_date, v.arxiv_id
                from read_parquet('{S('versions.parquet')}') as v join mm as m on m.old_id = v.paper_id)
          to '{O('versions.parquet')}' (format parquet)""")
    for n, col in (("paper_year", "cites"), ("paper_year_in_corpus", "cites_in_corpus")):
        q(f"""copy (select m.new_id as paper_id, t.year, sum(t.{col}) as {col}
                    from read_parquet('{S(n + '.parquet')}') as t join mm as m on m.old_id = t.paper_id group by 1, 2)
              to '{O(n + '.parquet')}' (format parquet)""")
    q(f"""copy (select distinct coalesce(mc.new_id, c.citing_id) as citing_id, c.citing_year, c.citing_topic, c.citing_subfield,
                       md.new_id as cited_id, c.citing_is_ai_candidate, c.is_self_cite
                from read_parquet('{S('citations.parquet')}') as c join mm as md on md.old_id = c.cited_id
                left join mm as mc on mc.old_id = c.citing_id
                where c.citing_id not in (select paper_id from junk)
                  and coalesce(mc.new_id, c.citing_id) <> md.new_id)
          to '{O('citations.parquet')}' (format parquet)""")
    q(f"""copy (select * exclude (rn) from (
                 select m1.new_id as focal_id, m2.new_id as comp_id, t.* exclude (focal_id, comp_id),
                        row_number() over (partition by m1.new_id, m2.new_id order by t.sem_sim desc) as rn
                 from read_parquet('{S('pairs.parquet')}') as t join mm as m1 on m1.old_id = t.focal_id
                 join mm as m2 on m2.old_id = t.comp_id where m1.new_id <> m2.new_id) where rn = 1)
          to '{O('pairs.parquet')}' (format parquet)""")
    q(f"""copy (with c as (select pt_topic as topic_id, first_public_year as yr, count(*) as n from p2 where is_sample group by 1, 2)
                select t.* exclude (n_ai_dedup), t.n_ai_dedup as n_ai_dedup_v1,
                       case when t.n_ai_dedup is not null then coalesce(c.n, 0) end as n_ai_dedup
                from read_parquet('{S('topic_year.parquet')}') as t left join c on c.topic_id = t.topic_id and c.yr = t.year)
          to '{O('topic_year.parquet')}' (format parquet)""")
    em = pd.read_parquet(S("emb_meta.parquet"), columns=["paper_id"])
    em["row"] = np.arange(len(em), dtype=np.int64)
    con.register("em_df", em)
    q(f"""copy (select p.paper_id, e.row, p.first_public_year, p.first_public_date, p.date_precise, p.date_precision, p.pt_topic,
                       p.is_sample, p.is_focal, p.venue_class
                from p2 as p join em_df as e using (paper_id) where p.is_sample)
          to '{O('emb_meta.parquet')}' (format parquet)""")
    for n in ("_sim_calibration.json",):
        if not os.path.lexists(O(n)):
            os.symlink(S(n), O(n))

    # ---------------- 5. 自查
    chk = {
        "papers_unique": one(f"select count(*) = count(distinct paper_id) from read_parquet('{O('papers.parquet')}')"),
        "every_sample_has_embedding": one(f"""select count(*) from read_parquet('{O('papers.parquet')}') where is_sample
                                             and paper_id not in (select paper_id from read_parquet('{O('emb_meta.parquet')}'))""") == 0,
        "paper_year_total_preserved": one(f"""select (select sum(cites) from read_parquet('{O('paper_year.parquet')}')) =
              (select sum(t.cites) from read_parquet('{S('paper_year.parquet')}') as t where t.paper_id not in (select paper_id from junk))"""),
        "citations_no_self_loop": one(f"select count(*) from read_parquet('{O('citations.parquet')}') where citing_id = cited_id") == 0,
        "pairs_no_self": one(f"select count(*) from read_parquet('{O('pairs.parquet')}') where focal_id = comp_id") == 0,
        "no_junk_left": one(f"select count(*) from read_parquet('{O('papers.parquet')}') where paper_id in (select paper_id from junk)") == 0,
        "focal_is_subset_of_sample": one(f"select count(*) from read_parquet('{O('papers.parquet')}') where is_focal and not is_sample") == 0,
    }
    summ["checks"] = chk
    summ["seconds"] = round(time.time() - t0)
    json.dump(summ, open(O("_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    log(f"自查 {chk}")
    print("EXP0 V2 CLEAN DONE" if all(chk.values()) else "EXP0 V2 CLEAN DONE WITH CHECK FAILURES", flush=True)


if __name__ == "__main__":
    main()
