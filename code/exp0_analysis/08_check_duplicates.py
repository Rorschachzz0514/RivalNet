"""
【08】自动识别未合并的"同一篇论文的不同版本"，估计其比例，并在剔除后重跑 0-5（撞车）与 0-2（微观）

用途
  数据构建时的版本合并（规则 R-d）只合并 DOI 相同、arXiv 编号相同、或"标题完全相同 + 共同作者 + 一方是预印本"的记录，
  漏掉了：改过标题的版本（arXiv 版改题后投会议）、会议版与期刊扩展版（两边都不是预印本）等。
  本脚本代替人工核验，把论文对分为四类（按顺序判定）：
    无效条目     任一方标题过短（规范化后 < 10 字符）或是书 / 期刊的附属部分（references、bibliography、index、preface、
                 contents、front matter、editorial 等），不是真正的论文
    疑似同一篇   作者重合度（Jaccard）≥ 0.5 且 标题相似度（Jaro-Winkler，小写去标点）≥ 0.9，或 规范化标题完全相同
    同一团队     有共同作者，但不满足上一条（同一组作者的相关但不同的论文）
    不同团队     没有共同作者（真正意义上的"撞车"，与 Hill & Stein 的定义最接近）
  A. 撞车论文对（【05】的 22,221 对）：各类比例（总体、按相似度、按时间差），每类示例；
     分类后重跑条件 Poisson（论文对固定效应）。
  B. 焦点论文的"发表前对手"（pairs 中相似度 ≥ 0.95、发表前 365 天内）：疑似同一篇的比例；
     从 prior_s2_95 中减去疑似同一篇的个数后重跑 0-2 主设定与诊断 D1。
     注意：pairs 只含 KNN 近邻与文献耦合对手，相似度 ≥ 0.95 的近邻基本都在其中，作为估计足够。

输入
  results/05_twins_pairs.parquet（【05】）、results/focal.parquet（【01】）
  DATA_DIR/papers.parquet（title、author_ids、type、arxiv_id）、DATA_DIR/pairs.parquet、DATA_DIR/sim_counts.parquet

输出（results/）
  08_twins_classified.parquet   撞车论文对 + 分类与相似度指标
  08_dup_share.csv              各类比例（撞车对按相似度 / 时间差分组；发表前对手）
  08_twins_by_class.csv         分类后的 0-5 条件 Poisson
  08_micro_dedup.csv            剔除疑似同一篇后的 0-2 与 D1
  08_duplicates.md              结果表、示例与解读

用法
  python 08_check_duplicates.py           全量
  python 08_check_duplicates.py --test    只用 2019 年（冒烟测试）

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
import os
import sys
import time
import warnings
from importlib import import_module

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

M3 = import_module("03_test_micro")
M5 = import_module("05_test_twins")
warnings.filterwarnings("ignore")
AUTH_J, TITLE_S = 0.5, 0.9

# 两篇论文 a、b 的相似指标与分类（DuckDB 表达式，a_/b_ 前缀为两篇的列）
FEATURES = f"""
    jaro_winkler_similarity(a_t, b_t) as title_sim,
    (a_t = b_t) as same_title,
    len(list_intersect(a_au, b_au)) as n_shared_auth,
    len(list_intersect(a_au, b_au)) / greatest(len(list_distinct(list_concat(a_au, b_au))), 1) as auth_jacc,
    (a_au[1] = b_au[1]) as same_first_author
"""
JUNK = ("^(references?|bibliography|index|author index|subject index|preface|foreword|contents|table of contents|"
        "front matter|back matter|frontmatter|backmatter|editorial|introduction|conclusions?|appendix.*|glossary|"
        "abbreviations|acknowledg.*|title page|copyright|list of (figures|tables)|notes|about the authors?|"
        "editorial board|cover|erratum|corrigendum)$")
CLASSIFY = f"""case when length(a_t) < 10 or length(b_t) < 10 or regexp_matches(a_t, '{JUNK}') or regexp_matches(b_t, '{JUNK}') then '无效条目'
                    when (auth_jacc >= {AUTH_J} and title_sim >= {TITLE_S}) or (same_title and n_shared_auth > 0) then '疑似同一篇'
                    when n_shared_auth > 0 then '同一团队'
                    else '不同团队' end"""
NORM = r"trim(regexp_replace(lower(coalesce(title, '')), '[^\p{L}\p{N}]+', ' ', 'g'))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sfx = "_test" if args.test else ""
    con = duckdb.connect()
    con.execute(f"SET threads={C.THREADS}; SET memory_limit='200GB'; SET temp_directory='{C.TMP_DIR}'")
    D = lambda n: os.path.join(C.DATA_DIR, n)
    con.execute(f"""create table pp as select paper_id, {NORM} as t, coalesce(author_ids, []) as au, type, arxiv_id, source_name
                    from read_parquet('{D("papers.parquet")}')""")
    yw = "where Y = 2019" if args.test else ""
    con.execute(f"create table f as select paper_id from read_parquet('{os.path.join(C.RES_DIR, 'focal.parquet')}') {yw}")

    # ---------- A. 撞车论文对 ----------
    con.execute(f"""create table tw as select * from read_parquet('{os.path.join(C.RES_DIR, '05_twins_pairs.parquet')}')
                    where early_id in (select paper_id from f)""")
    con.execute(f"""create table twc as
        select *, {CLASSIFY} as cls from (
          select tw.*, a.t as a_t, b.t as b_t, a.au as a_au, b.au as b_au,
                 a.type as type_early, b.type as type_late, a.source_name as source_early, b.source_name as source_late,
                 {FEATURES}
          from tw join pp as a on a.paper_id = tw.early_id join pp as b on b.paper_id = tw.late_id)""")
    tw = con.execute("select * exclude (a_au, b_au) from twc").df()
    tw.to_parquet(os.path.join(C.RES_DIR, f"08_twins_classified{sfx}.parquet"), index=False)
    print(f"A: 撞车对 {len(tw):,} ({time.time() - t0:.0f}s)", flush=True)

    def shares(df, by, label):
        g = df.groupby(by, observed=True).cls.value_counts(normalize=True).unstack(fill_value=0)
        g["n"] = df.groupby(by, observed=True).size()
        g = g.reset_index().rename(columns={by: "group"})
        g.insert(0, "table", label)
        g["group"] = g["group"].astype(str)
        return g

    tw["all"] = "全部"
    tw["sim_bin"] = pd.cut(tw.sim, [0.97, 0.98, 0.99, 1.0001], right=False, labels=["0.97–0.98", "0.98–0.99", "≥0.99"])
    tw["gap_bin"] = pd.cut(tw.gap, [0, 30, 90, 183], labels=["1–30 天", "31–90 天", "91–183 天"])
    sh = pd.concat([shares(tw, "all", "撞车对"), shares(tw, "sim_bin", "撞车对·相似度"), shares(tw, "gap_bin", "撞车对·时间差")])

    focal = pd.read_parquet(os.path.join(C.RES_DIR, "focal.parquet"),
                            columns=["paper_id", "y3", "n_authors", "n_institutions", "n_refs", "any_oa",
                                     "has_abstract", "has_funding", "has_preprint"])
    rows = []
    for name, sub in [("全部（原 0-5 主设定）", tw), ("剔除疑似同一篇与无效条目", tw[~tw.cls.isin(["疑似同一篇", "无效条目"])]),
                      ("只用不同团队（主）", tw[tw.cls == "不同团队"]), ("只用同一团队", tw[tw.cls == "同一团队"]),
                      ("只用疑似同一篇", tw[tw.cls == "疑似同一篇"]), ("只用无效条目", tw[tw.cls == "无效条目"])]:
        s = M5.stacked(sub.reset_index(drop=True), focal)
        r = M5.cond_pois(s, name)
        r.update({k: v for k, v in M5.describe(sub, name).items() if k in ("share_late_less", "share_late_more")})
        rows.append(r)
        if name == "只用不同团队（主）":
            rows.append(M5.cond_pois(s, "只用不同团队 + 控制变量", controls=True))
            for gb in ["1–30 天", "31–90 天", "91–183 天"]:
                ss = sub[sub.gap_bin == gb].reset_index(drop=True)
                rows.append(M5.cond_pois(M5.stacked(ss, focal), f"不同团队·时间差 {gb}"))
    tc = pd.DataFrame(rows)
    tc.to_csv(os.path.join(C.RES_DIR, f"08_twins_by_class{sfx}.csv"), index=False)

    ex = []
    rng = np.random.default_rng(7)
    for c in ["无效条目", "疑似同一篇", "同一团队", "不同团队"]:
        sub = tw[tw.cls == c]
        for _, r in sub.iloc[rng.choice(len(sub), size=min(5, len(sub)), replace=False)].iterrows():
            ex.append({"类别": c, "相似度": round(r.sim, 3), "天数": r.gap, "标题相似": round(r.title_sim, 2),
                       "作者重合": round(r.auth_jacc, 2), "先": f"{r.a_t[:70]} [{r.type_early}; {r.source_early}]",
                       "后": f"{r.b_t[:70]} [{r.type_late}; {r.source_late}]"})
    ex = pd.DataFrame(ex)

    # ---------- B. 焦点论文的发表前对手 ----------
    k = int(round(C.TAU * 100))
    con.execute(f"""create table pr as
        select * , {CLASSIFY} as cls from (
          select p.focal_id, p.comp_id, p.sem_sim, a.t as a_t, b.t as b_t, a.au as a_au, b.au as b_au, {FEATURES}
          from read_parquet('{D("pairs.parquet")}') as p
          join pp as a on a.paper_id = p.focal_id join pp as b on b.paper_id = p.comp_id
          where p.sem_sim >= {C.TAU} and p.days_after_focal between -{C.PRIOR_DAYS} and -1
            and p.focal_id in (select paper_id from f))""")
    prs = con.execute("select cls, count(*) as n from pr group by 1").df()
    prs["share"] = prs.n / prs.n.sum()
    n_pairs_pr = int(prs.n.sum())
    dup_by_focal = con.execute("""select focal_id as paper_id, count(*) filter (where cls in ('疑似同一篇', '无效条目')) as n_dup,
                                         count(*) filter (where cls <> '不同团队') as n_team
                                  from pr group by 1""").df()
    df = M3.prepare(args.test)
    tot_prior = int(df[f"prior_s2_{k}"].sum())
    df = df.merge(dup_by_focal, on="paper_id", how="left").fillna({"n_dup": 0, "n_team": 0})
    df["x_dedup"] = np.log1p((df[f"prior_s2_{k}"] - df.n_dup).clip(lower=0))
    df["x_other"] = np.log1p((df[f"prior_s2_{k}"] - df.n_team).clip(lower=0))
    share_focal_dup = (df.n_dup > 0).mean()
    sh = pd.concat([sh, pd.DataFrame([{"table": "发表前对手（S2≥0.95）", "group": "全部", "n": n_pairs_pr,
                                       **dict(zip(prs.cls, prs.share))}])])
    sh.to_csv(os.path.join(C.RES_DIR, f"08_dup_share{sfx}.csv"), index=False)

    mr = []
    for name, x, extra, model, y in [
        ("原主设定", f"x_s2_{k}", [], "pois", "y3"),
        ("剔除疑似同一篇与无效条目", "x_dedup", [], "pois", "y3"),
        ("剔除疑似同一篇与无效条目（OLS）", "x_dedup", [], "ols", "ly3"),
        ("只算不同团队的对手", "x_other", [], "pois", "y3"),
        ("剔除疑似同一篇与无效条目 + D1", "x_dedup", ["l_after"], "pois", "y3"),
        ("只算不同团队的对手 + D1", "x_other", ["l_after"], "pois", "y3"),
    ]:
        fml = f"{y} ~ {' + '.join([x] + M3.CONTROLS + extra)} | ty"
        fit = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": "topic"})
        t = fit.tidy().loc[x]
        mr.append({"spec": name, "model": "Poisson" if model == "pois" else "OLS", "beta": t["Estimate"],
                   "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": fit._N})
    mr = pd.DataFrame(mr)
    mr.to_csv(os.path.join(C.RES_DIR, f"08_micro_dedup{sfx}.csv"), index=False)

    pct = lambda v: f"{v:.1%}"
    md = ["# 未合并的同一篇论文：比例与影响\n",
          f"分类规则：**疑似同一篇** = 作者重合度 ≥ {AUTH_J} 且标题相似度 ≥ {TITLE_S}（或规范化标题完全相同且有共同作者）；"
          "**同一团队** = 有共同作者但标题不同；**不同团队** = 没有共同作者。\n",
          "## A. 撞车论文对的构成\n", sh[sh.table != "发表前对手（S2≥0.95）"].round(3).to_markdown(index=False), "\n",
          "示例（随机）：\n", ex.to_markdown(index=False), "\n",
          "## A. 分类后重跑 0-5（条件 Poisson，论文对固定效应）\n", "`pct_diff` = 后发表者被引的百分比差异。\n",
          tc.round(4).to_markdown(index=False), "\n",
          "## B. 焦点论文的发表前对手（相似度 ≥ 0.95）\n",
          f"pairs 中发表前 365 天内、相似度 ≥ 0.95 的对手 {n_pairs_pr:,} 对（sim_counts 全量计数合计 {tot_prior:,}）：",
          prs.round(4).to_markdown(index=False), "\n",
          f"有至少一个疑似同一篇\"对手\"的焦点论文占 {pct(share_focal_dup)}。\n",
          "剔除后重跑 0-2：\n", mr.round(4).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, f"08_duplicates{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"08 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
