"""
【15】实验 0 v2 · 检验 0-5 v2（撞车论文对）（预先登记见 实验0说明.md 7.4）

用途
  撞车论文对 = 两篇都在主分析集、日期都精确（到日或到月）、SPECTER2 相似度 ≥ 0.97、首次公开相差 1–183 天
  （任一方只精确到月时要求相差 ≥ 31 天，避免同月内先后不明）、互不引用（is_strict）、没有共同作者。
  v2 数据已剔除无效条目并补合并了同一篇论文的不同版本。先后由 v2 的首次公开日期重新计算。
  主设定：条件 Poisson（论文对固定效应，标准误按论文对聚类）+ 论文层面控制变量（log 作者数、log 机构数、
          log 参考文献数、开放获取、有摘要、有基金、有预印本、发表渠道类别）。
          y3 ~ β·后发表 + 控制变量 | 论文对。通过标准：β < 0 且 p < 0.05。
  稳健（不计入判定）：不加控制变量；只用同一年份的对；只用两方都精确到日的对；按时间差 1–30 / 31–90 / 91–183 天分组；
                     相似度 ≥ 0.98。

输入
  DATA：pairs.parquet、papers.parquet（作者）；results_v2/focal.parquet

输出（results_v2/）
  15_twins_pairs.parquet、15_twins.csv、15_twins.md

用法
  python 15_v2_test_twins.py

运行记录
  （见 实验0说明.md 第 6 节）
"""
import os
import sys
import time
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

warnings.filterwarnings("ignore")
CTRL = ["l_auth", "l_inst", "l_refs", "oa", "abs_", "fund", "pre"]


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET threads={C.THREADS}; SET memory_limit='100GB'; SET temp_directory='{C.TMP_DIR}'")
    D = lambda n: os.path.join(C.V2_DATA, n)
    F = os.path.join(C.V2_RES, "focal.parquet")
    con.execute(f"""create table f as select f.*, p.first_public_date::date as dte, p.author_ids
                    from read_parquet('{F}') as f join read_parquet('{D('papers.parquet')}') as p using (paper_id)
                    where f.in_main and f.date_precision <> 'year'""")
    df = con.execute(f"""
        with tw as (
          select least(t.focal_id, t.comp_id) as a, greatest(t.focal_id, t.comp_id) as b, max(t.sem_sim) as sim,
                 bool_and(t.is_strict) as strict
          from read_parquet('{D('pairs.parquet')}') as t
          where t.sem_sim >= {C.TWIN_SIM} and t.focal_id in (select paper_id from f) and t.comp_id in (select paper_id from f)
          group by 1, 2)
        select tw.a, tw.b, tw.sim, x.dte as da, y.dte as db, x.date_precision as pa, y.date_precision as pb, x.Y as ya, y.Y as yb,
               len(list_intersect(coalesce(x.author_ids, []), coalesce(y.author_ids, []))) as n_shared
        from tw join f as x on x.paper_id = tw.a join f as y on y.paper_id = tw.b
        where tw.strict""").df()
    df["gap"] = (df.db - df.da).dt.days
    df["early_id"] = np.where(df.gap > 0, df.a, df.b)
    df["late_id"] = np.where(df.gap > 0, df.b, df.a)
    df["gap"] = df.gap.abs()
    both_day = (df.pa == "day") & (df.pb == "day")
    keep = (df.n_shared == 0) & (df.gap >= 1) & (df.gap <= C.TWIN_DAYS) & (both_day | (df.gap >= 31))
    tw = df[keep].reset_index(drop=True)
    tw["both_day"] = both_day[keep].to_numpy()
    tw["same_year"] = (tw.ya == tw.yb).to_numpy()
    tw.to_parquet(os.path.join(C.V2_RES, "15_twins_pairs.parquet"), index=False)
    print(f"候选 {len(df):,} 对 → 撞车论文对 {len(tw):,}（{time.time() - t0:.0f}s）", flush=True)

    focal = pd.read_parquet(F, columns=["paper_id", "y3", "n_authors", "n_institutions", "n_refs", "any_oa", "has_abstract",
                                         "has_funding", "has_preprint", "venue_class"])

    def stacked(sub):
        sub = sub.reset_index(drop=True)
        e = pd.DataFrame({"pair": np.arange(len(sub)), "paper_id": sub.early_id, "late": 0})
        l = pd.DataFrame({"pair": np.arange(len(sub)), "paper_id": sub.late_id, "late": 1})
        s = pd.concat([e, l], ignore_index=True).merge(focal, on="paper_id", how="left")
        return s.assign(l_auth=np.log1p(s.n_authors), l_inst=np.log1p(s.n_institutions.fillna(0)), l_refs=np.log(s.n_refs),
                        oa=s.any_oa.astype(float), abs_=s.has_abstract.astype(float), fund=s.has_funding.astype(float),
                        pre=s.has_preprint.astype(float), venue=s.venue_class.astype("category"))

    def cond(sub, name, controls=True, group="稳健"):
        s = stacked(sub)
        if s.pair.nunique() < 30:
            return {"group": group, "spec": name, "n_pairs": s.pair.nunique()}
        rhs = " + ".join(["late"] + (CTRL + ["C(venue)"] if controls else []))
        m = pf.fepois(f"y3 ~ {rhs} | pair", data=s, vcov={"CRV1": "pair"})
        t = m.tidy().loc["late"]
        d = np.log1p(sub.merge(focal[["paper_id", "y3"]], left_on="late_id", right_on="paper_id").y3.to_numpy()) - \
            np.log1p(sub.merge(focal[["paper_id", "y3"]], left_on="early_id", right_on="paper_id").y3.to_numpy())
        return {"group": group, "spec": name, "n_pairs": len(sub), "n_pairs_used": m._N // 2, "beta": t["Estimate"],
                "se": t["Std. Error"], "p": t["Pr(>|t|)"], "pct_diff": (np.exp(t["Estimate"]) - 1) * 100,
                "share_late_less": float((d < 0).mean()), "share_late_more": float((d > 0).mean())}

    rows = [cond(tw, "主设定（加控制变量）", group="0-5"),
            cond(tw, "不加控制变量", controls=False),
            cond(tw[tw.same_year], "只用同一年份"),
            cond(tw[tw.both_day], "两方都精确到日"),
            cond(tw[tw.sim >= 0.98], "相似度 ≥ 0.98"),
            cond(tw[tw.gap <= 30], "时间差 1–30 天"),
            cond(tw[(tw.gap > 30) & (tw.gap <= 90)], "时间差 31–90 天"),
            cond(tw[tw.gap > 90], "时间差 91–183 天")]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.V2_RES, "15_twins.csv"), index=False)
    r = res.iloc[0]
    passed = bool(r.get("beta", 0) < 0 and r.get("p", 1) < 0.05)
    md = ["# 检验 0-5 v2（撞车论文对）\n",
          f"条件：两篇都在主分析集、日期精确、相似度 ≥ {C.TWIN_SIM}、相差 1–{C.TWIN_DAYS} 天（有一方只精确到月时 ≥ 31 天）、"
          f"互不引用、没有共同作者。共 {len(tw):,} 对。`pct_diff` = 后发表者被引的百分比差异。\n",
          res.round(4).to_markdown(index=False), "\n",
          f"**0-5 v2**：后发表者被引差异 {r.get('pct_diff', np.nan):+.1f}%（β = {r.get('beta', np.nan):+.4f}，p = {r.get('p', np.nan):.2g}）；"
          f"**{'通过' if passed else '未通过'}**。\n"]
    open(os.path.join(C.V2_RES, "15_twins.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"15 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
