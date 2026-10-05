"""
【05】检验 0-5（撞车）：几乎同时做出同一成果的两篇论文，后发表者是否被引更少

用途
  撞车论文对 = 两篇都是焦点论文、SPECTER2 相似度 ≥ TWIN_SIM（0.97）、首次公开相差 1–TWIN_DAYS（183）天、
  互不引用（is_strict）。无序对去重；相差 0 天的无法分先后，剔除。
  同一对内比较后发表者与先发表者的 y3（各自发表后第 1–3 个完整年份）：
    - 描述：log1p(y3_后) − log1p(y3_先) 的均值，后发表者更少 / 更多的比例，Wilcoxon 符号秩检验
    - 条件 Poisson（主）：两篇各一行，y3 ~ β·后发表 | 论文对固定效应，标准误按论文对聚类；
      exp(β) − 1 = 后发表者被引的百分比差异（参照 Hill & Stein 2025：−21%）
    - 稳健：加论文层面控制变量；只用同一年份的对；剔除相似度 ≥ 0.99（可能是未合并的重复记录）；按时间差分组
  通过标准：主设定 β < 0 且 p < 0.05。**需人工核验**：输出候选清单，人工确认是否真为"同一成果"。

输入
  DATA_DIR/pairs.parquet（T5）、DATA_DIR/papers.parquet（标题、DOI、日期）、results/focal.parquet（【01】）

输出（results/）
  05_twins_pairs.parquet          全部撞车论文对（先 / 后的 id、相似度、天数、y3 等）
  05_twins.csv                    各设定的 β、百分比差异、p 值、对数
  05_twins_for_manual_check.csv   人工核验清单：相似度 ≥ 0.99 的全部对 + 随机抽 200 对（固定种子），带标题、DOI、日期、y3
  05_twins.md                     结果表与解读

用法
  python 05_test_twins.py           全量
  python 05_test_twins.py --test    只用 2019 年的焦点论文（冒烟测试）

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
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


def build_pairs(test):
    con = duckdb.connect()
    con.execute(f"SET threads={C.THREADS}; SET memory_limit='100GB'; SET temp_directory='{C.TMP_DIR}'")
    F = os.path.join(C.RES_DIR, "focal.parquet")
    ywhere = "where Y = 2019" if test else ""
    con.execute(f"create table f as select * from read_parquet('{F}') {ywhere}")
    con.execute(f"""create table tw as
        select least(p.focal_id, p.comp_id) as a, greatest(p.focal_id, p.comp_id) as b,
               max(p.sem_sim) as sim, max(abs(p.days_after_focal)) as gap,
               -- 先发表者：days_after_focal > 0 表示 comp 晚于 focal
               any_value(case when p.days_after_focal > 0 then p.focal_id else p.comp_id end) as early_id,
               any_value(case when p.days_after_focal > 0 then p.comp_id else p.focal_id end) as late_id
        from read_parquet('{os.path.join(C.DATA_DIR, "pairs.parquet")}') as p
        where p.sem_sim >= {C.TWIN_SIM} and abs(p.days_after_focal) between 1 and {C.TWIN_DAYS} and p.is_strict
          and p.focal_id in (select paper_id from f) and p.comp_id in (select paper_id from f)
        group by 1, 2""")
    df = con.execute("""
        select tw.*, e.y3 as y3_early, l.y3 as y3_late, e.Y as Y_early, l.Y as Y_late,
               e.topic as topic_early, l.topic as topic_late
        from tw join f as e on e.paper_id = tw.early_id join f as l on l.paper_id = tw.late_id""").df()
    meta = con.execute(f"""select paper_id, title, doi, first_public_date, source_name
        from read_parquet('{os.path.join(C.DATA_DIR, "papers.parquet")}')
        where paper_id in (select early_id from tw union select late_id from tw)""").df()
    return con, df, meta


def stacked(df, focal):
    """两篇各一行；附上控制变量。"""
    e = pd.DataFrame({"pair": np.arange(len(df)), "paper_id": df.early_id, "late": 0})
    l = pd.DataFrame({"pair": np.arange(len(df)), "paper_id": df.late_id, "late": 1})
    s = pd.concat([e, l], ignore_index=True).merge(focal, on="paper_id", how="left")
    s = s.assign(l_auth=np.log1p(s.n_authors), l_inst=np.log1p(s.n_institutions.fillna(0)), l_refs=np.log(s.n_refs),
                 oa=s.any_oa.astype(float), abs_=s.has_abstract.astype(float), fund=s.has_funding.astype(float),
                 pre=s.has_preprint.astype(float))
    return s


def cond_pois(s, name, controls=False):
    rhs = " + ".join(["late"] + (CTRL if controls else []))
    n_pairs = s.pair.nunique()
    if n_pairs < 10:
        return {"spec": name, "n_pairs": n_pairs}
    fit = pf.fepois(f"y3 ~ {rhs} | pair", data=s, vcov={"CRV1": "pair"})
    t = fit.tidy().loc["late"]
    return {"spec": name, "n_pairs": n_pairs, "n_pairs_used": fit._N // 2, "beta": t["Estimate"], "se": t["Std. Error"],
            "p": t["Pr(>|t|)"], "pct_diff": (np.exp(t["Estimate"]) - 1) * 100}


def describe(df, name):
    d = np.log1p(df.y3_late) - np.log1p(df.y3_early)
    w = stats.wilcoxon(df.y3_late, df.y3_early, zero_method="wilcox") if len(df) > 10 else None
    return {"spec": name, "n_pairs": len(df), "mean_log_diff": d.mean(), "share_late_less": (df.y3_late < df.y3_early).mean(),
            "share_late_more": (df.y3_late > df.y3_early).mean(), "median_y3_early": df.y3_early.median(),
            "median_y3_late": df.y3_late.median(), "wilcoxon_p": w.pvalue if w else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sfx = "_test" if args.test else ""
    con, df, meta = build_pairs(args.test)
    assert (df.early_id != df.late_id).all()
    df.to_parquet(os.path.join(C.RES_DIR, f"05_twins_pairs{sfx}.parquet"), index=False)
    print(f"撞车论文对 {len(df):,}（{time.time() - t0:.0f}s）", flush=True)

    focal = pd.read_parquet(os.path.join(C.RES_DIR, "focal.parquet"),
                            columns=["paper_id", "y3", "n_authors", "n_institutions", "n_refs", "any_oa",
                                     "has_abstract", "has_funding", "has_preprint"])
    subsets = {
        "全部（主）": df,
        "同一年份": df[df.Y_early == df.Y_late],
        "剔除相似度≥0.99": df[df.sim < 0.99],
        "时间差 1–30 天": df[df.gap <= 30],
        "时间差 31–90 天": df[(df.gap > 30) & (df.gap <= 90)],
        "时间差 91–183 天": df[df.gap > 90],
    }
    rows_p, rows_d = [], []
    for name, sub in subsets.items():
        s = stacked(sub.reset_index(drop=True), focal)
        rows_p.append(cond_pois(s, name))
        if name == "全部（主）":
            rows_p.append(cond_pois(s, "全部 + 控制变量", controls=True))
        rows_d.append(describe(sub, name))
    rp, rd = pd.DataFrame(rows_p), pd.DataFrame(rows_d)
    rp.merge(rd, on=["spec", "n_pairs"], how="left").to_csv(os.path.join(C.RES_DIR, f"05_twins{sfx}.csv"), index=False)

    # 人工核验清单
    rng = np.random.default_rng(2026)
    hi = df[df.sim >= 0.99]
    rest = df[df.sim < 0.99]
    samp = rest.iloc[rng.choice(len(rest), size=min(200, len(rest)), replace=False)] if len(rest) else rest
    chk = pd.concat([hi.assign(reason="相似度≥0.99"), samp.assign(reason="随机抽样")], ignore_index=True)
    m = meta.set_index("paper_id")
    for side in ("early", "late"):
        ids = chk[f"{side}_id"]
        for c in ("title", "doi", "first_public_date", "source_name"):
            chk[f"{c}_{side}"] = ids.map(m[c]).to_numpy()
    chk["人工判断（同一成果/不同/重复记录）"] = ""
    cols = ["reason", "sim", "gap", "early_id", "late_id", "title_early", "title_late", "first_public_date_early",
            "first_public_date_late", "doi_early", "doi_late", "source_name_early", "source_name_late",
            "y3_early", "y3_late", "人工判断（同一成果/不同/重复记录）"]
    chk.sort_values("sim", ascending=False)[cols].to_csv(
        os.path.join(C.RES_DIR, f"05_twins_for_manual_check{sfx}.csv"), index=False, encoding="utf-8-sig")

    main_r = rp.iloc[0]
    passed = bool(main_r.get("beta", 0) < 0 and main_r.get("p", 1) < 0.05)
    fmt = lambda t: t.round(4).to_markdown(index=False)
    md = ["# 检验 0-5（撞车论文对）\n",
          f"定义：两篇都是焦点论文、相似度 ≥ {C.TWIN_SIM}、首次公开相差 1–{C.TWIN_DAYS} 天、互不引用。共 {len(df):,} 对"
          f"（其中相似度 ≥ 0.99：{len(hi):,} 对）。\n",
          "## 条件 Poisson（论文对固定效应）\n", "`pct_diff` = 后发表者被引的百分比差异。\n", fmt(rp), "\n",
          "## 描述\n", fmt(rd), "\n",
          f"**0-5 判定**：后发表者被引差异 {main_r.get('pct_diff', np.nan):.1f}%（β = {main_r.get('beta', np.nan):.4f}，"
          f"p = {main_r.get('p', np.nan):.2g}）；**{'通过' if passed else '未通过'}**（标准：β < 0 且显著；参照 Hill & Stein 2025：−21%）。"
          f"\n\n人工核验清单：`05_twins_for_manual_check.csv`（{len(chk)} 对），结论需人工确认后才算最终结果。\n"]
    open(os.path.join(C.RES_DIR, f"05_twins{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"05 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
