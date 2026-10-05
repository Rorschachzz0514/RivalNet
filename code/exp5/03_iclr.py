"""
【03】实验 5-4 · ICLR 审稿分作为独立质量锚点（预先登记见 实验5说明.md 第 2 节 5-4）

用途
  NAIDv2（ICLR 2021–2025 投稿的审稿分与录用结果，NAIP 作者公开）按规范化标题完全匹配到实验 2 的样本。
  5-4a  OLS log1p(y3) ~ rating + accept + 年份 + pred_nr + 竞争贡献（pred_full − pred_nr），HC1 稳健标准误
  5-4b  pred_full 与 rating 的 Spearman
  5-4c  log1p(y3) ~ log1p(前一年相似论文数) + 已被抢先 + 控制 | 审稿分五档 × 年份
  pred_full = MPCNet5_l1 五个种子预测的平均；pred_nr = B_no_rivals 五个种子预测的平均（验证 / 测试集）。

输入
  /path/to/mpcc/external/naidv2/*.csv；/path/to/mpcc/subsets/{exp0_v2/papers, pred_v2/samples}.parquet；
  /path/to/mpcc/exp2/runs/{MPCNet5_l1,B_no_rivals}_s*/preds.parquet

输出（results/）
  03_iclr_matched.parquet、03_iclr.md

用法
  python 03_iclr.py
"""
import glob
import os

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def ens(cfg):
    fs = sorted(glob.glob(f"/path/to/mpcc/exp2/runs/{cfg}_s*/preds.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=["paper_id", "split", "pred"]) for f in fs])
    return p.groupby("paper_id").pred.mean(), len(fs)


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("/path/to/mpcc/external/naidv2/*.csv"))])
    d["t"] = d.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    d = d.drop_duplicates("t")
    con = duckdb.connect()
    p = con.execute("""select paper_id, trim(regexp_replace(lower(title), '[^a-z0-9]+', ' ', 'g')) as t
                       from read_parquet('/path/to/mpcc/subsets/exp0_v2/papers.parquet') where is_focal""").df()
    s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet",
                        columns=["paper_id", "Y", "split", "y3", "c_m1_95", "n_preempted", "n_authors", "venue_at_T"])
    m = p.merge(d[["t", "pub_year", "score_mean", "accept"]], on="t").merge(s, on="paper_id")
    m = m[m.split.isin(["val", "test"])].drop_duplicates("paper_id")
    full, nf = ens("MPCNet5_l1")
    nr, nn = ens("B_no_rivals")
    m["pred_full"], m["pred_nr"] = m.paper_id.map(full), m.paper_id.map(nr)
    m = m.dropna(subset=["pred_full", "pred_nr"])
    m["comp"] = m.pred_full - m.pred_nr
    m["ly3"] = np.log1p(m.y3)
    m["lc"] = np.log1p(m.c_m1_95)
    m["pre"] = (m.n_preempted > 0).astype(int)
    m["rating"] = m.score_mean
    m["year"] = m.Y.astype(str)
    m["rbin"] = pd.qcut(m.rating.rank(method="first"), 5, labels=False).astype(str) + "_" + m.year
    m.to_parquet(os.path.join(RES, "03_iclr_matched.parquet"), index=False)

    base = smf.ols("ly3 ~ rating + accept + C(year)", data=m).fit(cov_type="HC1")
    a = smf.ols("ly3 ~ rating + accept + C(year) + pred_nr + comp", data=m).fit(cov_type="HC1")
    a_full = smf.ols("ly3 ~ rating + accept + C(year) + pred_full", data=m).fit(cov_type="HC1")
    c = smf.ols("ly3 ~ lc + pre + np.log1p(n_authors) + C(rbin)", data=m).fit(cov_type="HC1")
    rho = spearmanr(m.pred_full, m.rating).statistic
    rho_y = spearmanr(m.rating, m.y3).statistic
    passed = bool(a.params["comp"] > 0 and a.pvalues["comp"] < 0.05)
    tab = lambda r, keys: pd.DataFrame({"coef": r.params[keys], "se": r.bse[keys], "p": r.pvalues[keys]}).round(4).to_markdown()
    md = ["# 实验 5-4 ICLR 审稿分（质量锚点）\n",
          f"匹配样本 {len(m):,} 篇（2020 年 {int((m.Y == 2020).sum())}，2021 年 {int((m.Y == 2021).sum())}；录用 {m.accept.mean():.1%}）。"
          f"pred_full = MPCNet5_l1（{nf} 个种子平均），pred_nr = B_no_rivals（{nn} 个种子平均）。\n",
          f"审稿分与 3 年被引的 Spearman = {rho_y:.3f}；**5-4b**：MPC-Net 预测与审稿分的 Spearman = {rho:.3f}。\n",
          "## 5-4a 控制审稿分后，竞争信息是否仍能解释被引\n",
          f"只用审稿分 + 录用 + 年份：R² = {base.rsquared:.3f}；加 pred_full：R² = {a_full.rsquared:.3f}；"
          f"加 pred_nr + 竞争贡献：R² = {a.rsquared:.3f}。\n",
          tab(a, ["rating", "accept", "pred_nr", "comp"]), "\n",
          f"**判定**：竞争贡献系数 {a.params['comp']:+.3f}（p = {a.pvalues['comp']:.2g}）；**{'通过' if passed else '未通过'}**。\n",
          "## 5-4c 审稿分相近（五档 × 年份）时的竞争变量\n", tab(c, ["lc", "pre"]), "\n",
          f"（已被抢先的论文 {int(m.pre.sum())} 篇）\n"]
    open(os.path.join(RES, "03_iclr.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
