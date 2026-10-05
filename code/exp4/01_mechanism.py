"""
【01】实验 4 · 机制：竞争贡献 c_i = pred(MPCNet5_l1) − pred(B_no_rivals) 的分布、驱动因素、互补 / 替代、不平等校准、个案
（预先登记见 实验4说明.md 第 2 节）

用途
  4-1  测试集 c_i 的分布，按真实被引五档
  4-2  c_i ~ 竞争集合描述量（标准化）| 子课题固定效应（pyfixest，按子课题聚类）
  4-3  每篇测试论文与其 50 个对手"之后被同一篇论文共同引用"的比例（施引论文来自 OpenAlex 全库，用到未来信息，只用于解释）；
       c_i 对共被引比例的回归（含 4-2 的控制）
  4-5  子课题内基尼系数：expm1(预测) vs 真实 y3（MPC-Net、MLP-文本、LightGBM L4、NAIP）
  4-7  c_i 最负 / 最正（y3 > 0）的各 3 篇论文：标题、最相似的 3 个对手、y3、两种预测

输入
  /path/to/mpcc/exp2/runs/{MPCNet5_l1,B_no_rivals}_s*/preds.parquet；/path/to/mpcc/exp1/results/preds_*.parquet；
  /path/to/mpcc/subsets/pred_v2/{samples,rivals}.parquet；/path/to/mpcc/subsets/exp0_v2/{citations,papers}.parquet

输出（results/）
  01_contrib.parquet、01_drivers.csv、01_cocite.csv、01_gini.csv、01_cases.md、实验4结果.md

用法
  python 01_mechanism.py
"""
import glob
import os
import time
import warnings

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SP = "/path/to/mpcc/subsets/pred_v2/"
V2 = "/path/to/mpcc/subsets/exp0_v2/"


def ens(cfg):
    fs = sorted(glob.glob(f"/path/to/mpcc/exp2/runs/{cfg}_s*/preds.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=["paper_id", "split", "pred"]) for f in fs])
    return p[p.split == "test"].groupby("paper_id").pred.mean()


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else np.nan


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    s = pd.read_parquet(SP + "samples.parquet", columns=["paper_id", "split", "Y", "y3", "eval_c2000", "n_preempted", "c_m1_95", "topic"])
    s = s[s.split == "test"].copy()
    s["pred_full"], s["pred_nr"] = s.paper_id.map(ens("MPCNet5_l1")), s.paper_id.map(ens("B_no_rivals"))
    s["c"] = s.pred_full - s.pred_nr
    con = duckdb.connect()
    con.execute("SET threads=96; SET memory_limit='200GB'; SET temp_directory='/path/to/mpcc/tmp'")
    con.register("te", s[["paper_id"]])
    r = con.execute(f"""select r.* from read_parquet('{SP}rivals.parquet') as r join te on te.paper_id = r.focal_id""").df()
    g = r.groupby("focal_id")
    desc = pd.DataFrame({
        "sim_top10": r[r["rank"] <= 10].groupby("focal_id").sim.mean(),
        "n_prior": g.dyear.apply(lambda x: (x < 0).sum()),
        "rival_strength_mean": g.rival_cum.apply(lambda x: np.log1p(x).mean()),
        "rival_strength_max": g.rival_cum.apply(lambda x: np.log1p(x).max()),
        "share_coupled": g.coupled.mean(), "share_shared_auth": g.shared_auth.mean(),
        "share_focal_cites": g.focal_cites.mean(), "share_rival_cites": g.rival_cites.mean(),
    })
    s = s.merge(desc, left_on="paper_id", right_index=True, how="left")
    s["preempted"] = (s.n_preempted > 0).astype(float)
    s["log_density"] = np.log1p(s.c_m1_95)

    # 4-3 共被引：对手与焦点论文被同一篇施引论文共同引用
    con.register("rv", r[["focal_id", "comp_id"]])
    co = con.execute(f"""
        with cit as (select distinct citing_id, cited_id from read_parquet('{V2}citations.parquet')),
             fc as (select c.citing_id, c.cited_id as focal_id from cit as c join te on te.paper_id = c.cited_id)
        select rv.focal_id, rv.comp_id, count(c2.citing_id) as n_cocite
        from rv join fc on fc.focal_id = rv.focal_id join cit as c2 on c2.citing_id = fc.citing_id and c2.cited_id = rv.comp_id
        group by 1, 2""").df()
    cs = co.groupby("focal_id").agg(n_cocited_rivals=("comp_id", "nunique"), cocite_total=("n_cocite", "sum"))
    s = s.merge(cs, left_on="paper_id", right_index=True, how="left").fillna({"n_cocited_rivals": 0, "cocite_total": 0})
    s["share_cocited"] = s.n_cocited_rivals / 50
    s.to_parquet(os.path.join(RES, "01_contrib.parquet"), index=False)
    print(f"描述量与共被引完成（{time.time() - t0:.0f}s）", flush=True)

    # 4-1
    s["y_bin"] = pd.cut(s.y3, [-1, 0, 3, 10, 30, 1e9], labels=["0", "1–3", "4–10", "11–30", ">30"])
    d41 = s.groupby("y_bin", observed=True).c.agg(["size", "mean", "median", lambda x: (x > 0).mean()]).rename(columns={"<lambda_0>": "share_pos"})
    # 4-2
    X = ["sim_top10", "n_prior", "rival_strength_mean", "rival_strength_max", "share_coupled", "share_shared_auth",
         "share_focal_cites", "share_rival_cites", "preempted", "log_density"]
    z = s.copy()
    for c in X:
        z[c] = (z[c] - z[c].mean()) / z[c].std()
    m42 = pf.feols(f"c ~ {' + '.join(X)} | eval_c2000", data=z, vcov={"CRV1": "eval_c2000"}).tidy()
    m43 = pf.feols(f"c ~ share_cocited + {' + '.join(X)} | eval_c2000", data=z.assign(share_cocited=(z.share_cocited - z.share_cocited.mean()) / z.share_cocited.std()),
                   vcov={"CRV1": "eval_c2000"}).tidy()
    drivers = m42[["Estimate", "Std. Error", "Pr(>|t|)"]].round(4)
    drivers.to_csv(os.path.join(RES, "01_drivers.csv"))
    m43[["Estimate", "Std. Error", "Pr(>|t|)"]].round(4).to_csv(os.path.join(RES, "01_cocite.csv"))

    # 4-5 基尼
    preds = {"MPC-Net": s.set_index("paper_id").pred_full}
    p1 = pd.read_parquet("/path/to/mpcc/exp1/results/preds_cold.parquet")
    for m in ("MLP-文本", "L4 +竞争"):
        preds[m] = p1[(p1.model == m) & (p1.split == "test")].set_index("paper_id").pred
    nai = "/path/to/mpcc/exp1/results/preds_naip.parquet"
    if os.path.exists(nai):
        q = pd.read_parquet(nai)
        preds["NAIP (AAAI'25)"] = q[q.split == "test"].set_index("paper_id").pred
    gr = []
    for k, grp in s.groupby("eval_c2000"):
        if len(grp) < 20 or grp.y3.sum() == 0:
            continue
        row = {"g": k, "n": len(grp), "真实": gini(grp.y3)}
        for m, pr in preds.items():
            row[m] = gini(np.expm1(pr.reindex(grp.paper_id).to_numpy()))
        gr.append(row)
    gr = pd.DataFrame(gr)
    gr.to_csv(os.path.join(RES, "01_gini.csv"), index=False)
    gsum = gr.drop(columns=["g", "n"]).agg(["mean", "median"]).round(3)
    gcorr = {m: round(gr[["真实", m]].corr(method="spearman").iloc[0, 1], 3) for m in preds}

    # 4-7 个案
    p = con.execute(f"select paper_id, title, first_public_year from read_parquet('{V2}papers.parquet')").df().set_index("paper_id")
    cases = []
    for lab, sub in (("对手信息调低最多", s[s.y3 > 0].nsmallest(3, "c")), ("对手信息调高最多", s[s.y3 > 0].nlargest(3, "c"))):
        for _, x in sub.iterrows():
            top = r[(r.focal_id == x.paper_id) & (r["rank"] <= 3)]
            rivs = "；".join(f"{str(p.title.get(cid))[:60]}（{dy:+d} 年，sim {sm:.3f}）" for cid, dy, sm in zip(top.comp_id, top.dyear, top.sim))
            cases.append(f"- **{lab}**：{str(p.title.get(x.paper_id))[:90]}（{x.Y}）— 真实 y3 = {int(x.y3)}；"
                         f"MPC-Net 预测 {np.expm1(x.pred_full):.1f}，去掉对手 {np.expm1(x.pred_nr):.1f}（c = {x.c:+.2f}）；"
                         f"已被抢先 = {int(x.preempted)}；最相似对手：{rivs}")
    open(os.path.join(RES, "01_cases.md"), "w", encoding="utf-8").write("# 实验 4-7 个案\n\n" + "\n".join(cases) + "\n")

    sig = lambda v: "***" if v < .001 else "**" if v < .01 else "*" if v < .05 else ""
    dtab = "\n".join(f"| {k} | {r_['Estimate']:+.4f}{sig(r_['Pr(>|t|)'])} | {r_['Std. Error']:.4f} |" for k, r_ in drivers.iterrows())
    co_row = m43.loc["share_cocited"]
    md = ["# 实验 4 结果：机制与可解释性\n",
          f"测试集 {len(s):,} 篇。竞争贡献 c = pred(MPCNet5_l1) − pred(B_no_rivals)，log1p 空间，5 个种子平均。\n",
          "## 4-1 竞争贡献的分布\n",
          f"均值 {s.c.mean():+.3f}，中位数 {s.c.median():+.3f}，标准差 {s.c.std():.3f}；c > 0 的比例 {(s.c > 0).mean():.1%}；"
          f"|c| > 0.2（约 ±22% 被引）的比例 {(s.c.abs() > 0.2).mean():.1%}。\n", "按真实被引分档：\n", d41.round(3).to_markdown(), "\n",
          "## 4-2 什么样的竞争集合让模型加分 / 减分（标准化系数，子课题固定效应）\n",
          "| 变量 | 系数 | 标准误 |", "|---|---|---|", dtab, "",
          "## 4-3 互补 vs 替代（事后，用到未来的共被引）\n",
          f"对手中之后与焦点论文被共同引用的比例：均值 {s.share_cocited.mean():.3f}。"
          f"控制 4-2 全部描述量后，共被引比例（标准化）对 c 的系数 {co_row['Estimate']:+.4f}（p = {co_row['Pr(>|t|)']:.2g}）。\n",
          "## 4-5 子课题内被引不平等（基尼系数，子课题 ≥ 20 篇）\n", gsum.to_markdown(),
          f"\n\n各子课题预测基尼与真实基尼的 Spearman：{gcorr}\n",
          "## 4-7 个案\n见 `01_cases.md`。\n"]
    open(os.path.join(RES, "实验4结果.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"01 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
