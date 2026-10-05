"""
【01】实验 5-2 · 强对手出现事件：事件研究 / 双重差分（预先登记见 实验5说明.md 第 2 节）

用途
  处理组：首次公开年份 Y ∈ {2017, 2018, 2019} 的论文 i，在 Y+2 年（主分析）出现"强对手"k：SPECTER2 ≥ 0.95、
          k 没有引用 i、没有共同作者、k 第一个完整年份的被引处于同方向同年份前 5%；且 Y+1 年没有强对手。
  对照组：同一子课题(K=2000)、同一年份、同一发表渠道类别，Y+1、Y+2 都没有强对手的论文；按 (log1p c_Y, log1p c_{Y+1})
          最近邻匹配 3 篇（可放回，距离 ≤ 0.3）。
  模型：log1p(c) ~ Σ β_r·处理×1[r]（r = −2..3，基准 r = −1）| 论文 + 匹配组×相对时间，按匹配组聚类；
        单一系数版：处理×事件后；Poisson 版。
  稳健：E = Y+1 的事件（只有一个事件前时期）；强对手前 10% / 前 1%；相似度 ≥ 0.97。

输入
  /path/to/mpcc/subsets/pred_v2/samples.parquet、/path/to/mpcc/subsets/exp0_v2/{later_pairs_95, paper_year}.parquet

输出（results/）
  01_event_coefs.csv（各设定的逐期系数）、01_event_did.csv（单一系数）、01_event.png、01_event.md

用法
  python 01_event_study.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
S = "/path/to/mpcc/subsets/pred_v2/samples.parquet"
LP = "/path/to/mpcc/subsets/exp0_v2/later_pairs_95.parquet"
PY = "/path/to/mpcc/subsets/exp0_v2/paper_year.parquet"


def build(s, lp, cits, lag, sim_min, pct_min, n_match=3, caliper=0.3):
    strong = lp[(~lp.comp_cites_focal) & (~lp.shared_auth) & (lp.sim >= sim_min) & (lp.k_age1_pctl >= pct_min)]
    first = strong.groupby("focal_id").dyear.min()
    treated = s[s.paper_id.map(first) == lag]
    ctrl = s[~s.paper_id.isin(first.index)]
    pre_t = list(range(0, lag))                                      # 事件前的年份偏移（相对 Y）
    feat = [f"lc{k}" for k in pre_t]
    out, gid = [], 0
    keys = ["eval_c2000", "Y", "venue_final"]
    cg = {k: v for k, v in ctrl.groupby(keys)}
    for k, tg in treated.groupby(keys):
        if k not in cg:
            continue
        cc = cg[k]
        A, Bm = tg[feat].to_numpy(), cc[feat].to_numpy()
        d = np.sqrt(((A[:, None, :] - Bm[None, :, :]) ** 2).sum(2))
        for i in range(len(tg)):
            idx = np.argsort(d[i])[:n_match]
            idx = idx[d[i, idx] <= caliper]
            if len(idx) == 0:
                continue
            out.append((gid, tg.paper_id.iloc[i], 1))
            out += [(gid, cc.paper_id.iloc[j], 0) for j in idx]
            gid += 1
    m = pd.DataFrame(out, columns=["grp", "paper_id", "treat"])
    m = m.merge(s[["paper_id", "Y"]], on="paper_id")
    panel = m.merge(cits, on="paper_id")
    panel = panel[(panel.year >= panel.Y) & (panel.year <= panel.Y + lag + 3)]
    panel["rel"] = panel.year - (panel.Y + lag)
    panel["unit"] = panel.grp.astype(str) + "_" + panel.paper_id.astype(str)
    panel["lc"] = np.log1p(panel.cites)
    panel["post"] = (panel.rel >= 0).astype(int)
    panel["treat_post"] = panel.treat * panel.post
    return panel, int(m.treat.sum()), int((m.treat == 0).sum())


def estimate(panel, name, lag):
    rows, did = [], []
    ref = -1
    for r in sorted(panel.rel.unique()):
        if r != ref:
            panel[f"d_{r + 10}"] = ((panel.rel == r) & (panel.treat == 1)).astype(int)
    dvars = [c for c in panel.columns if c.startswith("d_")]
    m = pf.feols(f"lc ~ {' + '.join(dvars)} | unit + grp^rel", data=panel, vcov={"CRV1": "grp"})
    t = m.tidy()
    for c in dvars:
        rows.append({"spec": name, "rel": int(c[2:]) - 10, "beta": t.loc[c, "Estimate"], "se": t.loc[c, "Std. Error"], "p": t.loc[c, "Pr(>|t|)"]})
    rows.append({"spec": name, "rel": ref, "beta": 0.0, "se": 0.0, "p": np.nan})
    m2 = pf.feols("lc ~ treat_post | unit + grp^rel", data=panel, vcov={"CRV1": "grp"}).tidy().loc["treat_post"]
    m3 = pf.fepois("cites ~ treat_post | unit + grp^rel", data=panel, vcov={"CRV1": "grp"}).tidy().loc["treat_post"]
    did = {"spec": name, "n_groups": panel.grp.nunique(), "ols_beta": m2["Estimate"], "ols_se": m2["Std. Error"], "ols_p": m2["Pr(>|t|)"],
           "pois_beta": m3["Estimate"], "pois_se": m3["Std. Error"], "pois_p": m3["Pr(>|t|)"], "pois_pct": (np.exp(m3["Estimate"]) - 1) * 100}
    for c in dvars:
        del panel[c]
    return rows, did


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    s = pd.read_parquet(S, columns=["paper_id", "Y", "eval_c2000", "topic", "venue_final"])
    s = s[s.Y <= 2019].reset_index(drop=True)
    py = pd.read_parquet(PY)
    py = py[py.paper_id.isin(s.paper_id)]
    grid = s[["paper_id", "Y"]].merge(pd.DataFrame({"off": range(0, 6)}), how="cross")
    grid["year"] = grid.Y + grid.off
    cits = grid.merge(py, on=["paper_id", "year"], how="left").fillna({"cites": 0})[["paper_id", "year", "cites", "off"]]
    w = cits[cits.off <= 1].pivot(index="paper_id", columns="off", values="cites")
    s["lc0"], s["lc1"] = np.log1p(s.paper_id.map(w[0])), np.log1p(s.paper_id.map(w[1]))
    cits = cits[["paper_id", "year", "cites"]]
    lp = pd.read_parquet(LP)
    specs = [("主分析：E = Y+2，前 5%，相似度 ≥ 0.95", 2, 0.95, 0.95),
             ("稳健：E = Y+1", 1, 0.95, 0.95),
             ("稳健：前 10%", 2, 0.95, 0.90),
             ("稳健：前 1%", 2, 0.95, 0.99),
             ("稳健：相似度 ≥ 0.97", 2, 0.97, 0.95)]
    coefs, dids, sizes = [], [], []
    for name, lag, sm, pc in specs:
        panel, nt, nc = build(s, lp, cits, lag, sm, pc)
        sizes.append({"spec": name, "treated": nt, "control_matches": nc})
        r, d = estimate(panel, name, lag)
        coefs += r
        dids.append(d)
        print(f"{name}: 处理 {nt:,}，对照匹配 {nc:,}；单一系数 OLS {d['ols_beta']:+.4f}（p {d['ols_p']:.2g}），"
              f"Poisson {d['pois_pct']:+.1f}%（p {d['pois_p']:.2g}）（{time.time() - t0:.0f}s）", flush=True)
    co = pd.DataFrame(coefs).sort_values(["spec", "rel"])
    di = pd.DataFrame(dids).merge(pd.DataFrame(sizes), on="spec")
    co.to_csv(os.path.join(RES, "01_event_coefs.csv"), index=False)
    di.to_csv(os.path.join(RES, "01_event_did.csv"), index=False)

    main_name = specs[0][0]
    mc = co[co.spec == main_name].set_index("rel")
    md_ = di.set_index("spec").loc[main_name]
    pre_ok = bool(mc.loc[-2, "p"] > 0.1)
    passed = bool(md_.ols_beta < 0 and md_.ols_p < 0.05 and pre_ok)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.8))
        en = {specs[0][0]: "main: E=Y+2, top 5%, sim>=0.95", specs[2][0]: "top 10%", specs[3][0]: "top 1%", specs[4][0]: "sim>=0.97"}
        for name, _, _, _ in specs[:1] + specs[2:]:
            c = co[co.spec == name]
            ax.errorbar(c.rel + (0.05 * specs.index(next(x for x in specs if x[0] == name))), c.beta, yerr=1.96 * c.se,
                        marker="o", capsize=3, label=en[name], lw=1)   # 服务器无中文字体，图例用英文
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("years relative to strong-rival event")
        ax.set_ylabel("effect on log(1 + citations)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "01_event.png"), dpi=150)
    except Exception as e:
        print("画图失败:", e)
    md = ["# 实验 5-2 强对手出现事件（事件研究 / 双重差分）\n",
          "## 判定\n",
          f"主分析：处理 {int(md_.treated):,} 篇，对照匹配 {int(md_.control_matches):,} 篇次。"
          f"事件后平均效应 OLS β = {md_.ols_beta:+.4f}（p = {md_.ols_p:.2g}），Poisson {md_.pois_pct:+.1f}%（p = {md_.pois_p:.2g}）；"
          f"事件前 β_{{-2}} = {mc.loc[-2, 'beta']:+.4f}（p = {mc.loc[-2, 'p']:.2g}）；**{'通过' if passed else '未通过'}**"
          "（标准：事件后效应 < 0 且显著，事件前不显著）。\n",
          "## 单一系数（处理 × 事件后）\n", di.round(4).to_markdown(index=False), "\n",
          "## 逐期系数（基准 r = −1）\n", co.round(4).to_markdown(index=False), "\n", "图：`01_event.png`\n"]
    open(os.path.join(RES, "01_event.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:3]))
    print(f"01 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
