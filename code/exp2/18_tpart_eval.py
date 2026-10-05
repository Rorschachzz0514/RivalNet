"""
【18】修改计划 A3 · 按时间分通道的可分解竞争结构（TP_strict / TP_plus）评价

用途
  1. 精度：TP_strict、TP_plus 与 AS_r06_1（终版配置）种子 1–3 配对比较。主要看验证集（选择依据）：
     MALE、整体偏差、去偏 MALE（预测减去该集合的平均误差后的 MALE；第七轮发现验证 MALE 随整体水平波动）、子课题内 Spearman；
     测试集只作参考。另报告 3 种子集成。
  2. 通道是否塌缩：part_subst（"抢先"通道效应，进入预测时取负号）、part_compl（"带火"通道效应）的分布，
     以及它们与竞争集合结构的 Spearman 相关：
       n_S = 同期对手数（发表不早于焦点论文 183 天之前），n_V = 更早的前作数，maxsim_S / maxsim_V = 两类对手中的最高相似度，
       是否为孪生论文中的晚出者 / 早出者。
  3. 孪生反事实（若已跑 --placebo cf:twins）：屏蔽孪生对方后的效应，与 RivalNet-base 对照。

输出
  results/18_tpart_eval.md、results/18_tpart_eval.csv

用法
  python 18_tpart_eval.py
"""
import glob
import json
import os
from importlib import util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
SAMPLES = "/path/to/mpcc/subsets/pred_v2/samples.parquet"
RIVALS = "/path/to/mpcc/subsets/pred_v2/rivals.parquet"
TWINS = "/path/to/mpcc/exp0_analysis/results_v2/15_twins_pairs.parquet"
CFGS = ["AS_r06_1", "TP_strict", "TP_plus"]
SEEDS = [1, 2, 3]


def met(d):
    e = d.pred - d.ly
    m = EV.metrics(d)
    return {"MALE": m["MALE"], "bias": e.mean(), "MALE_debiased": (e - e.mean()).abs().mean(), "rho_sub": m["Spearman_子课题内"],
            "NDCG10": m["NDCG@10_子课题内"]}


def main():
    s = pd.read_parquet(SAMPLES, columns=["paper_id", "split", "y3", "topic", "eval_c2000"]).set_index("paper_id")
    rows, md = [], ["# 修改计划 A3 · 按时间分通道（D = 183 天）的可分解竞争结构\n"]
    preds = {}
    for cfg in CFGS:
        for sd in SEEDS:
            f = os.path.join(RUNS, f"{cfg}_s{sd}", "preds.parquet")
            d = pd.read_parquet(f)
            preds[(cfg, sd)] = d
            ep = json.load(open(os.path.join(RUNS, f"{cfg}_s{sd}", "summary.json")))["best_epoch"]
            for sp in ("val", "test"):
                x = d[d.split == sp].set_index("paper_id")
                q = s.loc[x.index].assign(pred=x.pred)
                rows.append({"config": cfg, "seed": sd, "split": sp, "best_epoch": ep,
                             **met(q.reset_index().assign(y=q.y3.values, ly=np.log1p(q.y3.values)))})
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(HERE, "results", "18_tpart_eval.csv"), index=False)
    md.append("## 1. 精度（种子 1–3 的均值 ± 标准差）\n")
    agg = r.groupby(["split", "config"])[["MALE", "bias", "MALE_debiased", "rho_sub", "NDCG10"]].agg(["mean", "std"])
    lines = ["| 集合 | 配置 | MALE | 偏差 | 去偏 MALE | ρ_sub | NDCG@10 |", "|---|---|---|---|---|---|---|"]
    for (sp, cfg), g in agg.iterrows():
        lines.append(f"| {sp} | {cfg} | " + " | ".join(f"{g[(k, 'mean')]:.4f} ± {g[(k, 'std')]:.4f}" for k in
                                                       ("MALE", "bias", "MALE_debiased", "rho_sub", "NDCG10")) + " |")
    md += lines
    # 配对差（同一种子，相对 AS_r06_1）
    md.append("\n配对差（同一种子，减去 AS_r06_1；均值 [最小, 最大]）：\n")
    md.append("| 集合 | 配置 | Δ 去偏 MALE | Δ ρ_sub |")
    md.append("|---|---|---|---|")
    for sp in ("val", "test"):
        base = r[(r.config == "AS_r06_1") & (r.split == sp)].set_index("seed")
        for cfg in CFGS[1:]:
            x = r[(r.config == cfg) & (r.split == sp)].set_index("seed")
            dm, dr = x.MALE_debiased - base.MALE_debiased, x.rho_sub - base.rho_sub
            md.append(f"| {sp} | {cfg} | {dm.mean():+.4f} [{dm.min():+.4f}, {dm.max():+.4f}] | {dr.mean():+.4f} [{dr.min():+.4f}, {dr.max():+.4f}] |")
    # 3 种子集成
    md.append("\n3 种子集成：\n")
    md.append("| 集合 | 配置 | MALE | 去偏 MALE | ρ_sub | NDCG@10 |")
    md.append("|---|---|---|---|---|---|")
    for sp in ("val", "test"):
        for cfg in CFGS:
            p = pd.concat([preds[(cfg, sd)].loc[lambda x: x.split == sp].set_index("paper_id").pred for sd in SEEDS], axis=1).mean(1)
            q = s.loc[p.index].assign(pred=p)
            m = met(q.reset_index().assign(y=q.y3.values, ly=np.log1p(q.y3.values)))
            md.append(f"| {sp} | {cfg} | {m['MALE']:.4f} | {m['MALE_debiased']:.4f} | {m['rho_sub']:.4f} | {m['NDCG10']:.4f} |")

    # ---- 2. 通道分析（验证 + 测试）
    ids = s.index[s.split.isin(["val", "test"])]
    rv = pd.read_parquet(RIVALS, columns=["focal_id", "sim", "dyear", "ddays"])
    rv = rv[rv.focal_id.isin(ids)]
    S = np.where(rv.ddays.isna(), rv.dyear == 0, rv.ddays >= -183)
    rv = rv.assign(S=S)
    g = rv.groupby("focal_id")
    feat = pd.DataFrame({"n_S": g.S.sum(), "maxsim_S": rv[rv.S].groupby("focal_id").sim.max(),
                         "maxsim_V": rv[~rv.S].groupby("focal_id").sim.max()})
    feat["n_V"] = g.size() - feat.n_S
    t = pd.read_parquet(TWINS)
    feat["twin_late"] = feat.index.isin(t.late_id)
    feat["twin_early"] = feat.index.isin(t.early_id)
    md.append("\n## 2. 两个通道是否\"活着\"（验证 + 测试，3 种子平均）\n")
    md.append("| 配置 | 通道 | 均值 | 标准差 | >0.01 的比例 | ρ(n_S) | ρ(n_V) | ρ(maxsim_S) | ρ(maxsim_V) | 孪生晚出者均值 | 孪生早出者均值 | 其他论文均值 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in CFGS[1:]:
        for col, lab in (("part_subst", "抢先（取负号进入预测）"), ("part_compl", "带火")):
            v = pd.concat([preds[(cfg, sd)].query("split in ['val', 'test']").set_index("paper_id")[col] for sd in SEEDS], axis=1).mean(1)
            f = feat.reindex(v.index)
            cor = lambda c: pd.concat([v, f[c].astype(float)], axis=1).corr("spearman").iloc[0, 1]
            md.append(f"| {cfg} | {lab} | {v.mean():.4f} | {v.std():.4f} | {(v > 0.01).mean():.3f} | {cor('n_S'):+.3f} | {cor('n_V'):+.3f} | "
                      f"{cor('maxsim_S'):+.3f} | {cor('maxsim_V'):+.3f} | {v[f.twin_late.values].mean():.4f} | {v[f.twin_early.values].mean():.4f} | "
                      f"{v[~(f.twin_late.values | f.twin_early.values)].mean():.4f} |")

    # ---- 3. 孪生反事实
    md.append("\n## 3. 孪生反事实：ŝ − ŝ(屏蔽孪生对方)，3 种子平均（log 尺度）\n")
    md.append("| 配置 | 对晚出者 | 对早出者 | 对数 |")
    md.append("|---|---|---|---|")
    for cfg in ["MPCNet5_l1"] + CFGS:
        effs = {"late": [], "early": []}
        n = 0
        for sd in SEEDS:
            dd = os.path.join(RUNS, f"{cfg}_s{sd}")
            fcf = os.path.join(dd, "preds_cf_twins.parquet")
            if not os.path.exists(fcf):
                continue
            a = pd.read_parquet(os.path.join(dd, "preds.parquet")).query("split in ['val', 'test']").set_index("paper_id").pred
            b = pd.read_parquet(fcf).set_index("paper_id").pred
            h = pd.read_parquet(os.path.join(dd, "cf_twins_hits.parquet"))
            for role in ("late", "early"):
                fid = h[(h.role == role) & h.hit].focal_id.unique()
                fid = [i for i in fid if i in b.index]
                effs[role].append((a.reindex(fid) - b.reindex(fid)))
                n = len(fid)
        if effs["late"]:
            el = pd.concat(effs["late"], axis=1).mean(1)
            ee = pd.concat(effs["early"], axis=1).mean(1)
            md.append(f"| {cfg} | {el.mean():+.4f}（{100 * (np.exp(el.mean()) - 1):+.1f}%） | {ee.mean():+.4f}（{100 * (np.exp(ee.mean()) - 1):+.1f}%） | {n} |")
    open(os.path.join(HERE, "results", "18_tpart_eval.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
