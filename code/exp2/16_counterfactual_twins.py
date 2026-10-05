"""
【16】修改计划 A2 · 反事实验证：模型学到的对手效应 vs 孪生论文准实验（实验 0 检验 0-5）

用途
  孪生论文对（exp0 v2：相似度 ≥ 0.97、相差 1–183 天、无共同作者、互不引用）中，两篇都在验证集或测试集的对
  （对 2017–2019 年训练的模型都是样本外）。对每一对计算"晚出 − 早出"：
    obs        真实 log(1+y) 差距（y = 三年被引，全部来源口径）
    full       RivalNet-base（MPCNet5_l1，5 种子平均）预测差距
    norival    去掉对手的同配置模型（B_no_rivals，5 种子平均）预测差距
    final      终版 A 组（AS_r06_1，10 种子平均）预测差距（若已跑）
  反事实屏蔽（02_mpcnet.py --placebo cf:twins）：把孪生对方从自己的竞争集合中屏蔽后重新预测，
    eff_late  = ŝ_晚出 − ŝ_晚出(屏蔽早出者)：早出孪生的存在对晚出者的效应（"被抢先"）
    eff_early = ŝ_早出 − ŝ_早出(屏蔽晚出者)：晚出孪生的存在对早出者的效应
    （一篇论文若属于多对，屏蔽的是它的全部孪生对方；只统计对方确实在竞争集合内的对）
  子组与实验 0 对照：全部、同一年份、相似度 ≥ 0.98、时间差 1–30 / 31–90 / 91–183 天、两方都精确到日。
  不确定性：按论文对重抽样 1,000 次，95% 区间。
  代码自查：反事实预测中，不属于任何孪生对的论文必须与原预测完全相同。

输出
  results/16_cf_twins.csv、results/16_cf_twins.md

用法
  python 16_counterfactual_twins.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SAMPLES = "/path/to/mpcc/subsets/pred_v2/samples.parquet"
TWINS = "/path/to/mpcc/exp0_analysis/results_v2/15_twins_pairs.parquet"
EXP0 = {"all": -13.9, "same year": -24.3, "sim ≥ 0.98": -28.5, "gap 1–30 d": -2.8, "gap 31–90 d": -14.6, "gap 91–183 d": -14.5,
        "both exact day": -7.7}


def ens(cfg, fname="preds.parquet", n_expected=None):
    fs = sorted(glob.glob(os.path.join(RUNS, f"{cfg}_s[0-9]*", fname)))
    fs = [f for f in fs if "_chk_" not in f]
    if not fs:
        return None, 0
    if n_expected:
        assert len(fs) == n_expected, (cfg, fname, len(fs))
    ps = []
    for f in fs:
        d = pd.read_parquet(f, columns=["paper_id", "split", "pred"])
        ps.append(d[d.split.isin(["val", "test"])].set_index("paper_id").pred)
    return pd.concat(ps, axis=1).mean(1), len(fs)


def self_check(cfg):
    """反事实预测里，不受屏蔽影响的论文应与原预测逐位相同"""
    worst = 0.0
    for d_ in sorted(glob.glob(os.path.join(RUNS, f"{cfg}_s[0-9]*"))):
        if "_chk_" in d_ or not os.path.exists(os.path.join(d_, "preds_cf_twins.parquet")):
            continue
        a = pd.read_parquet(os.path.join(d_, "preds.parquet"), columns=["paper_id", "split", "pred"])
        a = a[a.split.isin(["val", "test"])].set_index("paper_id").pred
        b = pd.read_parquet(os.path.join(d_, "preds_cf_twins.parquet")).set_index("paper_id").pred
        h = pd.read_parquet(os.path.join(d_, "cf_twins_hits.parquet"))
        touched = set(h.loc[h.hit, "focal_id"])
        keep = [i for i in b.index if i not in touched]
        worst = max(worst, float((a.reindex(keep) - b.reindex(keep)).abs().max()))
    return worst


def boot(x, rng, B=1000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def pct(v):
    return 100 * (np.exp(v) - 1)


def main():
    s = pd.read_parquet(SAMPLES, columns=["paper_id", "split", "y3"]).set_index("paper_id")
    t = pd.read_parquet(TWINS)
    sp_e, sp_l = s.split.reindex(t.early_id).to_numpy(), s.split.reindex(t.late_id).to_numpy()
    t = t[np.isin(sp_e, ["val", "test"]) & np.isin(sp_l, ["val", "test"])].reset_index(drop=True)
    ly = np.log1p(s.y3)
    t["obs"] = ly.reindex(t.late_id).to_numpy() - ly.reindex(t.early_id).to_numpy()
    md = ["# 修改计划 A2 · 反事实验证：孪生论文（晚出 − 早出）\n",
          f"论文对：两篇都在验证集或测试集，共 {len(t)} 对（全部 7,963 对中）。差距为 log(1+y) 之差；百分比 = exp(差距) − 1。"
          "区间为按论文对重抽样 1,000 次的 95% 区间。实验 0 的数值（全部年份、含控制变量）列在最后一列作参照。\n"]
    models = {}
    for key, cfg, n in (("full", "MPCNet5_l1", 5), ("norival", "B_no_rivals", 5), ("final", "AS_r06_1", 10)):
        p, k = ens(cfg, n_expected=n)
        if p is None:
            continue
        models[key] = p
        t[key] = p.reindex(t.late_id).to_numpy() - p.reindex(t.early_id).to_numpy()
        cf, kc = ens(cfg, "preds_cf_twins.parquet")
        if cf is not None and key != "norival":
            assert kc == k, (cfg, kc, k)
            chk = self_check(cfg)
            md.append(f"- 代码自查（{cfg}）：反事实预测中未受屏蔽的论文与原预测的最大差异 = {chk:.2e}（应为 0）")
            assert chk < 1e-5, f"{cfg} 反事实预测改变了不相关论文"
            hits = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(RUNS, f"{cfg}_s1", "cf_twins_hits.parquet"))])
            hl = hits[hits.role == "late"].set_index("focal_id").hit
            he = hits[hits.role == "early"].set_index("focal_id").hit
            t[f"{key}_hit_late"] = hl.groupby(level=0).max().reindex(t.late_id).fillna(False).to_numpy()
            t[f"{key}_hit_early"] = he.groupby(level=0).max().reindex(t.early_id).fillna(False).to_numpy()
            t[f"{key}_eff_late"] = p.reindex(t.late_id).to_numpy() - cf.reindex(t.late_id).to_numpy()
            t[f"{key}_eff_early"] = p.reindex(t.early_id).to_numpy() - cf.reindex(t.early_id).to_numpy()
    md.append("")
    groups = {"all": np.ones(len(t), bool), "same year": t.same_year.to_numpy(), "sim ≥ 0.98": (t.sim >= 0.98).to_numpy(),
              "gap 1–30 d": (t.gap <= 30).to_numpy(), "gap 31–90 d": ((t.gap > 30) & (t.gap <= 90)).to_numpy(),
              "gap 91–183 d": (t.gap > 90).to_numpy(), "both exact day": t.both_day.to_numpy()}
    rng = np.random.default_rng(2026)
    rows = []
    for g, msk in groups.items():
        sub = t[msk]
        r = {"group": g, "n_pairs": len(sub)}
        if "full" in sub and "norival" in sub:
            sub = sub.assign(diff=sub["full"] - sub["norival"])          # 对手模块对差距的贡献（同一论文对，配对）
        for col in ("obs", "full", "norival", "final", "diff"):
            if col in sub:
                m, lo, hi = boot(sub[col], rng)
                r.update({f"{col}": m, f"{col}_lo": lo, f"{col}_hi": hi})
        for key in ("full", "final"):
            for role in ("late", "early"):
                c, hcol = f"{key}_eff_{role}", f"{key}_hit_{role}"
                if c in sub:
                    v = sub.loc[sub[hcol].astype(bool), c]
                    if len(v):
                        m, lo, hi = boot(v, rng)
                        r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi, f"{c}_n": len(v)})
        if "full" in sub:
            r["spearman_full_obs"] = sub[["full", "obs"]].corr("spearman").iloc[0, 1]
            r["spearman_norival_obs"] = sub[["norival", "obs"]].corr("spearman").iloc[0, 1] if "norival" in sub else np.nan
        r["exp0_pct"] = EXP0[g]
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results", "16_cf_twins.csv"), index=False)

    def f(r, c):
        if c not in r or pd.isna(r[c]):
            return "–"
        return f"{pct(r[c]):+.1f}% [{pct(r[c + '_lo']):+.1f}, {pct(r[c + '_hi']):+.1f}]"

    md.append("## 1. 对内差距（晚出 − 早出）\n")
    md.append("| 子组 | 对数 | 真实 | RivalNet-base | 去掉对手 | 终版 A 组 | 实验 0 |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        md.append(f"| {r.group} | {r.n_pairs} | {f(r, 'obs')} | {f(r, 'full')} | {f(r, 'norival')} | {f(r, 'final')} | {r.exp0_pct:+.1f}% |")
    md.append("\n对手模块的贡献（RivalNet-base 差距 − 去掉对手的差距，log 尺度，配对重抽样 95% 区间）：\n")
    md.append("| 子组 | 贡献（log） | 占「真实 − 去掉对手」剩余差距的比例 |")
    md.append("|---|---|---|")
    for _, r in res.iterrows():
        if "diff" in r and not pd.isna(r["diff"]):
            share = r["diff"] / (r["obs"] - r["norival"]) if abs(r["obs"] - r["norival"]) > 0.01 else np.nan
            sh = f"{share:.0%}" if not pd.isna(share) else "–"
            md.append(f"| {r.group} | {r['diff']:+.4f} [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}] | {sh} |")
    md.append("\n## 2. 反事实屏蔽：孪生对方的存在对自己的效应（ŝ − ŝ(屏蔽对方)）\n")
    md.append("| 子组 | 对晚出者（RivalNet-base） | 对早出者（RivalNet-base） | 对晚出者（终版 A 组） | 对早出者（终版 A 组） |")
    md.append("|---|---|---|---|---|")
    for _, r in res.iterrows():
        md.append(f"| {r.group} | {f(r, 'full_eff_late')} | {f(r, 'full_eff_early')} | {f(r, 'final_eff_late')} | {f(r, 'final_eff_early')} |")
    md.append("\n## 3. 模型能否分辨「哪一对差距更大」（论文对之间的 Spearman：预测差距 vs 真实差距）\n")
    md.append("| 子组 | RivalNet-base | 去掉对手 |")
    md.append("|---|---|---|")
    for _, r in res.iterrows():
        if "spearman_full_obs" in r:
            md.append(f"| {r.group} | {r.spearman_full_obs:.3f} | {r.spearman_norival_obs:.3f} |")
    open(os.path.join(HERE, "results", "16_cf_twins.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
