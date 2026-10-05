"""第七轮 · 双通道（替代 / 互补）与吸引力份额项的机制检验：模型给出的"被抢走 / 被带动"是否与可观测的竞争情形一致（验证 + 测试集）"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R = "/path/to/mpcc/exp2/runs"
s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet",
                    columns=["paper_id", "split", "y3", "n_preempted", "c_m1_95", "c_m2_95", "c_pb365_95", "rival_cites_m1", "rival_cites_m1_max",
                             "n_cand_ym1", "sub_supply_Y"]).set_index("paper_id")
for cfg, cols in (("IN_r2_s30_tcs", ["part_subst", "part_compl"]), ("IN_r2_s30_luce", ["part_luce"])):
    ps = [pd.read_parquet(f"{R}/{cfg}_s{i}/preds.parquet", columns=["paper_id", "split"] + cols).set_index("paper_id") for i in (1, 2, 3, 4)]
    p = ps[0][["split"]].copy()
    for c in cols:
        p[c] = pd.concat([q[c] for q in ps], axis=1).mean(1)
    d = p.join(s.drop(columns="split"))
    d = d[d.split.isin(["val", "test"])]
    print(f"\n== {cfg}（4 种子平均；验证 + 测试 {len(d):,} 篇）")
    print(d[cols].describe().round(3).T.to_string())
    for c in cols:
        out = []
        for x in ["n_preempted", "c_pb365_95", "c_m1_95", "c_m2_95", "rival_cites_m1", "rival_cites_m1_max", "sub_supply_Y"]:
            ok = d[x].notna()
            out.append(f"{x} {spearmanr(d.loc[ok, c], d.loc[ok, x])[0]:+.3f}")
        print(f"  {c} 与可观测量的 Spearman：", "; ".join(out))
        pre = d.n_preempted.fillna(0) > 0
        print(f"  {c}：被抢先（n_preempted>0，{pre.mean():.1%}）均值 {d.loc[pre, c].mean():.4f} vs 其他 {d.loc[~pre, c].mean():.4f}")
