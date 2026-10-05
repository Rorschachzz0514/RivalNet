"""第七轮 · 结构多样性集成检验（只看验证集）"""
import numpy as np
import pandas as pd
from importlib import util
spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
R = "/path/to/mpcc/exp2/runs"
s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
st = s[s.split == "val"].set_index("paper_id")
cache = {}


def P(n, sd):
    if (n, sd) not in cache:
        p = pd.read_parquet(f"{R}/{n}_s{sd}/preds.parquet", columns=["paper_id", "split", "pred"])
        cache[(n, sd)] = p[p.split == "val"].set_index("paper_id").pred
    return cache[(n, sd)]


def ev(lst, name):
    p = pd.concat([P(n, sd) for n, sd in lst], axis=1).mean(1)
    d = st.assign(pred=p.reindex(st.index)).reset_index()
    e = d.pred - np.log1p(d.y3)
    m = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
    print(f"{name:28s} n={len(lst):2d} MALE {m['MALE']:.4f} 偏差 {e.mean():+.3f} 去偏 {(e - e.mean()).abs().mean():.4f} "
          f"rho {m['Spearman_子课题内']:.4f} ndcg {m['NDCG@10_子课题内']:.4f}", flush=True)


S = [1, 2, 3, 4]
ev([("IN_r2_s30", i) for i in S], "SWA 基准 4 种子")
mix = ["IN_r2_s30", "IN_r2_s30_tcs", "IN_r2_s30_coh1", "IN_r2_s30_xd3"]
ev([(mix[i], i + 1) for i in range(4)], "SWA 4 种结构各 1 个")
ev([("IN_r2_s30", i) for i in S] + [("IN_r3_swa10", i) for i in S], "SWA 基准 8 个")
ev([(n, i) for n in mix for i in [1, 2]], "SWA 4 种结构各 2 个")
ev([(n, i) for n in mix + ["IN_r2_s30_luce"] for i in S], "SWA 5 种结构 × 4 种子")
ev([("AS_r06_1", i) for i in S], "早停基准 4 种子")
ev([("AS_r06_1", i) for i in range(1, 9)], "早停基准 8 种子")
e5 = ["AS_r06_1", "IN_r1_tc_plus", "IN_r1_coh1", "IN_r1_luce"]
ev([(n, i) for n in e5 for i in [1, 2]], "早停 4 种结构各 2 个")
ev([(n, i) for n in e5 + ["IN_r1_xd3", "IN_r1_rel", "IN_r1_tc_strict"] for i in S], "早停 7 种结构 × 4 种子")
