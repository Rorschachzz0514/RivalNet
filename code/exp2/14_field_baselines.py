"""
【14】实验 6 补充 · 三个学科上 SHARE-Net-base 与最强对比方法的差异（论文跨学科表的最后两行）

用途
  每个学科（AI / 肿瘤学 _sf2730 / 应用数学 _sf2604）在测试集上评价：
    SHARE-Net-base（MPCNet5_l1 种子集成；AI 5 个、其他 3 个）、去掉对手（B_no_rivals）、
    MLP-文本、LightGBM L4（实验 1 / 6）、PLM-FT（【11】，种子集成）；
  "最强对比方法"分别按 MALE 与子课题内 Spearman 选出（不含 SHARE-Net 自身的变体），
  差异按子课题整组重抽样 1,000 次给出 95% 区间。

输出
  results/14_field_baselines.csv、results/14_field_baselines.md

用法
  python 14_field_baselines.py
"""
import glob
import os
from importlib import util

import numpy as np
import pandas as pd

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = {"AI": "", "Oncology": "_sf2730", "Applied math": "_sf2604"}


def ens(paths, split="test"):
    ps = []
    for f in sorted(paths):
        d = pd.read_parquet(f)
        d = d[d.split == split]
        ps.append(d.set_index("paper_id").pred)
    return pd.concat(ps, axis=1).mean(1) if ps else None, len(ps)


def main():
    rows, md = [], ["# 三个学科：SHARE-Net-base 与最强对比方法（测试集）\n"]
    for field, tag in FIELDS.items():
        s = pd.read_parquet(f"/path/to/mpcc/subsets/pred{tag}_v2/samples.parquet", columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
        st = s[s.split == "test"].set_index("paper_id")
        M = {}
        runs = os.path.join(HERE, f"runs{tag}")
        M["SHARE-Net-base"] = ens(glob.glob(f"{runs}/MPCNet5_l1_s*/preds.parquet"))
        M["w/o rivals"] = ens(glob.glob(f"{runs}/B_no_rivals_s*/preds.parquet"))
        pc = pd.read_parquet(f"/path/to/mpcc/exp1/results{tag}/preds_cold.parquet")
        for m, lab in (("MLP-文本", "MLP"), ("L4 +竞争", "LightGBM")):
            q = pc[(pc.model == m) & (pc.split == "test")].set_index("paper_id").pred
            M[lab] = (q, 1)
        M["PLM-FT"] = ens(glob.glob(f"/path/to/mpcc/baselines/plm_ft/preds_all{tag}_s[0-9]*.parquet"))
        if field == "AI":
            nf = "/path/to/mpcc/exp1/results/preds_naip.parquet"
            q = pd.read_parquet(nf)
            M["NAIP"] = (q[q.split == "test"].set_index("paper_id").pred, 1)
        met = {}
        for lab, (p, n) in M.items():
            if p is None:
                continue
            d = st.assign(pred=p.reindex(st.index)).reset_index().dropna(subset=["pred"])
            m = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
            met[lab] = m
            rows.append({"field": field, "method": lab, "seeds": n, "n": len(d), "MALE": m["MALE"], "rho_sub": m["Spearman_子课题内"],
                         "NDCG10": m["NDCG@10_子课题内"]})
        base = [k for k in met if k not in ("SHARE-Net-base", "w/o rivals")]
        best_m = min(base, key=lambda k: met[k]["MALE"])
        best_r = max(base, key=lambda k: met[k]["Spearman_子课题内"])
        sub = np.sort(st.eval_c2000.unique())
        rng = np.random.default_rng(2026)
        W = rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=1000).astype(float)
        ly = np.log1p(st.y3)

        def agg(p):
            d = st.assign(pred=p.reindex(st.index))
            g = d.assign(ae=(d.pred - ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "count")).reindex(sub).fillna(0)
            rs = EV.group_rho(d.dropna(subset=["pred"]).reset_index().assign(y=lambda x: x.y3), "eval_c2000").set_index("g")
            rn = (rs.rho * rs.n).reindex(sub).fillna(0).to_numpy()
            nr = rs.n.reindex(sub).fillna(0).to_numpy()
            return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy()), (W @ rn) / (W @ nr)

        fm, fr = agg(M["SHARE-Net-base"][0])
        bm, _ = agg(M[best_m][0])
        _, br = agg(M[best_r][0])
        dm, dr = fm - bm, fr - br
        md.append(f"## {field}\n")
        md.append(pd.DataFrame([r for r in rows if r["field"] == field]).round(4).to_markdown(index=False))
        md.append(f"\n最强对比方法（MALE）：{best_m}；ΔMALE（SHARE-Net-base − 它）= {dm.mean():+.4f} [{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]")
        md.append(f"最强对比方法（子课题内 Spearman）：{best_r}；Δρ = {dr.mean():+.4f} [{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]\n")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "14_field_baselines.csv"), index=False)
    open(os.path.join(HERE, "results", "14_field_baselines.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
