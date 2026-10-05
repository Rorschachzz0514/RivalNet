"""
【08】实验 2 · 第六轮迭代后的最终评价（测试集只在此处使用一次；方案在运行前写入 autosearch_notes/reflections.md）

最终方案
  配置 AS_r06_1（自动迭代中验证集最好）；
  A 组 = 2017–2019 训练、2020 早停，种子 1–10；B 组 = 2017–2020 重训、每个种子的轮数照搬 A 组同种子的最佳轮（FINAL_B<种子>）；
  最终预测 = A 组集成与 B 组集成的平均。
对比
  实验 1 的 MLP-文本、LightGBM L4、NAIP；第五版定版 MPCNet5_l1（5 个种子：单种子均值与集成）。
指标与显著性
  同实验 1 / 2（调用实验 1【04】的指标函数）；差异按子课题整组重抽样 1,000 次。

输出（results/）
  08_final_metrics.csv、实验2结果_终版.md

用法
  python 08_final_eval.py
"""
import glob
import os
import sys
from importlib import util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)


def load(pattern, split="test"):
    fs = sorted(glob.glob(os.path.join(C.RUNS, pattern, "preds.parquet")))
    ps = []
    for f in fs:
        d = pd.read_parquet(f)
        ps.append(d[d.split == split].set_index("paper_id").pred)
    return pd.concat(ps, axis=1), len(fs)


def main():
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
    st = s[s.split == "test"].set_index("paper_id")
    A, na = load("AS_r06_1_s*")
    B, nb = load("FINAL_B*_s*")
    V5, n5 = load("MPCNet5_l1_s*")
    preds = {"MPC-Net 终版（A + B 集成）": (A.mean(1) + B.mean(1)) / 2, "A 组集成（2017–2019 训练）": A.mean(1),
             "B 组集成（2017–2020 重训）": B.mean(1), "第五版 MPCNet5_l1（5 种子集成）": V5.mean(1)}
    pc = pd.read_parquet(os.path.join(C.EXP1_RES, "preds_cold.parquet"))
    for m in ("MLP-文本", "L4 +竞争"):
        preds[m] = pc[(pc.model == m) & (pc.split == "test")].set_index("paper_id").pred
    nf = os.path.join(C.EXP1_RES, "preds_naip.parquet")
    if os.path.exists(nf):
        q = pd.read_parquet(nf)
        preds["NAIP (AAAI'25)"] = q[q.split == "test"].set_index("paper_id").pred
    rows = []
    for m, p in preds.items():
        d = st.assign(pred=p.reindex(st.index)).reset_index()
        d = d.assign(y=d.y3, ly=np.log1p(d.y3))
        rows.append({"model": m, **EV.metrics(d)})
    single = []
    for i in range(V5.shape[1]):                       # 各种子的列名都是 pred，按位置取
        d = st.assign(pred=V5.iloc[:, i].reindex(st.index)).reset_index()
        single.append(EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3))))
    rows.append({"model": "第五版 MPCNet5_l1（单种子均值）", **pd.DataFrame(single).mean().to_dict()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "08_final_metrics.csv"), index=False)

    sub = np.sort(st.eval_c2000.unique())
    rng = np.random.default_rng(2026)
    W = rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=1000).astype(float)
    ly = np.log1p(st.y3)

    def agg(p):
        d = st.assign(pred=p.reindex(st.index))
        g = d.assign(ae=(d.pred - ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "size")).reindex(sub).fillna(0)
        rs = EV.group_rho(d.reset_index().assign(y=d.y3.to_numpy()), "eval_c2000").set_index("g")
        rn = (rs.rho * rs.n).reindex(sub).fillna(0).to_numpy()
        nr = rs.n.reindex(sub).fillna(0).to_numpy()
        return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy()), (W @ rn) / (W @ nr)

    fm, fr = agg(preds["MPC-Net 终版（A + B 集成）"])
    comp = []
    for m in ("MLP-文本", "第五版 MPCNet5_l1（5 种子集成）"):
        bm, br = agg(preds[m])
        dm, dr = fm - bm, fr - br
        comp.append({"比较": f"终版 − {m}", "MALE 差": dm.mean(), "MALE 95% 区间": f"[{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]",
                     "子课题内 Spearman 差": dr.mean(), "Spearman 95% 区间": f"[{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]"})
    cols = ["model", "MALE", "RMSLE", "MAE", "RMSE", "Spearman_子课题内", "NDCG@10_子课题内", "份额L1_子课题内", "Spearman_全体"]
    md = ["# 实验 2 结果（终版：第六轮自动迭代 + 训练验证重训 + 种子集成）\n",
          f"测试集 = 2021 年焦点论文（{len(st):,} 篇），只在此处使用一次。A 组 {na} 个模型、B 组 {nb} 个模型；第五版 {n5} 个种子。\n",
          res[cols].round(4).to_markdown(index=False), "\n", "## 显著性（按子课题重抽样 1,000 次）\n",
          pd.DataFrame(comp).round(4).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, "实验2结果_终版.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
