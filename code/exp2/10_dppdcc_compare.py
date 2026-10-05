"""
【10】实验 2 · 与 DPPDCC（CIKM 2024）的比较：DPPDCC 口径（语料内被引，Y+1～Y+3 年之和），2021 年论文

口径与公平性
  DPPDCC 的标签只统计语料内部的引用（data_pipeline/35 导出），与主实验"全部来源被引"不同；
  MPC-Net 在同一口径上重新训练（tensors_v6dp，配置 DP_full = 2017–2019 训练、DP_2019 = 与 DPPDCC 相同只用 2019 年训练），各 5 个种子。
  DPPDCC 严格冷启动（third_party/DPPDCC/prep_split_coldstart.py）：train = 2019 年、val = 2020 年、test = 2021 年论文；
  只用按验证集选出的全局最优检查点（DPPDCC 官方 get_test_results 会在测试集上挑检查点，我们不采用）。
指标与显著性
  同实验 1 / 2（调用实验 1【04】的指标函数）；差异按子课题整组重抽样 1,000 次。

输入
  /path/to/mpcc/third_party/DPPDCC/results/mpcc_ai/<模型名>_results_<k>.csv（k = 全局最优检查点在测试记录中的序号）
  runs/DP_full_s*/、runs/DP_2019_s*/；tensors_v6dp/{y.npy, meta.parquet}
输出（results/）
  10_dppdcc_metrics.csv、实验2结果_DPPDCC比较.md

用法
  python 10_dppdcc_compare.py
"""
import glob
import json
import os
import re
import sys
from importlib import util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
DP = "/path/to/mpcc/third_party/DPPDCC/results/mpcc_ai/"


def dppdcc_preds():
    """找到全局最优检查点（<模型名>.pkl，按验证集选出）对应的测试结果文件"""
    rec = [f for f in glob.glob(DP + "DPPDCC*_records_test.csv")]
    assert len(rec) == 1, rec
    lines = [l.strip() for l in open(rec[0], encoding="utf-8") if l.strip() and not l.startswith("epoch")]
    names = [l.split(",")[0] for l in lines]
    model_name = os.path.basename(rec[0]).replace("_records_test.csv", "")
    k = names.index(model_name + ".pkl")
    f = DP + f"{model_name}_results_{k}.csv"
    d = pd.read_csv(f, index_col=0)
    # 结果文件的行名是 DPPDCC 内部的论文节点编号（"tensor(434663)"）→ 换回论文 ID（sample_node_trans.json）
    inv = {v: k for k, v in json.load(open("/path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/sample_node_trans.json"))["paper"].items()}
    d.index = [inv[int(re.sub(r"\D", "", str(i)))] for i in d.index]
    print(f"DPPDCC 测试记录：{names}；采用第 {k} 个（{names[k]}，按验证集选出）→ {os.path.basename(f)}", flush=True)
    return d, names, lines


def main():
    meta = pd.read_parquet(os.path.join(C.TENSORS + "_v6dp", "meta.parquet"))
    y = np.load(os.path.join(C.TENSORS + "_v6dp", "y.npy")).sum(1)
    t = meta.assign(y3=y)[meta.split == "test"][["paper_id", "y3", "topic", "eval_c2000"]]
    t["paper_id"] = t.paper_id.astype(str)
    st = t.set_index("paper_id")
    dd, names, lines = dppdcc_preds()
    dd = dd[dd.time == 2021]
    # 口径核对：DPPDCC 的真实值应与我们的语料内标签一致
    tr = dd.true.reindex(st.index)
    print(f"覆盖 {tr.notna().mean():.4f}；DPPDCC true 与 log1p(我们的标签) 的最大差 {np.nanmax(np.abs(tr - np.log1p(st.y3))):.4g}，"
          f"与原始计数的最大差 {np.nanmax(np.abs(tr - st.y3)):.4g}", flush=True)
    log_scale = np.nanmax(np.abs(tr - np.log1p(st.y3))) < 1e-3
    p_dp = dd.pred if log_scale else np.log1p(dd.pred.clip(lower=0))

    def load(pattern):
        fs = sorted(glob.glob(os.path.join(C.RUNS, pattern, "preds.parquet")))
        ps = []
        for f in fs:
            q = pd.read_parquet(f, columns=["paper_id", "split", "pred"])
            q = q[q.split == "test"]
            ps.append(q.assign(paper_id=q.paper_id.astype(str)).set_index("paper_id").pred)
        return pd.concat(ps, axis=1), len(fs)

    A, na = load("DP_full_s*")
    B, nb = load("DP_2019_s*")
    preds = {"DPPDCC (CIKM'24)": p_dp, f"MPC-Net（2017–2019 训练，{na} 种子集成）": A.mean(1),
             f"MPC-Net（只用 2019 年训练，同 DPPDCC，{nb} 种子集成）": B.mean(1)}
    rows = []
    for m, p in preds.items():
        d = st.assign(pred=p.reindex(st.index)).reset_index()
        assert d.pred.notna().all(), m
        rows.append({"model": m, **EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))})
    for name, M in (("MPC-Net（2017–2019）单种子均值", A), ("MPC-Net（2019）单种子均值", B)):
        sg = []
        for i in range(M.shape[1]):
            d = st.assign(pred=M.iloc[:, i].reindex(st.index)).reset_index()
            sg.append(EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3))))
        rows.append({"model": name, **pd.DataFrame(sg).mean().to_dict()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "10_dppdcc_metrics.csv"), index=False)

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

    bm, br = agg(preds["DPPDCC (CIKM'24)"])
    comp = []
    for m in list(preds)[1:]:
        fm, fr = agg(preds[m])
        dm, dr = fm - bm, fr - br
        comp.append({"比较": f"{m} − DPPDCC", "MALE 差": dm.mean(), "MALE 95% 区间": f"[{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]",
                     "子课题内 Spearman 差": dr.mean(), "Spearman 95% 区间": f"[{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]"})
    cols = ["model", "MALE", "RMSLE", "MAE", "RMSE", "Spearman_子课题内", "NDCG@10_子课题内", "份额L1_子课题内", "Spearman_全体"]
    md = ["# 实验 2 · 与 DPPDCC（CIKM 2024）的比较（DPPDCC 口径：语料内被引）\n",
          f"测试集 = 2021 年论文（{len(st):,} 篇）；标签 = 语料内 Y+1～Y+3 年被引之和。DPPDCC 严格冷启动（2019 训练、2020 验证），"
          f"检查点按验证集选出（测试记录：{names}）。\n",
          res[cols].round(4).to_markdown(index=False), "\n", "## 显著性（按子课题重抽样 1,000 次）\n",
          pd.DataFrame(comp).round(4).to_markdown(index=False), "\n", "DPPDCC 各检查点的测试记录（只作参考，不用于选择）：\n",
          "\n".join("    " + l for l in lines), "\n"]
    open(os.path.join(C.RES_DIR, "实验2结果_DPPDCC比较.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
