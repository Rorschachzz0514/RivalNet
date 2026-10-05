"""
【19】修改计划 A4 · 第二个测试年份（时间整体前移一年：2017–2018 训练、2019 验证、2020 测试）

用途
  RivalNet：SIM_A（2017–2018 训练、2019 早停，5 种子）与 SIM_B*（2017–2019 按 A 的轮数重训，5 种子），终版 = 两组平均（与主设定相同）；
  对比：exp1【06】在同一前移设定下重训的 MLP、LightGBM（超参数用主设定验证集选定的组合）。
  所有方法的模型都没有用 2020 年论文训练；但主设定中所有方法的超参数都是在 2020 年（主设定的验证年）上选的，
  因此这是不同方法之间公平的稳健性检验，而不是完全干净的测试年份（论文中需说明）。
  差异按子课题整组重抽样 1,000 次。

输出
  results/19_second_test_year.md、results/19_second_test_year.csv

用法
  python 19_second_test_year.py
"""
import glob
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


def ens(pattern):
    fs = sorted(glob.glob(os.path.join(RUNS, pattern, "preds.parquet")))
    ps = [pd.read_parquet(f).query("split == 'test'").set_index("paper_id").pred for f in fs]
    return pd.concat(ps, axis=1).mean(1), len(fs)


def main():
    s = pd.read_parquet(SAMPLES, columns=["paper_id", "Y", "y3", "topic", "eval_c2000"]).set_index("paper_id")
    st = s[s.Y == 2020]
    A, na = ens("SIM_A_s*")
    B, nb = ens("SIM_B*_s*")
    assert na == 5 and nb == 5, (na, nb)
    M = {"RivalNet (A + B)": (A + B) / 2, "RivalNet A only": A}
    ps = pd.read_parquet("/path/to/mpcc/exp1/results/preds_shift.parquet").query("split == 'test'")
    for m, lab in (("MLP-文本（调参）", "MLP (tuned)"), ("L4 +竞争（调参）", "LightGBM (tuned)")):
        M[lab] = ps[ps.model == m].set_index("paper_id").pred
    # exp1【09】：计数特征取 log1p 的 MLP（排除语料起点造成的 n_cand_ym1 漂移，见 exp1【08】诊断）
    pl = pd.read_parquet("/path/to/mpcc/exp1/results/preds_shift_mlp_log.parquet").query("split == 'test'")
    M["MLP (tuned, log counts)"] = pl.set_index("paper_id").pred
    rows = []
    for lab, p in M.items():
        d = st.assign(pred=p.reindex(st.index)).reset_index()
        assert d.pred.notna().all(), lab
        m = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
        rows.append({"method": lab, "n": len(d), **{k: m[k] for k in ("MALE", "RMSLE", "Spearman_子课题内", "NDCG@10_子课题内", "Spearman_全体")}})
    res = pd.DataFrame(rows)
    # 与最强对比方法的差异（子课题重抽样）
    sub = np.sort(st.eval_c2000.unique())
    rng = np.random.default_rng(2026)
    W = rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=1000).astype(float)
    ly = np.log1p(st.y3)

    def agg(p):
        d = st.assign(pred=p.reindex(st.index))
        g = d.assign(ae=(d.pred - ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "count")).reindex(sub).fillna(0)
        rs = EV.group_rho(d.reset_index().assign(y=lambda x: x.y3), "eval_c2000").set_index("g")
        rn, nr = (rs.rho * rs.n).reindex(sub).fillna(0).to_numpy(), rs.n.reindex(sub).fillna(0).to_numpy()
        return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy()), (W @ rn) / (W @ nr)

    base = res[res.method.isin(["MLP (tuned)", "LightGBM (tuned)", "MLP (tuned, log counts)"])]
    bm = base.sort_values("MALE").method.iloc[0]
    br = base.sort_values("Spearman_子课题内", ascending=False).method.iloc[0]
    fm, fr = agg(M["RivalNet (A + B)"])
    dm = fm - agg(M[bm])[0]
    dr = fr - agg(M[br])[1]
    md = ["# 修改计划 A4 · 第二个测试年份（2020 年论文；2017–2018 训练、2019 验证）\n",
          "注：所有方法的超参数都是在主设定的验证年（2020）上选的，模型本身都没有用 2020 年论文训练；这是方法之间公平的稳健性检验，而非完全干净的测试年份。\n",
          res.round(4).to_markdown(index=False), "",
          f"RivalNet − 最强对比方法（MALE：{bm}）ΔMALE = {dm.mean():+.4f} [{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]",
          f"RivalNet − 最强对比方法（ρ_sub：{br}）Δρ_sub = {dr.mean():+.4f} [{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]"]
    res.to_csv(os.path.join(HERE, "results", "19_second_test_year.csv"), index=False)
    open(os.path.join(HERE, "results", "19_second_test_year.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
