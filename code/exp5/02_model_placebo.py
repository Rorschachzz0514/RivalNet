"""
【02】实验 5-3 · 模型内假对手：把 MPC-Net 测试集的竞争集合换成别的论文的，模型表现是否下降（预先登记见 实验5说明.md 第 2 节）

用途
  预测由实验 2 的 `02_mpcnet.py --placebo <模式>` 生成（模型参数不变，只换测试集的竞争集合）：
    shuffle_cell  换成同一格子（训练期子课题 × 年份）里另一篇论文的竞争集合
    shuffle_all   换成测试集任意另一篇论文的竞争集合
  与真实竞争集合下的预测比较 MALE 与子课题内 Spearman；按子课题整组重抽样 1,000 次，每次取 5 个种子指标的平均，
  得到差异的 95% 区间。
  通过标准：shuffle_cell 的子课题内 Spearman 显著低于真实对手。

输入
  /path/to/mpcc/exp2/runs/MPCNet5_l1_s*/preds.parquet 与 preds_placebo_*.parquet；样本表

输出（results/）
  02_placebo.csv、02_placebo.md

用法
  python 02_model_placebo.py
"""
import glob
import os
import time
from importlib import util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = "/path/to/mpcc/exp2/runs"
spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)


def main():
    t0 = time.time()
    s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
    s = s[s.split == "test"]
    sub_ids = np.sort(s.eval_c2000.unique())
    rng = np.random.default_rng(2026)
    W = rng.multinomial(len(sub_ids), np.full(len(sub_ids), 1 / len(sub_ids)), size=1000).astype(float)
    rows, boot = [], {}
    for mode in ("真实对手", "shuffle_cell", "shuffle_all"):
        bm, br = [], []
        for d in sorted(glob.glob(os.path.join(RUNS, "MPCNet5_l1_s*"))):
            f = os.path.join(d, "preds.parquet" if mode == "真实对手" else f"preds_placebo_{mode}.parquet")
            p = pd.read_parquet(f)
            p = p[p.split == "test"][["paper_id", "pred"]].merge(s, on="paper_id")
            p = p.assign(y=p.y3, ly=np.log1p(p.y3))
            rows.append({"mode": mode, "seed": os.path.basename(d), **EV.metrics(p)})
            g = p.assign(ae=(p.pred - p.ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "size")).reindex(sub_ids).fillna(0)
            rs = EV.group_rho(p, "eval_c2000").set_index("g")
            rn = (rs.rho * rs.n).reindex(sub_ids).fillna(0).to_numpy()
            nr = rs.n.reindex(sub_ids).fillna(0).to_numpy()
            bm.append((W @ g.se.to_numpy()) / (W @ g.n.to_numpy()))
            br.append((W @ rn) / (W @ nr))
        boot[mode] = (np.mean(bm, 0), np.mean(br, 0))
    met = pd.DataFrame(rows)
    summ = met.groupby("mode")[["MALE", "Spearman_子课题内", "NDCG@10_子课题内", "份额L1_子课题内"]].agg(["mean", "std"]).round(4)
    tests = []
    for mode in ("shuffle_cell", "shuffle_all"):
        dm = boot[mode][0] - boot["真实对手"][0]
        dr = boot["真实对手"][1] - boot[mode][1]
        tests.append({"mode": mode, "MALE 变差（假 − 真）": dm.mean(), "MALE_CI": f"[{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]",
                      "Spearman 下降（真 − 假）": dr.mean(), "Spearman_CI": f"[{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]",
                      "显著下降": bool(np.quantile(dr, .025) > 0)})
    tests = pd.DataFrame(tests)
    met.to_csv(os.path.join(RES, "02_placebo.csv"), index=False)
    passed = bool(tests.set_index("mode").loc["shuffle_cell", "显著下降"])
    md = ["# 实验 5-3 模型内假对手\n",
          "定版模型 MPCNet5_l1（5 个种子的已训练权重），只替换测试集的竞争集合。\n",
          "## 指标（5 个种子均值 ± 标准差）\n", summ.to_markdown(), "\n",
          "## 与真实对手的差异（按子课题重抽样）\n", tests.round(4).to_markdown(index=False), "\n",
          f"**判定**：换成同格子其他论文的对手后，子课题内 Spearman {'显著下降' if passed else '没有显著下降'}；**{'通过' if passed else '未通过'}**。\n"]
    open(os.path.join(RES, "02_placebo.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"02 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
