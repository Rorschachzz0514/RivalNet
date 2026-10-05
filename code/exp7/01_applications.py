"""
【01】实验 7 · 实际应用（AI 测试集 2021）：子课题内推荐、发表一年后的早期发现、被低估的论文

用途（预先登记见 实验7说明.md）
  7-2 推荐：每个子课题（全数据 K = 2000，≥ 20 篇）内按某个分数取前 10 篇，计算 NDCG@10（增益 = y3）与
      Precision@10（命中"子课题内 y3 前 10%"的比例）。分数：MPC-Net（MPCNet5_l1 五种子平均）、MLP-文本、LightGBM L4、
      NAIP、作者名气（作者截至 T 的最大累计被引）、渠道历史（渠道上一年论文第一年平均 log 被引）、作者人数、随机。
  7-1 早期发现：发表一年后（已知第一年被引 y1），找出"之后两年（y2 + y3a）进入全体前 1%"的论文。排序方法：
      y1；子课题内归一化的 y1（y1 / 子课题 y1 均值，类似领域归一化指标）；MPC-Net 冷启动预测；
      组合（log1p(y1) 与 MPC-Net 预测的标准化得分相加）；实验 1 的"发表 1 年后"LightGBM（L-after）；
      实验 2 的"发表 1 年后"MPC-Net（MPCNet5_after1y，五种子平均；7-1 首次运行后追加）。
      指标：取各方法前 1% 时的召回率（= 精确率），以及前 5%。
  7-3 被低估的论文：第一年被引 ≤ 1 的论文中，按 MPC-Net 预测分十档，比较之后两年被引（y2 + y3a）的中位数与均值。

输入
  /path/to/mpcc/subsets/pred_v2/samples.parquet；/path/to/mpcc/exp2/runs/MPCNet5_l1_s*/preds.parquet；
  /path/to/mpcc/exp1/results/{preds_cold, preds_naip, preds_after1y}.parquet

输出（results/）
  01_recommend.csv、01_early.csv、01_undervalued.csv、实验7结果.md

用法
  python 01_applications.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
E1 = "/path/to/mpcc/exp1/results/"


def ndcg_p10(d, score):
    nd, pk = [], []
    disc = 1 / np.log2(np.arange(2, 12))
    rng = np.random.default_rng(0)
    for _, x in d.groupby("eval_c2000"):
        if len(x) < 20 or x.y3.sum() == 0:
            continue
        sc = x[score].to_numpy() + rng.random(len(x)) * 1e-9
        top = np.argsort(-sc)[:10]
        y = x.y3.to_numpy()
        nd.append((y[top] * disc).sum() / (np.sort(y)[::-1][:10] * disc).sum())
        thr = np.quantile(y, 0.9)
        pk.append((y[top] >= thr).mean())
    return np.mean(nd), np.mean(pk)


def main():
    os.makedirs(RES, exist_ok=True)
    s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet",
                        columns=["paper_id", "split", "y1", "y2", "y3a", "y3", "eval_c2000", "auth_cites_max", "venue_age1_mean", "n_authors"])
    s = s[s.split == "test"].copy()
    mp = pd.concat([pd.read_parquet(f, columns=["paper_id", "split", "pred"]) for f in glob.glob("/path/to/mpcc/exp2/runs/MPCNet5_l1_s*/preds.parquet")])
    s["MPC-Net"] = s.paper_id.map(mp[mp.split == "test"].groupby("paper_id").pred.mean())
    pc = pd.read_parquet(E1 + "preds_cold.parquet")
    for m in ("MLP-文本", "L4 +竞争"):
        s[m] = s.paper_id.map(pc[(pc.model == m) & (pc.split == "test")].set_index("paper_id").pred)
    nf = E1 + "preds_naip.parquet"
    if os.path.exists(nf):
        q = pd.read_parquet(nf)
        s["NAIP (AAAI'25)"] = s.paper_id.map(q[q.split == "test"].set_index("paper_id").pred)
    s["作者名气"] = s.auth_cites_max.fillna(0)
    s["渠道历史"] = s.venue_age1_mean.fillna(s.venue_age1_mean.median())
    s["作者人数"] = s.n_authors
    s["随机"] = np.random.default_rng(1).random(len(s))
    scores = [c for c in ["MPC-Net", "MLP-文本", "L4 +竞争", "NAIP (AAAI'25)", "作者名气", "渠道历史", "作者人数", "随机"] if c in s]
    rec = pd.DataFrame([{"排序依据": c, **dict(zip(["NDCG@10", "P@10（子课题前 10%）"], ndcg_p10(s, c)))} for c in scores])
    rec.to_csv(os.path.join(RES, "01_recommend.csv"), index=False)

    # 7-1 早期发现
    s["future"] = s.y2 + s.y3a
    top1 = s.future >= s.future.quantile(0.99)
    top5 = s.future >= s.future.quantile(0.95)
    s["y1_norm"] = s.y1 / s.groupby("eval_c2000").y1.transform("mean").replace(0, np.nan)
    z = lambda v: (v - v.mean()) / v.std()
    s["组合"] = z(np.log1p(s.y1)) + z(s["MPC-Net"])
    pa = pd.read_parquet(E1 + "preds_after1y.parquet")
    s["L-after"] = s.paper_id.map(pa[(pa.model.str.startswith("L-after")) & (pa.split == "test")].set_index("paper_id").pred)
    ma = pd.concat([pd.read_parquet(f, columns=["paper_id", "split", "pred"]) for f in glob.glob("/path/to/mpcc/exp2/runs/MPCNet5_after1y_s*/preds.parquet")])
    s["MPC-after"] = s.paper_id.map(ma[ma.split == "test"].groupby("paper_id").pred.mean())
    early = []
    for c, lab in (("y1", "第一年被引 y1"), ("y1_norm", "子课题内归一化 y1"), ("MPC-Net", "MPC-Net 冷启动预测"),
                   ("组合", "y1 + MPC-Net 组合"), ("L-after", "LightGBM 发表 1 年后模型"), ("MPC-after", "MPC-Net 发表 1 年后模型")):
        v = s[c].fillna(-1e9) + np.random.default_rng(2).random(len(s)) * 1e-9
        r1 = (v >= np.quantile(v, 0.99)) & top1
        r5 = (v >= np.quantile(v, 0.95)) & top5
        early.append({"方法": lab, "前 1% 召回": r1.sum() / top1.sum(), "前 5% 召回": r5.sum() / top5.sum()})
    early = pd.DataFrame(early)
    early.to_csv(os.path.join(RES, "01_early.csv"), index=False)

    # 7-3 被低估
    low = s[s.y1 <= 1].copy()
    low["预测十档"] = pd.qcut(low["MPC-Net"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    und = low.groupby("预测十档", observed=True).future.agg(["size", "median", "mean", lambda x: (x >= 10).mean()]).rename(
        columns={"<lambda_0>": "之后两年 ≥10 次的比例"})
    und.to_csv(os.path.join(RES, "01_undervalued.csv"))
    md = ["# 实验 7 结果：实际应用（AI 测试集 2021）\n",
          "## 7-2 子课题内推荐（每个子课题取前 10 篇）\n", rec.round(4).to_markdown(index=False), "\n",
          "## 7-1 发表一年后的早期发现（目标：之后两年被引进入全体前 1% / 5%）\n", early.round(4).to_markdown(index=False), "\n",
          f"## 7-3 被低估的论文（第一年被引 ≤ 1，共 {len(low):,} 篇；按 MPC-Net 冷启动预测分十档）\n", und.round(3).to_markdown(), "\n"]
    open(os.path.join(RES, "实验7结果.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
