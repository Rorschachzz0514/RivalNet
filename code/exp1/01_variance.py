"""
【01】实验 1 · log1p(y3) 的方差分解：方向之间 / 子课题之间 / 组内

用途
  把论文 3 年被引（log1p）的总方差拆成"组间"（不同组的平均水平不同）与"组内"（同一组内论文之间）两部分，
  组 = 年份、方向 × 年份、子课题(K=2000) × 年份、子课题(K=5000) × 年份、发表渠道 × 年份。
  组间占比 = 只知道"属于哪个组"的完美模型（组均值）能解释的方差比例（R²），即绝对误差指标里"热度"能拿到的部分上界。
  在全部样本与每个年份（每个切分）上分别计算。

输入
  config_exp1.SAMPLES

输出（results/）
  01_variance.csv、01_variance.md

用法
  python 01_variance.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp1 as C


def between_share(y, g):
    m = y.groupby(g).transform("mean")
    return float(((m - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    os.makedirs(C.RES_DIR, exist_ok=True)
    s = pd.read_parquet(C.SAMPLES, columns=["Y", "split", "topic", "eval_c2000", "eval_c5000", "venue_final", "y3", "y1"])
    s["ly3"] = np.log1p(s.y3)
    rows = []
    for name, sub in [("全部 2017–2021", s)] + [(f"{y}", s[s.Y == y]) for y in sorted(s.Y.unique())]:
        y = sub.ly3
        r = {"sample": name, "n": len(sub), "var_total": float(y.var()),
             "年份": between_share(y, sub.Y.astype(str)),
             "方向×年份": between_share(y, sub.topic.astype(str) + "_" + sub.Y.astype(str)),
             "子课题(2000)×年份": between_share(y, sub.eval_c2000.astype(str) + "_" + sub.Y.astype(str)),
             "子课题(5000)×年份": between_share(y, sub.eval_c5000.astype(str) + "_" + sub.Y.astype(str)),
             "渠道类别×年份": between_share(y, sub.venue_final + "_" + sub.Y.astype(str)),
             "方向×子课题(2000)×年份": between_share(y, sub.topic.astype(str) + "_" + sub.eval_c2000.astype(str) + "_" + sub.Y.astype(str))}
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(C.RES_DIR, "01_variance.csv"), index=False)
    show = res.copy()
    for c in show.columns[3:]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    md = ["# 实验 1 · 01 方差分解（log1p 3 年被引）\n",
          "数值 = 组间方差占总方差的比例，即\"只知道论文属于哪个组\"的完美模型能解释的部分（R²）。"
          "子课题组越细，组间占比会机械地变大（组内样本少），解读时要结合组的个数。\n",
          show.round(4).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, "01_variance.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
