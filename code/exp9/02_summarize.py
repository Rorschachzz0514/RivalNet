"""
【02】实验 9 · 汇总效率测量：MPC-Net 各训练规模的耗时与显存；实验 1 各模型训练耗时（由日志时间戳推算）；NAIP 推理吞吐

输入
  /path/to/mpcc/exp2/runs/MPCNet5_l1_eff*_s1/summary.json；/path/to/mpcc/logs/exp1_02.log；/path/to/mpcc/logs/exp1_03d.log
输出
  /path/to/mpcc/exp9/results/实验9结果.md、01_efficiency.csv
"""
import glob
import json
import os
import re

import pandas as pd

RES = "/path/to/mpcc/exp9/results"
os.makedirs(RES, exist_ok=True)
rows = []
for f in sorted(glob.glob("/path/to/mpcc/exp2/runs/MPCNet5_l1_eff*_s1/summary.json")):
    d = json.load(open(f))
    rows.append({"训练格子比例": float(re.search(r"eff([0-9.]+)_s1", f).group(1)), "训练论文数": d["n_train_papers"],
                 "每轮秒数": round(d["sec_per_epoch"], 2), "显存峰值 GB": d["peak_mem_gb"], "参数量": d["n_params"],
                 "验证集预测秒数": d["predict_sec"]["val"], "测试集预测秒数": d["predict_sec"]["test"]})
eff = pd.DataFrame(rows).sort_values("训练格子比例")
eff["每秒训练论文数"] = (eff["训练论文数"] / eff["每轮秒数"]).round(0)
eff.to_csv(os.path.join(RES, "01_efficiency.csv"), index=False)
# 实验 1 各模型耗时（日志时间戳之差）
t = []
log = open("/path/to/mpcc/logs/exp1_02.log", encoding="utf-8").read().splitlines()
prev = None
for line in log:
    m = re.match(r"(\d\d):(\d\d):(\d\d) (.*)", line)
    if m:
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if prev is not None and ("完成" in m.group(4) or "MLP" in m.group(4)):
            t.append({"步骤": m.group(4)[:40], "耗时秒": sec - prev})
        prev = sec
naip = [l for l in open("/path/to/mpcc/logs/exp1_03d.log", encoding="utf-8", errors="ignore") if "] DONE" in l]
md = ["# 实验 9 结果：效率与可扩展性\n",
      "## MPC-Net（定版，单张 RTX 4090；每个规模训练 5 轮）\n", eff.to_markdown(index=False), "\n",
      "## 对比：实验 1 各模型在全量训练集上的耗时（128 线程 CPU / MLP 用 1 张 GPU）\n", pd.DataFrame(t).to_markdown(index=False), "\n",
      "## NAIP（LLaMA-3-8B，8 比特）推理\n", "".join(f"- {l.strip()}\n" for l in naip),
      "\n（每张 GPU 约 22,500 篇；验证 + 测试共 157,966 篇，7 张 GPU 并行约 10 分钟）\n"]
open(os.path.join(RES, "实验9结果.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
