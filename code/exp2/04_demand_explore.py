"""
【04】实验 2 · 需求塔（第一阶段）的方法比较（只用训练格子拟合、验证格子比较；测试集不参与）

用途
  迭代 v2 发现：用真实格子总量时 MPC-Net 的验证 MALE = 0.711（优于全部对比方法），瓶颈在格子总量的预测
  （端到端训练的需求塔在验证格子上 log 误差 ≈ 0.54）。设计文档 5.4 节本来规定需求塔"先单独训练，再固定"。
  这里比较几种第一阶段方法，目标 = log(格子第 t 年总被引 + 1)，t = 1, 2, 3：
    R0  log(n_cell) + 训练期全部格子"每篇平均 log 总量"的常数
    R1  岭回归（格子特征，标准化）
    R2  LightGBM（格子特征）
    R3  log(n_cell) + LightGBM 预测"每篇平均被引的 log"（把供给的作用固定为线性，模型只学人均水平）
    R4  R3 的岭回归版本
  指标：验证格子上 |预测 − 真实| 的平均（按格内论文数加权，与论文层面 MALE 的权重一致）。

输入
  config_exp2.TENSORS（cell_x、cell_tot、cell_split、cell_ptr）

输出（results/）
  04_demand_explore.csv、04_demand_explore.md

用法
  python 04_demand_explore.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

L = lambda n: np.load(os.path.join(C.TENSORS, n))
X, T, sp, ptr = L("cell_x.npy"), L("cell_tot.npy"), L("cell_split.npy"), L("cell_ptr.npy")
n = np.diff(ptr).astype(float)
ln = np.log(n)
tr, va = sp == "train", sp == "val"
Yt = np.log(T + 1)
rows = []
for t in range(3):
    y = Yt[:, t]
    rate = y - ln
    preds = {
        "R0 log(n) + 常数": ln + np.average(rate[tr], weights=n[tr]),
        "R1 岭回归": RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr], sample_weight=n[tr]).predict(X),
    }
    m = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=64, seed=1),
                  lgb.Dataset(X[tr], y[tr], weight=n[tr]), num_boost_round=600)
    preds["R2 LightGBM"] = m.predict(X)
    m3 = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
                        bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=64, seed=1),
                   lgb.Dataset(X[tr], rate[tr], weight=n[tr]), num_boost_round=600)
    preds["R3 log(n) + LightGBM 人均"] = ln + m3.predict(X)
    preds["R4 log(n) + 岭回归 人均"] = ln + RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], rate[tr], sample_weight=n[tr]).predict(X)
    for k, p in preds.items():
        rows.append({"year": t + 1, "method": k, "train_err": np.average(np.abs(p - y)[tr], weights=n[tr]),
                     "val_err": np.average(np.abs(p - y)[va], weights=n[va]), "val_bias": np.average((p - y)[va], weights=n[va])})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(C.RES_DIR, "04_demand_explore.csv"), index=False)
piv = res.pivot_table(index="method", columns="year", values=["val_err", "val_bias"]).round(3)
md = ["# 需求塔第一阶段方法比较（验证格子，按论文数加权）\n", piv.to_markdown(), "\n"]
open(os.path.join(C.RES_DIR, "04_demand_explore.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
