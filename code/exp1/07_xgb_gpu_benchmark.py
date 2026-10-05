"""
【07】GPU 版梯度提升树测试：XGBoost（CUDA）vs LightGBM（CPU），同一数据与特征

用途
  服务器 CPU 长期超载（负载 400+ / 192 线程）而 GPU 相对空闲，测试能否把梯度提升树基线挪到 GPU 上。
  数据与特征与 02 / 05 完全相同（L4 = 元数据 + 热度 + SPECTER2 PCA-64 + 竞争计数；目标 log1p(y3)；
  训练 2017–2019、验证 2020 早停、测试 2021）。
    LightGBM（CPU）：05 选定的参数（num_leaves 511，学习率 0.02，min_data_in_leaf 500），单种子，记录耗时。
    XGBoost（GPU）：对齐的参数——hist、lossguide、max_leaves 511、eta 0.02、min_child_weight 500（平方损失下 = 叶子最少样本数）、
                    subsample / colsample 0.8、lambda 1、max_bin 255、类别特征原生处理，验证集早停 200 轮、最多 5000 轮。
  比较：耗时、验证 / 测试 MALE、子课题内 Spearman、NDCG@10。

输出（results/）
  07_xgb_gpu_benchmark.md

用法
  CUDA_VISIBLE_DEVICES=6 python 07_xgb_gpu_benchmark.py
"""
import os
import sys
import time
from importlib import util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_exp1 as C

spec = util.spec_from_file_location("m02", os.path.join(HERE, "02_models.py"))
M = util.module_from_spec(spec)
spec.loader.exec_module(M)
spec = util.spec_from_file_location("ev", os.path.join(HERE, "04_evaluate.py"))
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)


def evaluate(s, pred, name, sec):
    rows = []
    for sp in ("val", "test"):
        d = s[s.split == sp].assign(pred=pred[s.split == sp])
        m = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
        rows.append({"方法": name, "集合": sp, "耗时（秒）": round(sec), "MALE": m["MALE"], "ρ_sub": m["Spearman_子课题内"],
                     "NDCG@10": m["NDCG@10_子课题内"]})
    return rows


def main():
    import xgboost as xgb
    s, X = M.prepare(False)
    PCS = [f"pc{i}" for i in range(C.N_PCA)]
    L4 = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PCS + C.COMP
    rows = []

    tr, va = s[s.split == "train"], s[s.split == "val"]
    t0 = time.time()
    dtr = xgb.DMatrix(tr[L4], tr.ly3, enable_categorical=True, nthread=16)
    dva = xgb.DMatrix(va[L4], va.ly3, enable_categorical=True, nthread=16)
    params = dict(tree_method="hist", device="cuda", grow_policy="lossguide", max_leaves=511, max_depth=0, eta=0.02,
                  min_child_weight=500, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=255, objective="reg:squarederror",
                  eval_metric="rmse", seed=C.SEED)
    bst = xgb.train(params, dtr, num_boost_round=5000, evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=False)
    dall = xgb.DMatrix(s[L4], enable_categorical=True, nthread=16)
    pred_xgb = bst.predict(dall, iteration_range=(0, bst.best_iteration + 1))
    sec_xgb = time.time() - t0
    M.log(f"XGBoost（GPU）：{bst.best_iteration + 1} 轮，{sec_xgb:.0f} 秒")
    rows += evaluate(s, pred_xgb, "XGBoost（GPU）", sec_xgb)

    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(C.RES_DIR, "07_xgb_partial.csv"), index=False)   # GPU 结果先存一份
    print(r.round(4).to_string(), flush=True)
    t0 = time.time()
    p_lgb, it = M.fit_lgb(s, L4, "ly3", "lgb", dict(num_leaves=511, learning_rate=0.02, min_data_in_leaf=500))
    sec_lgb = time.time() - t0
    pl = p_lgb.set_index(["paper_id", "split"]).pred
    pred_lgb = np.full(len(s), np.nan)
    ev = s.split.isin(["val", "test"]).to_numpy()
    pred_lgb[ev] = pl.reindex(pd.MultiIndex.from_arrays([s.paper_id[ev], s.split[ev]])).to_numpy()
    M.log(f"LightGBM（CPU）：{it} 轮，{sec_lgb:.0f} 秒")
    rows += evaluate(s, pred_lgb, f"LightGBM（CPU，{C.THREADS} 线程）", sec_lgb)

    r = pd.DataFrame(rows)
    md = ["# GPU 版梯度提升树测试（XGBoost CUDA vs LightGBM CPU）\n",
          f"测试时服务器负载：{os.getloadavg()[0]:.0f}（192 线程）。LightGBM {it} 轮；XGBoost {bst.best_iteration + 1} 轮。\n",
          r.round(4).to_markdown(index=False)]
    open(os.path.join(C.RES_DIR, "07_xgb_gpu_benchmark.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
