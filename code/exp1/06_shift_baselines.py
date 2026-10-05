"""
【06】修改计划 A4 · 第二个测试年份：时间整体前移一年（2017–2018 训练、2019 验证 / 早停、2020 测试）

用途
  RivalNet 已有前移设定下的运行（02_mpcnet.py 的 sim_shift 配置，用于事先确定 A + B 集成方案）。
  本脚本在同一前移设定下重训 MLP 与 LightGBM（超参数用【05】在主设定验证集上选定的组合，不在 2020 年上重新选择），
  输出 2020 年论文的预测，供 exp2/19_second_test_year.py 统一评价。
  PCA 与标准化只在前移后的训练集（2017–2018）上拟合。

输出（results/）
  preds_shift.parquet   paper_id, split（val = 2019 / test = 2020）, model, pred

用法
  python 06_shift_baselines.py --gpu 4
"""
import argparse
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
spec5 = util.spec_from_file_location("m05", os.path.join(HERE, "05_tune_baselines.py"))
T5 = util.module_from_spec(spec5)
spec5.loader.exec_module(T5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    t0 = time.time()
    tun = pd.read_csv(os.path.join(C.RES_DIR, "05_tuning.csv"))
    lg = tun[(tun.model == "LightGBM") & (tun.text == "pca64")].sort_values("val_MALE").iloc[0]
    use256 = tun[(tun.model == "LightGBM") & (tun.text == "pca256")].val_MALE.min() < lg.val_MALE
    ml = tun[tun.model == "MLP"].sort_values("val_MALE").iloc[0]
    M.log(f"LightGBM：{lg.to_dict()}，PCA-256={use256}；MLP：{ml.to_dict()}")
    s = pd.read_parquet(C.SAMPLES)
    s["split"] = np.where(s.Y <= 2018, "train", np.where(s.Y == 2019, "val", np.where(s.Y == 2020, "test", "none")))
    s = s[s.split != "none"].reset_index(drop=True)
    s.to_parquet("/tmp/_shift_samples.parquet")
    C.SAMPLES = "/tmp/_shift_samples.parquet"                 # prepare() 读这个文件；PCA 只在新的训练集上拟合
    n_pca = 256 if use256 else C.N_PCA
    C.N_PCA = n_pca
    s, X = M.prepare(False)
    PCS = [f"pc{i}" for i in range(n_pca)]
    L4 = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PCS + C.COMP
    out = []
    # 先跑 MLP（GPU，几分钟），结果先存一份
    PC64 = PCS[:64] if n_pca >= 64 else PCS
    L4m = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PC64 + C.COMP     # MLP 与主设定一致：完整向量 + L4（含 PCA-64）
    ps = [T5.fit_mlp(s, X, L4m, args.gpu, int(ml.width), int(ml.depth), float(ml.dropout), float(ml.lr), C.SEED + k)[0]
          .set_index(["paper_id", "split"]).pred for k in range(5)]
    out.append(pd.concat(ps, axis=1).mean(1).rename("pred").reset_index().assign(model="MLP-文本（调参）"))
    out[0].to_parquet(os.path.join(C.RES_DIR, "preds_shift_mlp.parquet"), index=False)
    M.log("MLP 完成")
    # LightGBM 3 个种子并行（每个 THREADS // 3 线程）：服务器超载时，单进程 128 线程依次跑 3 个种子要 2 小时以上
    from joblib import Parallel, delayed
    p = dict(num_leaves=int(lg.num_leaves), learning_rate=float(lg.learning_rate), min_data_in_leaf=int(lg.min_data_in_leaf),
             num_threads=C.THREADS // 3)
    res = Parallel(n_jobs=3, backend="loky")(
        delayed(M.fit_lgb)(s, L4, "ly3", "lgb", dict(p, seed=C.SEED + k, bagging_seed=C.SEED + k, feature_fraction_seed=C.SEED + k))
        for k in range(3))
    ps = [r[0].set_index(["paper_id", "split"]).pred for r in res]
    out.append(pd.concat(ps, axis=1).mean(1).rename("pred").reset_index().assign(model="L4 +竞争（调参）"))
    M.log("LightGBM 完成")
    po = pd.concat(out, ignore_index=True)[["paper_id", "split", "model", "pred"]]
    po.to_parquet(os.path.join(C.RES_DIR, "preds_shift.parquet"), index=False)
    M.log(f"06 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
