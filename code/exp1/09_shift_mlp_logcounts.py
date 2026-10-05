"""
【09】修改计划 A4 补充：前移设定下 MLP 的计数特征改为 log1p 后重训（排除语料起点造成的特征漂移）

背景（08 诊断）：语料从 2016 年开始，2017 年论文的"前一年相似论文数"（n_cand_ym1）等计数被低估；前移设定只用 2017–2018 训练时，
  2019 / 2020 年论文的 n_cand_ym1 相对训练分布的 z 值达 +30，MLP（原始计数按训练集标准化）线性外推，验证年系统性高估 +0.25。
  RivalNet 的论文特征不含这类原始计数（计数取 log1p），不受影响；LightGBM 对单调变换不变，无需重跑。
做法：与 06 相同的数据、前移切分与选定超参数（宽度 512、2 层、dropout 0.3、学习率 3e-4、5 个种子），
  只把非负、长尾（训练集 99 分位 > 20）的数值特征取 log1p 后再标准化。
输出：results/preds_shift_mlp_log.parquet（model = "MLP-文本（调参，计数取对数）"）

用法
  python 09_shift_mlp_logcounts.py --gpu 0 [--main]
"""
import argparse
import os
import sys
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
    ap.add_argument("--main", action="store_true", help="主设定（2017–2019 训练、2020 验证、2021 测试）下的同一变体，检查主表的 MLP 是否也应改用")
    args = ap.parse_args()
    if not args.main:
        s = pd.read_parquet(C.SAMPLES)
        s["split"] = np.where(s.Y <= 2018, "train", np.where(s.Y == 2019, "val", np.where(s.Y == 2020, "test", "none")))
        s = s[s.split != "none"].reset_index(drop=True)
        s.to_parquet("/tmp/_shift_samples_log.parquet")
        C.SAMPLES = "/tmp/_shift_samples_log.parquet"
    s, X = M.prepare(False)
    PCS = [f"pc{i}" for i in range(C.N_PCA)]
    L4 = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PCS + C.COMP
    tr = (s.split == "train").to_numpy()
    logged = []
    for c in L4:
        if c in C.CATEG or c.startswith("pc"):
            continue
        v = s[c].astype(float)
        if v[tr].min() >= 0 and v[tr].quantile(0.99) > 20:
            s[c] = np.log1p(v)
            logged.append(c)
    M.log(f"取 log1p 的计数特征（{len(logged)} 个）：{logged}")
    ps = [T5.fit_mlp(s, X, L4, args.gpu, 512, 2, 0.3, 3e-4, C.SEED + k)[0].set_index(["paper_id", "split"]).pred for k in range(5)]
    po = pd.concat(ps, axis=1).mean(1).rename("pred").reset_index().assign(model="MLP-文本（调参，计数取对数）")
    po.to_parquet(os.path.join(C.RES_DIR, "preds_main_mlp_log.parquet" if args.main else "preds_shift_mlp_log.parquet"), index=False)
    ly = s.set_index("paper_id").ly3
    for sp in ("val", "test"):
        p = po[po.split == sp].set_index("paper_id").pred
        e = p - ly.reindex(p.index)
        M.log(f"{sp}：MALE {e.abs().mean():.4f}，偏差 {e.mean():+.4f}，去偏 MALE {(e - e.mean()).abs().mean():.4f}")
    M.log("09 DONE")


if __name__ == "__main__":
    main()
