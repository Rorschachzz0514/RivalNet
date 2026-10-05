"""
【03】实验 2 / 3 · 评价：MPC-Net、消融与实验 1 的全部对比方法在同一测试集上比较，P1–P4 判定（预先登记见 实验2说明.md 第 4 节）

用途
  指标与实验 1 完全相同（直接调用实验 1【04】的指标函数）。MPC-Net 与各消融：
    每个种子单独计算指标 → 报告 5 个种子的均值 ± 标准差（判定用）；另报告 5 个种子预测取平均后的"集成"结果。
  显著性：按子课题（K = 2000）整组重抽样 1,000 次；每次重抽样中，MPC-Net 的指标 = 5 个种子指标的平均，
  与对比方法的指标相减，得到差异的 95% 区间（比只用集成预测更保守）。
  判定：
    P1  MPC-Net 的 MALE 低于最好的非上帝视角对比方法（按测试集 MALE 选），差异区间不含 0
    P2  MPC-Net 的子课题内 Spearman 高于最好的非上帝视角对比方法（按测试集该指标选），差异区间不含 0
    P3  A1_direct 的 MALE 显著高于 MPC-Net
    P4  A3_no_rivals 的子课题内 Spearman 显著低于 MPC-Net

输入
  runs/<配置>_s<种子>/preds.parquet（【02】）；实验 1 的 results/preds_cold.parquet、preds_naip.parquet；样本表

输出（results/）
  03_metrics.csv（全部模型 × 种子 × 切分）、03_summary.csv（均值 ± 标准差、集成）、03_tests.csv、实验2结果.md

用法
  python 03_evaluate.py --version v5      （v4 = 预先登记的显式分解结构，v5 = 按验证集定版的结构）
"""
import glob
import os
import sys
import time
from importlib import util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)

VERSIONS = {
    # v4：预先登记的结构（log 空间显式分解）及消融
    "v4": dict(main="MPCNet", direct="A1_direct", no_rivals="A3_no_rivals",
               configs=["MPCNet", "A1_direct", "A2_oracle_level", "A3_no_rivals", "A7_no_share_loss", "A11_meanpool",
                        "A13_count_space", "A17_coupled", "A17_top10", "A18_no_text"], out="实验2结果_v4.md"),
    # v5：按验证集定版的结构（直接输出 + 格子需求上下文 + 份额辅助损失）及消融；λ 在 MPCNet5 / MPCNet5_l1 中按验证集选
    "v5": dict(main=["MPCNet5", "MPCNet5_l1"], direct=None, decomp="B_decomp", no_rivals="B_no_rivals",
               configs=["MPCNet5", "MPCNet5_l1", "B_decomp", "B_no_cell", "B_no_rivals", "B_no_share_loss", "B_meanpool",
                        "B_coupled", "B_top10", "B_no_text"], out="实验2结果.md"),
    # 实验 6 跨学科：只训练定版结构与去掉对手的消融（超参数沿用 AI，不调参）
    # 发表 1 年后：目标 = 第 2、3 年被引之和；对比方法 = 实验 1 的 D2 与 L-after
    "after1y": dict(main="MPCNet5_after1y", direct=None, decomp=None, no_rivals="B_no_rivals_after1y", after1y=True,
                    configs=["MPCNet5_after1y", "B_no_rivals_after1y"], out="实验2结果_发表1年后.md"),
    "field": dict(main="MPCNet5_l1", direct=None, decomp=None, no_rivals="B_no_rivals",
                  configs=["MPCNet5_l1", "B_no_rivals"], out="实验2结果_跨学科.md"),
}
ORACLE = ["D1*-方向（上帝视角）", "D1*-子课题（上帝视角）", "D1*-方向（上帝视角，含自身）", "D1*-子课题（上帝视角，含自身）", "A2_oracle_level"]
B = 1000


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v5")
    ap_version = ap.parse_args().version
    V = VERSIONS[ap_version]
    FINAL = V["configs"]
    t0 = time.time()
    os.makedirs(C.RES_DIR, exist_ok=True)
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y2", "y3a", "y3", "topic", "eval_c2000"])
    if V.get("after1y"):
        s["y3"] = s.y2 + s.y3a                                          # 目标换成第 2、3 年被引之和
    base = pd.read_parquet(os.path.join(C.EXP1_RES, "preds_after1y.parquet" if V.get("after1y") else "preds_cold.parquet")).assign(seed=0)
    npth = os.path.join(C.EXP1_RES, "preds_naip.parquet")
    if os.path.exists(npth) and not V.get("after1y"):
        base = pd.concat([base, pd.read_parquet(npth).assign(seed=0)], ignore_index=True)
    else:
        print("警告：没有 NAIP 预测")
    runs = []
    for cfg in FINAL:
        for f in sorted(glob.glob(os.path.join(C.RUNS, f"{cfg}_s*", "preds.parquet"))):
            r = pd.read_parquet(f, columns=["paper_id", "split", "seed", "pred"])
            runs.append(r.assign(model=cfg))
    runs = pd.concat(runs, ignore_index=True)
    ens = runs.groupby(["model", "split", "paper_id"], as_index=False).pred.mean().assign(seed=-1)
    ens["model"] = ens.model + "（5 种子集成）"
    allp = pd.concat([base, runs, ens], ignore_index=True)

    rows, agg = [], {}
    sub_ids = np.sort(s.eval_c2000.unique())
    for (m, sd, sp), d in allp.groupby(["model", "seed", "split"], sort=False):
        d = d.merge(s, on=["paper_id", "split"])
        d = d.assign(y=d.y3, ly=np.log1p(d.y3))
        rows.append({"model": m, "seed": sd, "split": sp, "n": len(d), **EV.metrics(d)})
        if sp == "test":
            g = d.assign(ae=(d.pred - d.ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "size"))
            rs = EV.group_rho(d, "eval_c2000").set_index("g")
            g["rn"] = (rs.rho * rs.n).reindex(g.index).fillna(0)
            g["nr"] = rs.n.reindex(g.index).fillna(0)
            agg[(m, sd)] = g.reindex(sub_ids).fillna(0)
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(C.RES_DIR, f"03_metrics_{ap_version}.csv"), index=False)
    print(f"指标完成（{time.time() - t0:.0f}s）", flush=True)

    cols = ["MALE", "RMSLE", "MAE", "RMSE", "Spearman_方向内", "Spearman_子课题内", "NDCG@10_子课题内", "份额L1_子课题内", "Spearman_全体"]
    summ = met.groupby(["model", "split"])[cols].agg(["mean", "std"])
    summ.to_csv(os.path.join(C.RES_DIR, f"03_summary_{ap_version}.csv"))

    rng = np.random.default_rng(2026)
    W = rng.multinomial(len(sub_ids), np.full(len(sub_ids), 1 / len(sub_ids)), size=B).astype(np.float64)

    def boot(model):
        keys = [k for k in agg if k[0] == model]
        bm = np.mean([(W @ agg[k].se.to_numpy()) / (W @ agg[k].n.to_numpy()) for k in keys], axis=0)
        br = np.mean([(W @ agg[k].rn.to_numpy()) / (W @ agg[k].nr.to_numpy()) for k in keys], axis=0)
        return bm, br

    t = met[met.split == "test"].groupby("model")[cols].mean()
    vv = met[met.split == "val"].groupby("model")[cols].mean()
    main_cfg = V["main"] if isinstance(V["main"], str) else vv.loc[V["main"], "MALE"].idxmin()   # 按验证集选
    print(f"主模型（验证集选择）：{main_cfg}", flush=True)
    baselines = [m for m in base.model.unique() if m not in ORACLE]
    best_male = t.loc[baselines, "MALE"].idxmin()
    best_rho = t.loc[baselines, "Spearman_子课题内"].idxmax()
    bm_full, br_full = boot(main_cfg)
    tests = []

    def add(name, diff, better_if_negative, label):
        lo, hi = np.quantile(diff, [0.025, 0.975])
        ok = (hi < 0) if better_if_negative else (lo > 0)
        tests.append({"test": name, "comparison": label, "diff_mean": diff.mean(), "ci_lo": lo, "ci_hi": hi, "passed": bool(ok)})

    bmb, _ = boot(best_male)
    add("P1", bm_full - bmb, True, f"MALE：MPC-Net − {best_male}")
    _, brb = boot(best_rho)
    add("P2", br_full - brb, False, f"子课题内 Spearman：MPC-Net − {best_rho}")
    if V.get("direct"):
        bma, _ = boot(V["direct"])
        add("P3", bma - bm_full, False, f"MALE：{V['direct']} − {main_cfg}（应 > 0：显式分解有用）")
    elif V.get("decomp"):
        bmd, _ = boot(V["decomp"])
        add("P3（v5 口径）", bm_full - bmd, False, f"MALE：{main_cfg} − {V['decomp']}（> 0 表示显式分解更好）")
    _, brn = boot(V["no_rivals"])
    add("P4", br_full - brn, False, f"子课题内 Spearman：{main_cfg} − {V['no_rivals']}（应 > 0）")
    for ab in [c for c in FINAL if c != main_cfg]:
        bma, bra = boot(ab)
        add("消融", bma - bm_full, False, f"MALE：{ab} − {main_cfg}")
        add("消融", br_full - bra, False, f"子课题内 Spearman：{main_cfg} − {ab}")
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(C.RES_DIR, f"03_tests_{ap_version}.csv"), index=False)

    def fmt(model, sp="test"):
        r = summ.loc[(model, sp)]
        if pd.isna(r[("MALE", "std")]):
            return {c: f"{r[(c, 'mean')]:.4f}" for c in cols}
        return {c: f"{r[(c, 'mean')]:.4f} ± {r[(c, 'std')]:.4f}" for c in cols}

    order = [m for m in base.model.unique()] + FINAL + [f"{m}（5 种子集成）" for m in FINAL]
    tab = pd.DataFrame({m: fmt(m) for m in order if (m, "test") in summ.index}).T
    tab_val = pd.DataFrame({m: fmt(m, "val") for m in order if (m, "val") in summ.index}).T[["MALE", "Spearman_子课题内"]]
    pk = tests[tests.test.str.startswith("P")]
    md = ["# 实验 2 结果：MPC-Net 主预测（含实验 3 消融）\n",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}。测试集 = 2021 年焦点论文；对比方法来自实验 1（同一测试集）。"
          "MPC-Net 与消融为 5 个随机种子的均值 ± 标准差；\"集成\" = 5 个种子预测取平均。设计与标准见 `实验2说明.md`。\n",
          "## 1. 判定\n", pk.round(4).to_markdown(index=False), "\n",
          "## 2. 测试集全部指标\n", tab.to_markdown(), "\n",
          "## 3. 消融：与完整模型的差异（正数 = 完整模型更好）\n", tests[tests.test == "消融"].round(4).to_markdown(index=False), "\n",
          "## 4. 验证集（模型选择依据）\n", tab_val.to_markdown(), "\n"]
    md.insert(1, f"版本 {ap_version}；主模型 = **{main_cfg}**（按验证集 MALE 选择）。\n")
    open(os.path.join(C.RES_DIR, V["out"]), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:6]))
    print(f"03 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
