"""
【03】检验 0-2（微观）与 0-3（稳健）：撞题对手多的论文，之后被引是否更少

用途
  一行 = 一篇焦点论文（391,439 篇）。固定效应 = 方向 × 首次公开年份（同方向同年份的论文互相比较），
  标准误按方向聚类（77 个方向）。
  0-2 主设定：
    Poisson  y3 ~ β·log1p(prior_s2_95) + 控制变量 | 方向×年份
    OLS      log1p(y3) ~ 同上
    prior_s2_95 = 发表前 365 天内、SPECTER2 相似度 ≥ 0.95 的样本论文数（全量比较，不截断）
    控制变量：log1p(作者数)、log1p(机构数)、log(参考文献数)、开放获取、有摘要、有基金、有预印本、
             log1p(引用了多少篇发表前的相似论文)
  0-3 稳健性：阈值 0.90–0.98；只算同方向的对手；S3 文献耦合（全部 / 只限互不引用）；S2 与 S3 同时放入；
             因变量换成样本内被引；加入时间窗内全部论文数（控制一般拥挤程度）
  诊断（事后追加，不计入通过判定）：主设定的 β 为正时，用来区分"子方向热度"与"先后顺序"
    D1  控制发表后 365 天内的相似论文数 log1p(after_95)（子方向热度的代理）
    D2  同 D1，只限同方向
    D3  y3 ~ 发表前占比 before/(before+after) + log1p(before+after)：邻域大小固定时，越晚进入（前面对手占比越高）
        是否被引越少；只用 before+after ≥ 1 的论文
  通过标准：0-2 主设定 Poisson 与 OLS 的 β 都为负且显著（p < 0.05）；0-3 各设定方向一致。
  解读：Poisson 的 β 是弹性——对手数（加 1 后）翻倍，被引变化约 2^β − 1。

输入
  results/focal.parquet（【01】）
  DATA_DIR/sim_counts.parquet（诊断用：n_ge0.95_after、n_ge0.95_after_same）

输出（results/）
  03_micro.csv   每个设定的 β、标准误、p 值、样本量
  03_micro.md    结果表与解读

用法
  python 03_test_micro.py           全量
  python 03_test_micro.py --test    只用 2019 年（冒烟测试）

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

CONTROLS = ["l_auth", "l_inst", "l_refs", "oa", "abs_", "fund", "pre", "l_cited"]


def prepare(test=False):
    df = pd.read_parquet(os.path.join(C.RES_DIR, "focal.parquet"))
    if test:
        df = df[df.Y == 2019]
    df = df.assign(
        l_auth=np.log1p(df.n_authors), l_inst=np.log1p(df.n_institutions.fillna(0)), l_refs=np.log(df.n_refs),
        oa=df.any_oa.astype(float), abs_=df.has_abstract.astype(float), fund=df.has_funding.astype(float),
        pre=df.has_preprint.astype(float), l_cited=np.log1p(df.n_prior_cited),
        ly3=np.log1p(df.y3), ty=(df.topic.astype(np.int64) * 10000 + df.Y).astype(np.int64),
        l_cand=np.log1p(df.n_cand_before),
    )
    for t in C.TAU_GRID:
        k = int(round(t * 100))
        df[f"x_s2_{k}"] = np.log1p(df[f"prior_s2_{k}"])
        df[f"x_s2_{k}_same"] = np.log1p(df[f"prior_s2_{k}_same"])
    k = int(round(C.TAU * 100))
    aft = pd.read_parquet(os.path.join(C.DATA_DIR, "sim_counts.parquet"),
                          columns=["focal_id", f"n_ge{C.TAU:.2f}_after", f"n_ge{C.TAU:.2f}_after_same"])
    aft.columns = ["paper_id", "after", "after_same"]
    n0 = len(df)
    df = df.merge(aft, on="paper_id", how="left")
    assert len(df) == n0 and df.after.notna().all()
    df["l_after"] = np.log1p(df.after)
    df["l_after_same"] = np.log1p(df.after_same)
    nb = df[f"prior_s2_{k}"] + df.after
    df["l_nbhd"] = np.log1p(nb)
    df["share_before"] = np.where(nb > 0, df[f"prior_s2_{k}"] / nb.where(nb > 0, 1), np.nan)
    df["x_s3"] = np.log1p(df.prior_s3)
    df["x_s3_strict"] = np.log1p(df.prior_s3_strict)
    return df


def run(df, name, x, y="y3", model="pois", extra=None, group="0-3"):
    rhs = " + ".join([x] + CONTROLS + (extra or []))
    fml = f"{y} ~ {rhs} | ty"
    fit = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": "topic"})
    t = fit.tidy().loc[x]
    return {"group": group, "spec": name, "model": "Poisson" if model == "pois" else "OLS", "y": y, "x": x,
            "beta": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": fit._N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df = prepare(args.test)
    k = int(round(C.TAU * 100))
    rows = [
        run(df, f"主设定 S2 τ={C.TAU}", f"x_s2_{k}", group="0-2"),
        run(df, f"主设定 S2 τ={C.TAU}", f"x_s2_{k}", y="ly3", model="ols", group="0-2"),
    ]
    for t in C.TAU_GRID:
        kk = int(round(t * 100))
        if kk != k:
            rows.append(run(df, f"S2 阈值 τ={t}", f"x_s2_{kk}"))
    rows += [
        run(df, f"S2 τ={C.TAU} 只限同方向", f"x_s2_{k}_same"),
        run(df, "S3 文献耦合", "x_s3"),
        run(df, "S3 文献耦合（只限互不引用）", "x_s3_strict"),
        run(df, "S2 与 S3 同时放入：S2", f"x_s2_{k}", extra=["x_s3"]),
        run(df, "S2 与 S3 同时放入：S3", "x_s3", extra=[f"x_s2_{k}"]),
        run(df, f"S2 τ={C.TAU}，因变量 = 样本内被引", f"x_s2_{k}", y="y3_in_corpus"),
        run(df, f"S2 τ={C.TAU}，控制时间窗内全部论文数", f"x_s2_{k}", extra=["l_cand"]),
        run(df, "S3，OLS", "x_s3", y="ly3", model="ols"),
    ]
    rows += [
        run(df, f"D1 控制发表后相似论文数", f"x_s2_{k}", extra=["l_after"], group="诊断"),
        run(df, f"D1 控制发表后相似论文数（OLS）", f"x_s2_{k}", y="ly3", model="ols", extra=["l_after"], group="诊断"),
        run(df, f"D2 同方向版本", f"x_s2_{k}_same", extra=["l_after_same"], group="诊断"),
        run(df[df.share_before.notna()], "D3 发表前占比（邻域大小固定）", "share_before", extra=["l_nbhd"], group="诊断"),
        run(df[df.share_before.notna()], "D3 发表前占比（OLS）", "share_before", y="ly3", model="ols", extra=["l_nbhd"], group="诊断"),
    ]
    res = pd.DataFrame(rows)
    res["effect_if_doubled_%"] = np.where(res.model == "Poisson", (2 ** res.beta - 1) * 100, np.nan)
    res.to_csv(os.path.join(C.RES_DIR, "03_micro.csv"), index=False)

    main_p, main_o = res.iloc[0], res.iloc[1]
    passed_02 = main_p.beta < 0 and main_p.p < 0.05 and main_o.beta < 0 and main_o.p < 0.05
    rob = res[res.group == "0-3"]
    consistent = int((rob.beta < 0).sum())
    sig_neg = int(((rob.beta < 0) & (rob.p < 0.05)).sum())
    show = res[["group", "spec", "model", "y", "beta", "se", "p", "effect_if_doubled_%", "n"]].copy()
    show[["beta", "se"]] = show[["beta", "se"]].round(4)
    show["p"] = show.p.map(lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}")
    show["effect_if_doubled_%"] = show["effect_if_doubled_%"].round(2)
    md = ["# 检验 0-2（微观）与 0-3（稳健）\n",
          "固定效应：方向 × 首次公开年份；标准误按方向聚类；控制变量见脚本说明。"
          "`effect_if_doubled_%` = 对手数（加 1 后）翻倍时被引的变化（Poisson）。\n",
          show.to_markdown(index=False), "\n",
          f"**0-2**：Poisson β = {main_p.beta:.4f}（p = {main_p.p:.2e}），OLS β = {main_o.beta:.4f}（p = {main_o.p:.2e}）；"
          f"**{'通过' if passed_02 else '未通过'}**（标准：两者都为负且显著）。\n",
          f"**0-3**：{len(rob)} 个稳健性设定中 {consistent} 个为负，{sig_neg} 个负且显著；"
          f"**{'通过' if consistent == len(rob) else '部分不一致'}**（标准：方向一致）。\n"]
    open(os.path.join(C.RES_DIR, "03_micro.md" if not args.test else "03_micro_test.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"03 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
