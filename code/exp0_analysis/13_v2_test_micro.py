"""
【13】实验 0 v2 · 检验 0-2 v2（微观）、0-3 v2（稳健）、0-7 v2（异质性）（预先登记见 实验0说明.md 7.4）

用途
  一行 = 一篇焦点论文（主分析集：期刊 / 会议 / 论文集 / arXiv）。
  0-2 v2 主设定：
    Poisson  y3 ~ β·log1p(P1) + 控制变量 | 子课题(K=2000)×年份 + 发表月份 + 发表渠道
    OLS      log1p(y3) ~ 同上
    P1 = 前一个日历年首次公开、SPECTER2 相似度 ≥ 0.95 的样本论文数；标准误按子课题聚类。
  0-3 v2 稳健：阈值 0.90–0.98；K = 1000 / 5000；对手窗口 Y−2～Y−1；日期精确子样本用"发表前 365 天"计数；
              焦点集合换成全部 / 只期刊 + 会议；因变量换成样本内被引；S3 文献耦合（前一年）。通过标准：全部 β < 0。
  0-7 v2 异质性（描述）：按发表渠道、有无预印本、日期精度、子课题需求增长三分位、年份分组估计主设定。
  另有事后诊断（不计入判定，单独列出）：加入当年 / 后一年的相似论文数（热度代理，可能是坏控制）。

输入
  results_v2/focal.parquet（【11】）

输出（results_v2/）
  13_micro.csv、13_micro.md

用法
  python 13_v2_test_micro.py [--test]      --test 只用 2019 年

运行记录
  （见 实验0说明.md 第 6 节）
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp0 as C

warnings.filterwarnings("ignore")
CONTROLS = ["l_auth", "l_inst", "l_refs", "oa", "abs_", "fund", "pre", "l_cited"]
T = C.V2_TAU


def prepare(test=False):
    df = pd.read_parquet(os.path.join(C.V2_RES, "focal.parquet"))
    if test:
        df = df[df.Y == 2019]
    df = df.assign(
        l_auth=np.log1p(df.n_authors), l_inst=np.log1p(df.n_institutions.fillna(0)), l_refs=np.log(df.n_refs),
        oa=df.any_oa.astype(float), abs_=df.has_abstract.astype(float), fund=df.has_funding.astype(float),
        pre=df.has_preprint.astype(float), l_cited=np.log1p(df.n_cited_prior), ly3=np.log1p(df.y3),
    )
    for k in C.V2_KS:
        df[f"cy{k}"] = (df[f"c{k}"].astype(np.int64) * 10000 + df.Y).astype(np.int64)
    for t in C.V2_TAUS:
        for d in ("m2", "m1", "0", "p1"):
            df[f"x_{d}_{t}"] = np.log1p(df[f"n_y{d}_ge{t}"])
        df[f"x_pb_{t}"] = np.log1p(df[f"n_pb365_ge{t}"])
    df["x_m12"] = np.log1p(df[f"n_ym1_ge{T}"] + df[f"n_ym2_ge{T}"])
    df["x_s3"] = np.log1p(df.s3_m1)
    df["x_s3s"] = np.log1p(df.s3_m1_strict)
    df["l_pa"] = np.log1p(df[f"n_pa365_ge{T}"])
    df["l_cand_pb"] = np.log1p(df.n_cand_pb365)
    df["venue"] = df.venue_class.astype("category")
    return df


def run(df, name, x, group, y="y3", model="pois", extra=None, k=None, fe_extra="pub_month + venue"):
    k = k or C.V2_K
    rhs = " + ".join([x] + CONTROLS + (extra or []))
    fml = f"{y} ~ {rhs} | cy{k} + {fe_extra}"
    fit = (pf.fepois if model == "pois" else pf.feols)(fml, data=df, vcov={"CRV1": f"c{k}"})
    t = fit.tidy().loc[x]
    return {"group": group, "spec": name, "model": "Poisson" if model == "pois" else "OLS", "y": y, "x": x, "K": k,
            "beta": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": fit._N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df = prepare(args.test)
    m = df[df.in_main]
    X = f"x_m1_{T}"
    rows = [run(m, "主设定", X, "0-2"), run(m, "主设定", X, "0-2", y="ly3", model="ols")]
    # 0-3 稳健
    for t in C.V2_TAUS:
        if t != T:
            rows.append(run(m, f"阈值 {t}", f"x_m1_{t}", "0-3"))
    for k in C.V2_KS:
        if k != C.V2_K:
            rows.append(run(m, f"子课题 K = {k}", X, "0-3", k=k))
    rows += [
        run(m, "对手窗口 Y−2～Y−1", "x_m12", "0-3"),
        run(m[m[f"n_pb365_ge{T}"].notna()], "日期精确子样本：发表前 365 天", f"x_pb_{T}", "0-3", extra=["l_cand_pb"]),
        run(df, "焦点集合 = 全部 v2 焦点", X, "0-3"),
        run(df[df.in_jc], "焦点集合 = 只期刊 + 会议", X, "0-3"),
        run(m, "因变量 = 样本内被引", X, "0-3", y="y3_in_corpus"),
        run(m, "S3 文献耦合（前一年）", "x_s3", "0-3"),
        run(m, "S3 文献耦合（前一年，互不引用）", "x_s3s", "0-3"),
        run(m, "OLS，K = 5000", X, "0-3", y="ly3", model="ols", k=5000),
    ]
    # 事后诊断（不计入判定）
    rows += [
        run(m, "事后：加当年相似论文数", X, "事后诊断", extra=[f"x_0_{T}"]),
        run(m, "事后：加当年 + 后一年相似论文数", X, "事后诊断", extra=[f"x_0_{T}", f"x_p1_{T}"]),
        run(m[m[f"n_pb365_ge{T}"].notna()], "事后：日期精确，发表前 365 天 + 控制发表后 365 天", f"x_pb_{T}", "事后诊断",
            extra=["l_cand_pb", "l_pa"]),
    ]
    # 0-7 异质性
    grp = {
        "发表渠道": m.venue_class,
        "有无预印本": m.has_preprint.map({True: "有预印本", False: "无预印本"}),
        "日期精度": m.date_precision,
        "子课题需求增长": pd.qcut(m.growth_c2000.rank(method="first"), 3, labels=["低增长", "中增长", "高增长"]),
        "年份": m.Y.astype(str),
    }
    for gname, g in grp.items():
        for lv in sorted(g.dropna().unique(), key=str):
            sub = m[g == lv]
            if len(sub) < 2000:
                continue
            try:
                r = run(sub, f"{gname}：{lv}", X, "0-7")
                rows.append(r)
            except Exception as e:  # 小组无法估计时跳过并记录
                print(f"跳过 {gname}={lv}: {e}")
    res = pd.DataFrame(rows)
    res["effect_if_doubled_%"] = np.where(res.model == "Poisson", (2 ** res.beta - 1) * 100, np.nan)
    sfx = "_test" if args.test else ""
    res.to_csv(os.path.join(C.V2_RES, f"13_micro{sfx}.csv"), index=False)

    p, o = res.iloc[0], res.iloc[1]
    passed2 = p.beta < 0 and p.p < 0.05 and o.beta < 0 and o.p < 0.05
    rob = res[res.group == "0-3"]
    passed3 = bool((rob.beta < 0).all())
    show = res.copy()
    show[["beta", "se"]] = show[["beta", "se"]].round(4)
    show["p"] = show.p.map(lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}")
    show["effect_if_doubled_%"] = show["effect_if_doubled_%"].round(2)
    md = ["# 检验 0-2 v2（微观）、0-3 v2（稳健）、0-7 v2（异质性）\n",
          f"固定效应：子课题(K)×年份 + 发表月份 + 发表渠道；标准误按子课题聚类。主竞争变量 = log1p(前一年相似度 ≥ {T} 的论文数)。"
          "`effect_if_doubled_%` = 对手数（加 1 后）翻倍时被引的变化。\n",
          show[["group", "spec", "model", "y", "K", "beta", "se", "p", "effect_if_doubled_%", "n"]].to_markdown(index=False), "\n",
          f"**0-2 v2**：Poisson β = {p.beta:+.4f}（p = {p.p:.2g}），OLS β = {o.beta:+.4f}（p = {o.p:.2g}）；**{'通过' if passed2 else '未通过'}**。\n",
          f"**0-3 v2**：{len(rob)} 个设定中 {int((rob.beta < 0).sum())} 个为负（{int(((rob.beta < 0) & (rob.p < 0.05)).sum())} 个显著），"
          f"{int(((rob.beta > 0) & (rob.p < 0.05)).sum())} 个显著为正；**{'通过' if passed3 else '未通过'}**（标准：全部为负）。\n"]
    open(os.path.join(C.V2_RES, f"13_micro{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"13 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
