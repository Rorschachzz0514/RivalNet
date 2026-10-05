"""
【12】实验 2 · 对比方法总表：近年 A / B 类会议与期刊的引用预测方法，在同一测试集（2021 年论文）、两种被引口径上与 MPC-Net 比较

口径
  all = 全部来源被引（主表；samples.parquet 的 y3）；dp = 语料内被引（DPPDCC 口径；tensors_v6dp/y.npy）。
方法（每个方法的预测都是 log(1 + 三年被引)；只在验证集上选模型 / 早停，测试集只评价）
  MPC-Net 终版（A + B 集成）/ A 组集成 / 只用 2019 年训练（MAIN_2019，与 DPPDCC、H2CGL 的训练年份相同）
  MPC-Net 语料内口径：DP_full、DP_2019
  MLP-文本、LightGBM L4（实验 1）、NAIP（AAAI'25，实验 1）
  PLM-FT（SPECTER2-base 微调，【11】）
  DPPDCC（CIKM'24）：third_party/DPPDCC/results/<数据源>/ 按验证集选出的全局最优检查点（多个种子 = 多个数据源目录）
  HINTS（WWW'21）、H2CGL（IP&M'23）：baselines/<方法>/preds_<口径>.parquet（子代理移植，见各自 README）
  多种子的方法：报告"种子集成"与"单种子均值"。缺少预测文件的方法自动跳过。
显著性
  MPC-Net 与每个对比方法的差异，按子课题整组重抽样 1,000 次（同【08】【10】）。

输出（results/）
  12_baselines_<口径>.csv、实验2结果_对比方法总表.md

用法
  python 12_baseline_table.py
"""
import glob
import json
import os
import re
import sys
from importlib import util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)
DPP = "/path/to/mpcc/third_party/DPPDCC/"
BL = "/path/to/mpcc/baselines/"


def runs(pattern, split="test"):
    fs = sorted(glob.glob(os.path.join(C.RUNS, pattern, "preds.parquet")))
    out = []
    for f in fs:
        d = pd.read_parquet(f, columns=["paper_id", "split", "pred"])
        out.append(d[d.split == split].set_index("paper_id").pred)
    return out


def dppdcc(source):
    """按验证集选出的全局最优检查点（<模型名>.pkl）的测试结果；行名为内部节点编号，换回论文 ID；只取 2021 年论文"""
    rec = glob.glob(DPP + f"results/{source}/DPPDCC*_records_test.csv")
    if len(rec) != 1:
        return None
    lines = [l.strip() for l in open(rec[0], encoding="utf-8") if l.strip() and not l.startswith("epoch")]
    names = [l.split(",")[0] for l in lines]
    mname = os.path.basename(rec[0]).replace("_records_test.csv", "")
    if mname + ".pkl" not in names:
        return None
    k = names.index(mname + ".pkl")
    f = DPP + f"results/{source}/{mname}_results_{k}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, index_col=0)
    inv = {v: k_ for k_, v in json.load(open(DPP + "data/mpcc_ai/sample_node_trans.json"))["paper"].items()}
    d.index = [int(inv[int(re.sub(r"\D", "", str(i)))]) for i in d.index]
    d = d[d.time == 2021]
    return np.log1p(d.pred.clip(lower=0)), d.true


def files(path, split="test"):
    """baselines/<方法>/preds_*.parquet：paper_id, split, seed, pred"""
    fs = sorted(glob.glob(path))
    out = []
    for f in fs:
        d = pd.read_parquet(f)
        d = d[d.split == split]
        if "seed" in d.columns and d.seed.nunique() > 1:
            out += [g.set_index("paper_id").pred for _, g in d.groupby("seed")]
        else:
            out.append(d.set_index("paper_id").pred)
    return out


def main():
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
    st_all = s[s.split == "test"].set_index("paper_id")
    meta = pd.read_parquet(os.path.join(C.TENSORS + "_v6dp", "meta.parquet"), columns=["paper_id"])
    ydp = pd.Series(np.load(os.path.join(C.TENSORS + "_v6dp", "y.npy")).sum(1), index=meta.paper_id.to_numpy())
    st_dp = st_all.assign(y3=ydp.reindex(st_all.index).to_numpy())
    pc = pd.read_parquet(os.path.join(C.EXP1_RES, "preds_cold.parquet"))
    md = ["# 实验 2 · 对比方法总表（2021 年论文，测试集只评价一次）\n",
          "所有方法只在验证集（2020）上选模型 / 早停。多种子方法报告种子集成与单种子均值；差异 = 对比方法 − MPC-Net 参照，按子课题重抽样 1,000 次。\n"]
    for target, st in (("all", st_all), ("dp", st_dp)):
        M = {}                                             # 方法 → 各种子预测列表
        if target == "all":
            A, B = runs("AS_r06_1_s*"), runs("FINAL_B*_s*")
            if A and B:
                M["MPC-Net 终版（A + B 集成）"] = [(pd.concat(A, axis=1).mean(1) + pd.concat(B, axis=1).mean(1)) / 2]
                M["MPC-Net A 组（2017–2019，10 种子）"] = A
            m19 = runs("MAIN_2019_s*")
            if m19:
                M["MPC-Net（只用 2019 年训练）"] = m19
            for m, lab in (("MLP-文本", "MLP-文本"), ("L4 +竞争", "LightGBM（L4）")):
                M[lab] = [pc[(pc.model == m) & (pc.split == "test")].set_index("paper_id").pred]
            tf = os.path.join(C.EXP1_RES, "preds_tuned.parquet")      # 修改计划 A5：验证集调参后的 MLP / LightGBM
            if os.path.exists(tf):
                pt = pd.read_parquet(tf)
                for m, lab in (("MLP-文本（调参）", "MLP-文本（调参）"), ("L4 +竞争（调参）", "LightGBM（L4，调参）")):
                    M[lab] = [pt[(pt.model == m) & (pt.split == "test")].set_index("paper_id").pred]
            tl = os.path.join(C.EXP1_RES, "preds_main_mlp_log.parquet")  # 计数特征取 log1p 的调参 MLP（exp1【09】--main）
            if os.path.exists(tl):
                pl = pd.read_parquet(tl)
                M["MLP-文本（调参，计数取对数）"] = [pl[pl.split == "test"].set_index("paper_id").pred]
            nf = os.path.join(C.EXP1_RES, "preds_naip.parquet")
            if os.path.exists(nf):
                q = pd.read_parquet(nf)
                M["NAIP（AAAI'25）"] = [q[q.split == "test"].set_index("paper_id").pred]
            srcs = ["mpcc_ai_all", "mpcc_ai_all_s2", "mpcc_ai_all_s3"]
        else:
            for pat, lab in (("DP_full_s*", "MPC-Net（2017–2019 训练）"), ("DP_2019_s*", "MPC-Net（只用 2019 年训练）")):
                r = runs(pat)
                if r:
                    M[lab] = r
            srcs = ["mpcc_ai", "mpcc_ai_s2", "mpcc_ai_s3"]
        dp = [x for x in (dppdcc(z) for z in srcs) if x is not None]
        if dp:
            tr = dp[0][1].reindex(st.index)
            assert np.nanmax(np.abs(tr - st.y3)) < 1e-6, "DPPDCC 的真实值与我们的标签不一致"
            M["DPPDCC（CIKM'24）"] = [x[0] for x in dp]
        for name, lab in (("plm_ft", "PLM-FT（SPECTER2 微调）"), ("hints", "HINTS（WWW'21）"), ("h2cgl", "H2CGL（IP&M'23）"),
                          ("knn", "kNN（检索）"), ("knn_mlp", "检索增强 MLP")):   # 修改计划 A1
            # 只匹配本学科（AI）的文件：preds_<口径>.parquet 或 preds_<口径>_s<种子>.parquet；其他学科的文件带 _sf 标签
            r = files(BL + f"{name}/preds_{target}.parquet") + files(BL + f"{name}/preds_{target}_s[0-9]*.parquet")
            if r:
                M[lab] = r
        rows, allp = [], {}
        for lab, ps in M.items():
            ens = pd.concat(ps, axis=1).mean(1)
            allp[lab] = ens
            d = st.assign(pred=ens.reindex(st.index)).reset_index()
            miss = int(d.pred.isna().sum())
            d = d.dropna(subset=["pred"])
            met = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
            row = {"方法": lab, "种子数": len(ps), "缺失": miss, **{k: met[k] for k in ("MALE", "RMSLE", "Spearman_子课题内", "NDCG@10_子课题内", "Spearman_全体")}}
            if len(ps) > 1:
                sg = []
                for p in ps:
                    d1 = st.assign(pred=p.reindex(st.index)).reset_index().dropna(subset=["pred"])
                    sg.append(EV.metrics(d1.assign(y=d1.y3, ly=np.log1p(d1.y3))))
                sg = pd.DataFrame(sg)
                row.update({"单种子 MALE": sg.MALE.mean(), "单种子 MALE 标准差": sg.MALE.std(), "单种子 Spearman": sg["Spearman_子课题内"].mean()})
            rows.append(row)
        res = pd.DataFrame(rows)
        # 各方法测试集（种子集成）预测，供数据集发布与复现显著性检验
        pd.concat([v.rename("pred").rename_axis("paper_id").reset_index().assign(method=k) for k, v in allp.items()])[
            ["paper_id", "method", "pred"]].to_parquet(os.path.join(C.RES_DIR, f"12_test_preds_{target}.parquet"), index=False)
        res.to_csv(os.path.join(C.RES_DIR, f"12_baselines_{target}.csv"), index=False)
        # 显著性：MPC-Net 参照 vs 每个对比方法
        ref = "MPC-Net 终版（A + B 集成）" if target == "all" else "MPC-Net（2017–2019 训练）"
        sub = np.sort(st.eval_c2000.unique())
        rng = np.random.default_rng(2026)
        W = rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=1000).astype(float)
        ly = np.log1p(st.y3)

        def agg(p):
            d = st.assign(pred=p.reindex(st.index))
            g = d.assign(ae=(d.pred - ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "count")).reindex(sub).fillna(0)
            rs = EV.group_rho(d.dropna(subset=["pred"]).reset_index().assign(y=lambda x: x.y3), "eval_c2000").set_index("g")
            rn = (rs.rho * rs.n).reindex(sub).fillna(0).to_numpy()
            nr = rs.n.reindex(sub).fillna(0).to_numpy()
            return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy()), (W @ rn) / (W @ nr)

        comp = []
        if ref in M:
            fm, fr = agg(pd.concat(M[ref], axis=1).mean(1))
            for lab, ps in M.items():
                if lab == ref or lab.startswith("MPC-Net"):
                    continue
                bm, br = agg(pd.concat(ps, axis=1).mean(1))
                dm, dr = bm - fm, br - fr
                comp.append({"对比方法 − MPC-Net": lab, "MALE 差（>0 = MPC-Net 更好）": dm.mean(),
                             "95% 区间": f"[{np.quantile(dm, .025):+.4f}, {np.quantile(dm, .975):+.4f}]",
                             "子课题内 Spearman 差（<0 = MPC-Net 更好）": dr.mean(),
                             "95% 区间 ": f"[{np.quantile(dr, .025):+.4f}, {np.quantile(dr, .975):+.4f}]"})
        title = "全部来源被引（主表口径）" if target == "all" else "语料内被引（DPPDCC 口径）"
        md += [f"\n## {title}（{len(st):,} 篇）\n", res.round(4).to_markdown(index=False), "\n",
               f"显著性（参照：{ref}）：\n", pd.DataFrame(comp).round(4).to_markdown(index=False) if comp else "（无）", "\n"]
        print("\n".join(md[-5:]), flush=True)
    open(os.path.join(C.RES_DIR, "实验2结果_对比方法总表.md"), "w", encoding="utf-8").write("\n".join(md))


if __name__ == "__main__":
    main()
