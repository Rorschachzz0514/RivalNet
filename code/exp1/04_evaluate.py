"""
【04】实验 1 · 评价：绝对误差 vs 组内排序，自助法置信区间，排序颠倒，C1-a～d 判定（预先登记见 实验1说明.md 第 4 节）

用途
  对 02、03 的全部模型，在测试集（2021，主）与验证集（2020）上计算：
    绝对误差  MALE、RMSLE（log1p 空间）；MAE、RMSE（原始计数，预测 = expm1(pred) 截断到 ≥ 0）
    组内排序  方向内 / 子课题内 Spearman（每组 ≥ 10 篇，按组大小加权平均；组内预测完全相同时记 0）；
              子课题内 NDCG@10（每组 ≥ 20 篇，增益 = y3，预测并列时用固定种子随机打破）；全体 Spearman
    份额误差  子课题内预测份额与真实份额的 L1 距离（每组 ≥ 10 篇且 Σy > 0）
  置信区间：按子课题（全数据 K = 2000）整组有放回重抽样 1,000 次；所有模型用同一组重抽样，模型间差异也有区间。
  判定：
    C1-a  D1*-子课题（留一均值）的 MALE ≤ 最好学习模型的 MALE
    C1-b  D1-方向+子课题的 MALE ≤ 1.10 × 最好学习模型的 MALE
    C1-c  存在 A、B：A 的 MALE 显著更低且子课题内 Spearman 显著更低（两项差异的 95% 区间都不含 0）
    C1-d  按 MALE 与按子课题内 Spearman 的模型排名的 Kendall τ < 0.5
  学习模型 = L1–L4、MLP-文本、NAIP。"含自身"的上帝视角版本只作对照，不参与判定与排名。
  "发表 1 年后"设定只报告指标表。
  事后分析（看到 02 的结果后追加，不计入判定）：把每个模型的均方对数误差 MSE = RMSLE² 精确拆成三部分——
    整体偏差² （所有论文的平均误差）+ 组间部分（各子课题平均误差偏离整体偏差的加权方差）+ 组内部分（同一子课题内误差的方差），
  看模型之间的差距来自"整体水平 / 子课题热度猜得准不准"还是"组内分得清不清"。

输入
  config_exp1.SAMPLES；results/preds_cold.parquet、preds_naip.parquet（若有）、preds_after1y.parquet

输出（results/）
  04_metrics.csv、04_metrics_after1y.csv、04_boot_ci.csv、04_reversals.csv、实验1结果.md

用法
  python 04_evaluate.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp1 as C

LEARNED = ["L1 元数据", "L2 元数据+热度", "L3 +文本", "L4 +竞争", "MLP-文本", "NAIP (AAAI'25)"]
B = 1000


def group_rho(d, key, min_n=10):
    """每组 Spearman（预测全同记 0），返回 DataFrame[g, n, rho]。"""
    out = []
    for g, x in d.groupby(key, sort=False):
        if len(x) < min_n:
            continue
        if x.pred.nunique() <= 1 or x.y.nunique() <= 1:
            r = 0.0
        else:
            r = spearmanr(x.pred, x.y).statistic
        out.append((g, len(x), 0.0 if np.isnan(r) else r))
    return pd.DataFrame(out, columns=["g", "n", "rho"])


def ndcg10(d, key, min_n=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _, x in d.groupby(key, sort=False):
        if len(x) < min_n or x.y.sum() == 0:
            continue
        tie = rng.random(len(x)) * 1e-9
        top = np.argsort(-(x.pred.to_numpy() + tie))[:10]
        disc = 1 / np.log2(np.arange(2, 12))
        dcg = (x.y.to_numpy()[top] * disc[:len(top)]).sum()
        idcg = (np.sort(x.y.to_numpy())[::-1][:10] * disc[:len(top)]).sum()
        vals.append(dcg / idcg)
    return float(np.mean(vals))


def share_l1(d, key, min_n=10):
    vals = []
    for _, x in d.groupby(key, sort=False):
        if len(x) < min_n or x.y.sum() == 0:
            continue
        r = np.clip(np.expm1(x.pred.to_numpy()), 0, None)
        p = r / r.sum() if r.sum() > 0 else np.full(len(r), 1 / len(r))
        vals.append(np.abs(p - x.y.to_numpy() / x.y.sum()).sum())
    return float(np.mean(vals))


def metrics(d):
    e = d.pred - d.ly
    raw = np.clip(np.expm1(d.pred), 0, None)
    rt, rs = group_rho(d, "topic"), group_rho(d, "eval_c2000")
    return {"MALE": e.abs().mean(), "RMSLE": np.sqrt((e ** 2).mean()), "MAE": (raw - d.y).abs().mean(),
            "RMSE": np.sqrt(((raw - d.y) ** 2).mean()),
            "Spearman_方向内": np.average(rt.rho, weights=rt.n), "Spearman_子课题内": np.average(rs.rho, weights=rs.n),
            "NDCG@10_子课题内": ndcg10(d, "eval_c2000"), "份额L1_子课题内": share_l1(d, "eval_c2000"),
            "Spearman_全体": spearmanr(d.pred, d.y).statistic}


def main():
    t0 = time.time()
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y1", "y2", "y3a", "y3", "topic", "eval_c2000"])
    pc = pd.read_parquet(os.path.join(C.RES_DIR, "preds_cold.parquet"))
    if os.path.exists(os.path.join(C.RES_DIR, "preds_naip.parquet")):
        pc = pd.concat([pc, pd.read_parquet(os.path.join(C.RES_DIR, "preds_naip.parquet"))], ignore_index=True)
    else:
        print("警告：没有 NAIP 预测，本次评价不含 NAIP")
    pa = pd.read_parquet(os.path.join(C.RES_DIR, "preds_after1y.parquet"))
    models = list(dict.fromkeys(pc.model))
    rows, boot_store = [], {}
    sub_ids = None
    for sp in ("test", "val"):
        for m in models:
            d = pc[(pc.model == m) & (pc.split == sp)].merge(s, on=["paper_id", "split"])
            d = d.assign(y=d.y3, ly=np.log1p(d.y3))
            rows.append({"split": sp, "model": m, "n": len(d), **metrics(d)})
            if sp == "test":
                # 自助法所需的分组聚合
                g = d.assign(ae=(d.pred - d.ly).abs()).groupby("eval_c2000").agg(se=("ae", "sum"), n=("ae", "size"))
                rs = group_rho(d, "eval_c2000").set_index("g")
                g["rn"] = (rs.rho * rs.n).reindex(g.index).fillna(0)
                g["nr"] = rs.n.reindex(g.index).fillna(0)
                if sub_ids is None:
                    sub_ids = g.index.to_numpy()
                boot_store[m] = g.reindex(sub_ids).fillna(0)
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(C.RES_DIR, "04_metrics.csv"), index=False)
    print(f"指标完成（{time.time() - t0:.0f}s）", flush=True)

    # 自助法：所有模型共用同一组重抽样权重
    rng = np.random.default_rng(C.SEED)
    W = rng.multinomial(len(sub_ids), np.full(len(sub_ids), 1 / len(sub_ids)), size=B).astype(np.float64)
    bm, br = {}, {}
    for m, g in boot_store.items():
        bm[m] = (W @ g.se.to_numpy()) / (W @ g.n.to_numpy())
        br[m] = (W @ g.rn.to_numpy()) / (W @ g.nr.to_numpy())
    ci = pd.DataFrame([{"model": m, "MALE_lo": np.quantile(bm[m], .025), "MALE_hi": np.quantile(bm[m], .975),
                        "Sp_sub_lo": np.quantile(br[m], .025), "Sp_sub_hi": np.quantile(br[m], .975)} for m in bm])
    ci.to_csv(os.path.join(C.RES_DIR, "04_boot_ci.csv"), index=False)

    judged = [m for m in models if "含自身" not in m]
    rev = []
    for a in judged:
        for b in judged:
            if a == b:
                continue
            dm, dr = bm[a] - bm[b], br[a] - br[b]
            dm_lo, dm_hi, dr_lo, dr_hi = *np.quantile(dm, [.025, .975]), *np.quantile(dr, [.025, .975])
            if dm_hi < 0 and dr_hi < 0:
                rev.append({"A（绝对误差更好）": a, "B（组内排序更好）": b, "MALE_A": bm[a].mean(), "MALE_B": bm[b].mean(),
                            "dMALE_CI": f"[{dm_lo:+.4f}, {dm_hi:+.4f}]", "Sp_A": br[a].mean(), "Sp_B": br[b].mean(),
                            "dSp_CI": f"[{dr_lo:+.4f}, {dr_hi:+.4f}]"})
    rev = pd.DataFrame(rev)
    rev.to_csv(os.path.join(C.RES_DIR, "04_reversals.csv"), index=False)

    t = met[met.split == "test"].set_index("model")
    learned = [m for m in LEARNED if m in t.index]
    best = t.loc[learned, "MALE"].idxmin()
    c1a = t.loc["D1*-子课题（上帝视角）", "MALE"] <= t.loc[best, "MALE"]
    c1b = t.loc["D1-方向+子课题", "MALE"] <= 1.10 * t.loc[best, "MALE"]
    c1c = len(rev) > 0
    tau = kendalltau(t.loc[judged, "MALE"].rank(), (-t.loc[judged, "Spearman_子课题内"]).rank()).statistic
    c1d = tau < 0.5

    # 发表 1 年后
    ra = []
    for sp in ("test", "val"):
        for m in dict.fromkeys(pa.model):
            d = pa[(pa.model == m) & (pa.split == sp)].merge(s, on=["paper_id", "split"])
            d = d.assign(y=d.y2 + d.y3a, ly=np.log1p(d.y2 + d.y3a))
            ra.append({"split": sp, "model": m, "n": len(d), **metrics(d)})
    ma = pd.DataFrame(ra)

    # 事后：误差分解（测试集）
    dec = []
    for m in models:
        d = pc[(pc.model == m) & (pc.split == "test")].merge(s, on=["paper_id", "split"])
        e = d.pred - np.log1p(d.y3)
        eg = e.groupby(d.eval_c2000).transform("mean")
        dec.append({"model": m, "MSE": (e ** 2).mean(), "整体偏差²": e.mean() ** 2, "组间（子课题）": ((eg - e.mean()) ** 2).mean(),
                    "组内": ((e - eg) ** 2).mean(), "平均误差（偏差）": e.mean()})
    dec = pd.DataFrame(dec)
    assert np.allclose(dec.MSE, dec["整体偏差²"] + dec["组间（子课题）"] + dec["组内"]), "误差分解不闭合"
    dec.to_csv(os.path.join(C.RES_DIR, "04_error_decomposition.csv"), index=False)
    ma.to_csv(os.path.join(C.RES_DIR, "04_metrics_after1y.csv"), index=False)

    var = pd.read_csv(os.path.join(C.RES_DIR, "01_variance.csv"))
    v21 = var[var["sample"].astype(str) == "2021"].iloc[0]
    show = t.drop(columns=["split", "n"]).copy()
    show = show.join(ci.set_index("model"))
    show["MALE"] = show.apply(lambda r: f"{r.MALE:.4f} [{r.MALE_lo:.3f}, {r.MALE_hi:.3f}]" if pd.notna(r.MALE_lo) else f"{r.MALE:.4f}", axis=1)
    show["Spearman_子课题内"] = show.apply(lambda r: f"{r['Spearman_子课题内']:.4f} [{r.Sp_sub_lo:.3f}, {r.Sp_sub_hi:.3f}]"
                                         if pd.notna(r.Sp_sub_lo) else f"{r['Spearman_子课题内']:.4f}", axis=1)
    show = show.drop(columns=["MALE_lo", "MALE_hi", "Sp_sub_lo", "Sp_sub_hi"])
    mk = lambda ok: "✅ 通过" if ok else "❌ 未通过"
    md = ["# 实验 1 结果：评价诊断 C1\n",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}。测试集 = 2021 年焦点论文（{int(t.n.max()):,} 篇），目标 = 发表后第 1–3 年被引之和。"
          "置信区间：按子课题整组重抽样 1,000 次。设计与标准见 `实验1说明.md`（预先登记）。\n",
          "## 1. 方差分解（2021 年）\n",
          f"log1p(y3) 的方差中，方向 × 年份之间占 {v21['方向×年份']:.1%}，子课题(2000) × 年份之间占 {v21['子课题(2000)×年份']:.1%}，"
          f"子课题(5000) × 年份之间占 {v21['子课题(5000)×年份']:.1%}。完整表见 `01_variance.md`。\n",
          "## 2. 冷启动：所有模型的两类指标（测试集）\n",
          show.round(4).to_markdown(), "\n",
          "## 3. 假设判定\n",
          "| 假设 | 结果 | 判定 |", "|---|---|---|",
          f"| C1-a 完美热度的傻模型在 MALE 上赢过学习模型 | D1*-子课题 {t.loc['D1*-子课题（上帝视角）', 'MALE']:.4f} vs 最好学习模型（{best}）{t.loc[best, 'MALE']:.4f} | {mk(c1a)} |",
          f"| C1-b T 时可得的热度模型 MALE ≤ 1.10 × 最好学习模型 | D1-方向+子课题 {t.loc['D1-方向+子课题', 'MALE']:.4f} vs 1.10 × {t.loc[best, 'MALE']:.4f} = {1.1 * t.loc[best, 'MALE']:.4f} | {mk(c1b)} |",
          f"| C1-c 存在排序颠倒 | {len(rev)} 对模型 | {mk(c1c)} |",
          f"| C1-d 两种排名不一致（Kendall τ < 0.5） | τ = {tau:.3f} | {mk(c1d)} |", "",
          "## 4. 排序颠倒的模型对（绝对误差显著更好、子课题内排序显著更差）\n",
          rev.round(4).to_markdown(index=False) if len(rev) else "（无）", "\n",
          "## 5. 事后分析：均方对数误差分解（测试集，不计入判定）\n",
          "MSE = 整体偏差² + 组间（各子课题平均误差的离散）+ 组内。前两项是\"整体水平 / 热度猜得准不准\"，第三项才和组内区分论文有关。"
          "注意：留一上帝视角模型的组内 Spearman 是构造性的负值（组内其他论文的均值与自身高低相反），C1-c 中涉及它们的颠倒不具实质意义。\n",
          dec.round(4).to_markdown(index=False), "\n",
          "## 6. 发表 1 年后（目标 = 第 2、3 年被引之和，测试集）\n",
          ma[ma.split == "test"].drop(columns=["split"]).round(4).to_markdown(index=False), "\n"]
    open(os.path.join(C.RES_DIR, "实验1结果.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    print(f"04 DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
