"""
把 MPCC 全部实验的结果整理成一个 Excel 工作簿（每张表一个工作表）+ 每张表一个 CSV + 一份 Markdown 汇总。

数据来源：服务器 /path/to/mpcc/ 下各实验的结果文件（全部直接读取，不手工抄数）；
少数只以 Markdown 形式存在的结果（ICLR 质量锚点、自引稳健性）从 Markdown 表格中解析；
基线耗时来自运行日志（见 baselines/*/README.md），在 RUNTIME 中列出。

输出（服务器）：/path/to/mpcc/summary/MPCC_全部实验结果.xlsx、tables/*.csv、MPCC_全部实验结果.md

用法（服务器）：/path/to/conda/envs/mpcc/bin/python collect_tables.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

R = "/path/to/mpcc"
OUT = f"{R}/summary"
FIELDS = {"AI": "", "肿瘤学": "_sf2730", "应用数学": "_sf2604"}
T = []          # (表名, 说明, DataFrame)


def add(name, note, df):
    if df is not None and len(df):
        T.append((name, note, df))
    else:
        print("跳过（无数据）:", name)


def read(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def by_field(rel, note_name, note):
    parts = []
    for f, tag in FIELDS.items():
        d = read(f"{R}/{rel.format(tag=tag)}")
        if d is not None:
            parts.append(d.assign(学科=f))
    if parts:
        d = pd.concat(parts, ignore_index=True)
        add(note_name, note, d[["学科"] + [c for c in d.columns if c != "学科"]])


def md_tables(path):
    """解析 Markdown 文件中的全部表格"""
    if not os.path.exists(path):
        return []
    out, cur = [], []
    for line in open(path, encoding="utf-8").read().splitlines() + [""]:
        if line.strip().startswith("|"):
            cur.append(line.strip())
        elif cur:
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in cur if not re.match(r"^\|[\s:\-|]+\|$", r)]
            if len(rows) >= 2:
                out.append(pd.DataFrame(rows[1:], columns=rows[0]))
            cur = []
    return out


def main():
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    # ---------------- 数据
    rows = []
    import duckdb
    con = duckdb.connect()
    for f, tag in FIELDS.items():
        s = pd.read_parquet(f"{R}/subsets/pred{tag}_v2/samples.parquet", columns=["split", "y3"])
        n_corpus = con.execute(f"select count(*) from read_parquet('{R}/subsets/exp0{tag}_v2/papers.parquet')").fetchone()[0]
        rows.append({"学科": f, "语料论文数（2016–2024）": n_corpus, "焦点论文数（2017–2021）": len(s),
                     "训练（2017–2019）": int((s.split == "train").sum()), "验证（2020）": int((s.split == "val").sum()),
                     "测试（2021）": int((s.split == "test").sum()), "零被引比例": round((s.y3 == 0).mean(), 4),
                     "三年被引均值": round(s.y3.mean(), 2), "三年被引中位数": s.y3.median()})
    add("01_数据集统计", "三个学科的语料与焦点论文（论文 Table 2）", pd.DataFrame(rows))
    # ---------------- 实验 0
    by_field("exp0_analysis/results_v2{tag}/12_macro.csv", "02_实验0_宏观", "0-1：格子论文数 → 格子总被引（Poisson 弹性 b）")
    by_field("exp0_analysis/results_v2{tag}/13_micro.csv", "03_实验0_微观", "0-2/0-3/0-7：前一年相似论文数 → 被引（含稳健性与异质性）")
    by_field("exp0_analysis/results_v2{tag}/15_twins.csv", "04_实验0_撞车", "0-5：撞车论文对中后发表者的被引差异")
    by_field("exp0_analysis/results_v2{tag}/16_crossdomain.csv", "05_实验0_跨域", "0-6：第二方向拥挤（控制需求）")
    add("06_实验0_假对手", "0-4：组内置换的假对手检验", read(f"{R}/exp0_analysis/results_v2/14_placebo.csv"))
    by_field("exp0_analysis/results_v2{tag}/18_wave.csv", "07_实验0_浪潮_事后", "【事后，不计入判定】浪潮位置分析")
    noself = []
    for k in ("12_macro", "13_micro", "15_twins", "16_crossdomain"):
        d = read(f"{R}/exp0_analysis/results_v2_noself/{k}.csv")
        if d is not None:
            noself.append(d.assign(检验=k))
    if noself:
        add("08_实验8_剔除自引", "实验 8 R7：剔除自引后重做实验 0 的检验（AI）", pd.concat(noself, ignore_index=True))
    add("09_实验0_图1数据", "论文 Figure 1 的画图数据", read(f"{R}/exp0_analysis/results_v2/19_teaser_data.csv"))
    # ---------------- 实验 1
    by_field("exp1/results{tag}/01_variance.csv", "10_实验1_方差分解", "被引差异在方向 / 子课题之间与之内的分解")
    add("11_实验1_模型指标", "傻模型、LightGBM、MLP、NAIP 在测试集上的全部指标", read(f"{R}/exp1/results/04_metrics.csv"))
    add("12_实验1_误差分解", "【事后】模型误差差距的水平 / 组间 / 组内分解", read(f"{R}/exp1/results/04_error_decomposition.csv"))
    add("13_实验1_显著性", "C1 预先登记检验的重抽样区间", read(f"{R}/exp1/results/04_boot_ci.csv"))
    # ---------------- 实验 2 / 3
    add("14_主结果_全部来源", "论文 Table 3 上半：全部对比方法（全部来源被引，2021 测试集）", read(f"{R}/exp2/results/12_baselines_all.csv"))
    add("15_主结果_语料内", "论文 Table 3 下半：语料内被引口径（图方法的口径）", read(f"{R}/exp2/results/12_baselines_dp.csv"))
    add("16_终版评价", "终版（A + B 集成）测试一次的全部指标", read(f"{R}/exp2/results/08_final_metrics.csv"))
    add("17_消融_v5", "论文 Table 5：基础配置的消融（5 种子）", read(f"{R}/exp2/results/03_summary_v5.csv"))
    add("18_消融_v5_检验", "消融与 P1–P4 的重抽样检验", read(f"{R}/exp2/results/03_tests_v5.csv"))
    add("19_v4_全量", "v4（log 空间显式分解）10 配置 × 5 种子", read(f"{R}/exp2/results/03_summary_v4.csv"))
    add("20_发表一年后", "已知第一年被引时的预测（对手不再有用）", read(f"{R}/exp2/results/03_summary_after1y.csv"))
    add("21_需求塔探索", "需求塔（格子总量）各方法的验证集比较", read(f"{R}/exp2/results/04_demand_explore.csv"))
    st = json.load(open(f"{R}/exp2/autosearch/state.json"))
    base = {"logshare": 1.0, "aux_w": 0.3, "n_self": 1, "heads": 4, "d": 256, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4, "bs": 3000,
            "patience": 5, "ema": 0.0, "rdrop": 0.0, "loss_fn": "mse", "tdir": "_v6", "min_ep": 0}
    tr = []
    for t in st["trials"]:
        c = {**base, **t["cfg"]}
        tr.append({"配置": t["name"], "轮": t["round"], **c, "各种子验证MALE": " / ".join(f"{v:.4f}" for v in t["val"]),
                   "验证MALE均值": np.mean(t["val"]) if t["val"] else np.nan, "最佳轮": "/".join(map(str, t["epochs"]))})
    add("22_自动迭代_全部配置", "第六轮自动迭代：36 个配置 × 3 种子（论文 Table 敏感性的来源）", pd.DataFrame(tr).sort_values("验证MALE均值"))
    inn = []
    for k in ("r1", "r2", "r3"):
        d = read(f"{R}/exp2/innov/{k}_summary.csv")
        if d is not None:
            inn.append(d.assign(轮=k))
    if inn:
        add("23_第七轮_新结构", "第七轮：由研究设想推出的 5 种结构（验证集，配对比较）", pd.concat(inn, ignore_index=True))
    add("24_跨学科_对比基线", "论文 Table 跨学科最后两行：三个学科与最强对比方法", read(f"{R}/exp2/results/14_field_baselines.csv"))
    for f, tag in list(FIELDS.items())[1:]:
        add(f"25_跨学科_{f}", f"{f}：MPC-Net 基础配置与消融（3 种子）", read(f"{R}/exp2/results{tag}/03_summary_field.csv"))
    add("26_实验6_跨学科汇总", "实验 6：三个学科的预先登记检验汇总", read(f"{R}/exp6/results/02_fields.csv"))
    add("27_误差来源_图3", "论文 Figure 3 的画图数据（竞争贡献分档；各届逐年被引）", read(f"{R}/exp2/results/13_fig3_data.csv"))
    # ---------------- 实验 4 / 5 / 7 / 9
    add("28_实验4_驱动因素", "竞争贡献对竞争集合描述量的回归（标准化系数）", read(f"{R}/exp4/results/01_drivers.csv"))
    add("29_实验4_共被引", "【事后】共被引（互补）与竞争贡献", read(f"{R}/exp4/results/01_cocite.csv"))
    add("30_实验4_基尼", "子课题内被引不平等：真实 vs 各模型", read(f"{R}/exp4/results/01_gini.csv"))
    add("31_实验5_事件研究_系数", "强对手出现事件：逐期系数", read(f"{R}/exp5/results/01_event_coefs.csv"))
    add("32_实验5_事件研究_DID", "强对手出现事件：单一系数（处理 × 事后）", read(f"{R}/exp5/results/01_event_did.csv"))
    add("33_实验5_假对手", "论文 Table 假对手：换成其他论文的竞争集合", read(f"{R}/exp5/results/02_placebo.csv"))
    for i, d in enumerate(md_tables(f"{R}/exp5/results/03_iclr.md")):
        add(f"34_实验5_ICLR_{i + 1}", "ICLR 审稿分质量锚点（解析自 03_iclr.md）", d)
    add("35_实验7_推荐", "子课题内推荐前 10 篇（论文 Table 应用）", read(f"{R}/exp7/results/01_recommend.csv"))
    add("36_实验7_早期发现", "发表一年后找出未来高被引论文", read(f"{R}/exp7/results/01_early.csv"))
    add("37_实验7_被低估论文", "第一年被引 ≤ 1 的论文按冷启动预测分十档", read(f"{R}/exp7/results/01_undervalued.csv"))
    add("38_实验9_效率", "SHARE-Net 基础配置的训练 / 预测耗时与规模", read(f"{R}/exp9/results/01_efficiency.csv"))
    runtime = pd.DataFrame([
        ("SHARE-Net（终版配置）", "1 × RTX 4090", "无（向量与对手检索预先算好）", "每轮 15.4 s，约 7 分钟（早停）", "84,566 篇 2.9 s", "runs/AS_r06_1_s*/summary.json"),
        ("DPPDCC", "1 × RTX 4090（CPU 受限）", "约 2 小时（子图）", "每轮约 13 分钟（5 轮）", "约 15 分钟", "logs/dppdcc_*.log"),
        ("H2CGL", "1 × RTX 4090（CPU 受限）", "约 3.5 小时", "约 7 小时（20 轮）", "约 38 分钟", "baselines/h2cgl/README.md"),
        ("HINTS", "CPU 8 线程（TF1 不支持 RTX 4090）", "约 1 分钟", "每次 4.6–5.8 小时", "含在训练中", "baselines/hints/README.md"),
        ("PLM-FT（SPECTER2 微调）", "1 × RTX 4090", "分词约 5 分钟", "每次约 35 分钟（3 轮）", "数秒", "baselines/plm_ft/log_*.csv"),
        ("NAIP", "4 × RTX 4090（8 比特）", "无", "不训练（官方权重）", "每张卡 22.5K 篇约 13 分钟", "exp1/logs"),
        ("LightGBM / MLP", "128 线程 CPU / 1 GPU", "无", "23 s / 14 s", "秒级", "exp9/results"),
    ], columns=["方法", "硬件", "预处理", "训练", "预测（测试集）", "来源"])
    add("39_各方法耗时", "论文 Efficiency 段落的来源（基线耗时取自运行日志）", runtime)
    # ---------------- 写出
    idx = pd.DataFrame([(n, note, len(d), d.shape[1]) for n, note, d in T], columns=["工作表", "内容", "行数", "列数"])
    xlsx = f"{OUT}/MPCC_全部实验结果.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        idx.to_excel(w, sheet_name="00_目录", index=False)
        for n, note, d in T:
            d.to_excel(w, sheet_name=n[:31], index=False)
            d.to_csv(f"{OUT}/tables/{n}.csv", index=False, encoding="utf-8-sig")
    md = ["# MPCC 全部实验结果汇总\n", "每张表直接由服务器上的结果文件生成（`collect_tables.py`）。表名前的编号与 Excel 工作表一致。\n",
          idx.to_markdown(index=False), "\n"]
    for n, note, d in T:
        md += [f"\n## {n}\n", f"{note}\n", d.round(4).to_markdown(index=False), "\n"]
    open(f"{OUT}/MPCC_全部实验结果.md", "w", encoding="utf-8").write("\n".join(md))
    print(f"写出 {len(T)} 张表 → {xlsx}")
    print(idx.to_string(index=False))


if __name__ == "__main__":
    main()
