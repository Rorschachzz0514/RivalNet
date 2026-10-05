"""
【03】实验 1 · 已发表方法 NAIP（AAAI 2025）：只读标题 + 摘要的 LLM 影响力预测（多 GPU 推理 + 保序校准）

用途
  官方权重 ssocean/NAIP（LLaMA-3-8B 序列分类头，8 比特量化）+ 同一仓库里的 LoRA adapter（必须叠加，否则是未微调的基座），
  提示词与官方 NAID_Dataset 完全一致：
    "Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):"
  输出 sigmoid(logit) ∈ (0, 1)。对验证集（2020）与测试集（2021）的全部样本推理；在验证集上用保序回归把分数映射到
  log1p(y3)，再用于测试集（NAIP 原本预测的是归一化分数，不是被引数，校准后才能算绝对误差；组内排序不受校准影响）。
  注意：NAIP 的训练数据（NAID）由作者在 2024 年左右构建，可能与本实验的测试论文重叠，且其标签用到了这些论文之后的被引，
  对 NAIP 偏有利；结果解读时说明。
  并行：每张 GPU 一个进程（spawn），各处理一部分论文，按文本长度排序后分批（减少填充）；每块结果单独保存，可断点续跑。

输入
  config_exp1.SAMPLES（val / test 的 paper_id）、config_exp1.PAPERS（标题、摘要）、config_exp1.NAIP_DIR（权重）

输出（results/）
  naip_scores.parquet   paper_id, split, naip_score
  preds_naip.parquet    paper_id, split, model = "NAIP (AAAI'25)", pred（校准到 log1p(y3)）
  _naip_parts/          各 GPU 分块

用法
  python 03_naip.py --gpus 0,1,3,4,5,7 [--test]
"""
import argparse
import glob
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp1 as C

PROMPT = "Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):"
PARTS = os.path.join(C.RES_DIR, "_naip_parts")


def worker(gpu, part, nparts, batch, test):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    out_p = os.path.join(PARTS, f"part{part:02d}{'_test' if test else ''}.parquet")
    if os.path.exists(out_p):
        print(f"[gpu{gpu}] 已完成，跳过", flush=True)
        return
    d = pd.read_parquet(os.path.join(PARTS, "_input.parquet"))
    d = d.iloc[np.array_split(np.arange(len(d)), nparts)[part]]
    if test:
        d = d.head(256)
    tok = AutoTokenizer.from_pretrained(C.NAIP_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    from peft import PeftModel
    model = AutoModelForSequenceClassification.from_pretrained(C.NAIP_DIR, num_labels=1, device_map={"": 0}).eval()
    model.config.pad_token_id = tok.pad_token_id
    # 官方仓库里的完整权重是"基座"，微调结果在 LoRA adapter（q/v 投影 + score 头）里，必须叠加；
    # 只加载完整权重时，验证集上与被引的 Spearman ≈ −0.07，叠加后 ≈ 0.38（800 篇抽查，见运行记录）
    model = PeftModel.from_pretrained(model, C.NAIP_ADAPTER).eval()
    texts = [PROMPT.format(title=str(t).strip().replace("\n", ""), abstract=str(a).strip().replace("\n", ""))
             for t, a in zip(d.title.fillna(""), d.abstract.fillna(""))]
    order = np.argsort([len(x) for x in texts])
    scores = np.empty(len(texts), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([texts[i] for i in idx], max_length=512, padding=True, truncation=True, return_tensors="pt").to("cuda")
            scores[idx] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
            if (b // batch) % 100 == 0:
                print(f"[gpu{gpu}] {b + len(idx):,}/{len(order):,} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame({"paper_id": d.paper_id.to_numpy(), "split": d.split.to_numpy(), "naip_score": scores}).to_parquet(out_p, index=False)
    print(f"[gpu{gpu}] DONE {len(d):,} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,3,4,5,7")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    import numpy as np
    import pandas as pd
    os.makedirs(PARTS, exist_ok=True)
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y3"])
    s = s[s.split.isin(["val", "test"])]
    if not os.path.exists(os.path.join(PARTS, "_input.parquet")):
        p = pd.read_parquet(C.PAPERS, columns=["paper_id", "title", "abstract"])
        s[["paper_id", "split"]].merge(p, on="paper_id", how="left").to_parquet(os.path.join(PARTS, "_input.parquet"), index=False)
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(g, i, len(gpus), args.batch, args.test)) for i, g in enumerate(gpus)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    files = sorted(glob.glob(os.path.join(PARTS, f"part*{'_test' if args.test else ''}.parquet")))
    files = [f for f in files if args.test or not f.endswith("_test.parquet")]
    if len(files) != len(gpus):
        sys.exit(f"分块缺失 {len(files)}/{len(gpus)}")
    sc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if args.test:
        print(sc.describe())
        return
    assert len(sc) == len(s) and sc.paper_id.is_unique, "NAIP 分数条数应等于验证 + 测试样本数"
    sc.to_parquet(os.path.join(C.RES_DIR, "naip_scores.parquet"), index=False)
    from sklearn.isotonic import IsotonicRegression
    d = sc.merge(s[["paper_id", "y3"]], on="paper_id")
    d["ly3"] = np.log1p(d.y3)
    v = d[d.split == "val"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(v.naip_score, v.ly3)
    d["pred"] = iso.predict(d.naip_score)
    d.assign(model="NAIP (AAAI'25)")[["paper_id", "split", "model", "pred"]].to_parquet(os.path.join(C.RES_DIR, "preds_naip.parquet"), index=False)
    from scipy.stats import spearmanr
    for sp in ("val", "test"):
        x = d[d.split == sp]
        print(sp, "Spearman(score, y3) =", round(spearmanr(x.naip_score, x.y3).statistic, 4), " MALE =", round((x.pred - x.ly3).abs().mean(), 4))
    print("03 DONE", flush=True)


if __name__ == "__main__":
    main()
