"""
【11】实验 2 · 对比方法：预训练语言模型微调（PLM-FT，SPECTER2-base / SciBERT 系列），只用标题 + 摘要

用途
  近年引用预测论文（如 NAIP AAAI'25、EACL'23 Findings 的新论文被引预测）的常用对比：
  把 [标题 [SEP] 摘要] 输入 SPECTER2-base（SciBERT 初始化），[CLS] 向量接线性层回归 log(1 + 三年被引)。
  与 MLP-文本（冻结 SPECTER2 向量）的区别：整个编码器一起微调。
  训练 2017–2019，按 2020 验证集 MALE 选检查点（每半轮评价一次），2021 测试只在最后预测一次。

口径
  --target all：全部来源被引（主表口径，samples.parquet 的 y3）
  --target dp ：语料内被引（DPPDCC 口径，tensors_v6dp/y.npy）

输出
  /path/to/mpcc/baselines/plm_ft/preds_<口径>_s<种子>.parquet：paper_id, split, seed, pred（log1p 尺度）
  /path/to/mpcc/baselines/plm_ft/log_<口径>_s<种子>.csv

用法
  python 11_plm_finetune.py --target all --seed 1 --gpu 0
  python 11_plm_finetune.py --target all --tag _sf2730 --seed 1 --gpu 0     其他学科（肿瘤学 / 应用数学 _sf2604）
"""
import argparse
import os
import time

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL = "/path/to/mpcc/models/hub/models--allenai--specter2_base/snapshots/3447645e1def9117997203454fa4495937bfbd83/"
OUT = "/path/to/mpcc/baselines/plm_ft/"


def load_data(target, tag=""):
    s = pd.read_parquet(f"/path/to/mpcc/subsets/pred{tag}_v2/samples.parquet", columns=["paper_id", "split", "y3"])
    if target == "dp":
        meta = pd.read_parquet("/path/to/mpcc/exp2/tensors_v6dp/meta.parquet", columns=["paper_id"])
        ydp = pd.Series(np.load("/path/to/mpcc/exp2/tensors_v6dp/y.npy").sum(1), index=meta.paper_id.to_numpy())
        s["y3"] = ydp.reindex(s.paper_id).to_numpy()
        assert s.y3.notna().all()
    con = duckdb.connect()
    con.register("ids", s[["paper_id"]])
    t = con.execute(f"""select p.paper_id, coalesce(p.title, '') as title, coalesce(p.abstract, '') as abstract
                       from read_parquet('/path/to/mpcc/subsets/exp0{tag}_v2/papers.parquet') p join ids using (paper_id)""").df()
    s = s.merge(t, on="paper_id", how="left")
    assert s.title.notna().all()
    return s


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.out(self.drop(h)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all", choices=["all", "dp"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--tag", default="", help="学科标签：空 = AI；_sf2730 = 肿瘤学；_sf2604 = 应用数学（只支持 --target all）")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    t0 = time.time()
    assert not (args.tag and args.target == "dp"), "其他学科只有全部来源口径"
    s = load_data(args.target, args.tag)
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(list(s.title), list(s.abstract), truncation="longest_first", max_length=args.max_len, padding="max_length", return_tensors="np")
    X, M = torch.from_numpy(enc["input_ids"]), torch.from_numpy(enc["attention_mask"])
    y = torch.from_numpy(np.log1p(s.y3.to_numpy(np.float32)))
    sp = s.split.to_numpy()
    tr, va, te = [np.where(sp == k)[0] for k in ("train", "val", "test")]
    mu = float(y[tr].mean())
    print(f"[{args.target} s{args.seed}] 数据 {len(s):,}（训练 {len(tr):,}），分词 {time.time() - t0:.0f}s", flush=True)
    dev = torch.device("cuda")
    net = Reg().to(dev)
    with torch.no_grad():
        net.out.bias.fill_(mu)                             # 输出偏置初始化为训练集平均，避免开头的大误差
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    @torch.no_grad()
    def predict(idx):
        net.eval()
        out = []
        for b in range(0, len(idx), 512):
            j = idx[b:b + 512]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(net(X[j].to(dev), M[j].to(dev)).float().cpu())
        net.train()
        return torch.cat(out).numpy()

    best, logs, step = 1e9, [], 0
    ck = os.path.join(OUT, f"model_{args.target}{args.tag}_s{args.seed}.pt")
    eval_every = (len(tr) // args.bs) // 2
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        perm = rng.permutation(tr)
        for b in range(0, len(perm) - args.bs + 1, args.bs):
            j = perm[b:b + args.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(X[j].to(dev), M[j].to(dev)).float()
            loss = nn.functional.mse_loss(p, y[j].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if step % eval_every == 0:
                pv = predict(va)
                male = float(np.abs(pv - y[va].numpy()).mean())
                logs.append({"step": step, "epoch": round(step / (len(tr) // args.bs), 2), "val_MALE": male, "sec": round(time.time() - t0)})
                print(f"[{args.target} s{args.seed}] step {step} 轮 {logs[-1]['epoch']} val MALE {male:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if male < best:
                    best = male
                    torch.save(net.state_dict(), ck)
    pd.DataFrame(logs).to_csv(os.path.join(OUT, f"log_{args.target}{args.tag}_s{args.seed}.csv"), index=False)
    net.load_state_dict(torch.load(ck))
    rows = []
    for k, idx in (("val", va), ("test", te)):
        rows.append(pd.DataFrame({"paper_id": s.paper_id.to_numpy()[idx], "split": k, "seed": args.seed, "pred": predict(idx)}))
    pd.concat(rows).to_parquet(os.path.join(OUT, f"preds_{args.target}{args.tag}_s{args.seed}.parquet"), index=False)
    os.remove(ck)
    print(f"[{args.target} s{args.seed}] DONE best val MALE {best:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
