"""
【08】诊断：前移设定（2017–2018 训练、2019 验证）下 MLP 在验证年系统性高估 +0.31，而主设定验证年只有 +0.03。
  检查：① 训练年的平均预测是否等于真实均值（检验训练本身是否正常）；② 2019 年哪些输入特征相对训练年漂移最大
  （MLP 的数值特征按训练集标准化、截断在 ±10，原始计数类特征随年份增长时会外推）；③ 主设定下同样的漂移量作对照。
用法
  python 08_shift_mlp_diag.py --gpu 6
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


def drift(s, num, tr_mask, ev_mask):
    A = s[num].astype(np.float64)
    mu, sd = A[tr_mask].mean(), A[tr_mask].std().replace(0, 1)
    z = ((A[ev_mask] - mu) / sd).mean()
    return z.sort_values(key=np.abs, ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    import torch
    s0 = pd.read_parquet(C.SAMPLES)
    PCS = [f"pc{i}" for i in range(C.N_PCA)]
    out = []
    for name, rule in (("主设定（训练 2017–2019，验证 2020）", lambda y: np.where(y <= 2019, "train", np.where(y == 2020, "val", "none"))),
                       ("前移设定（训练 2017–2018，验证 2019）", lambda y: np.where(y <= 2018, "train", np.where(y == 2019, "val", "none")))):
        s = s0.copy()
        s["split"] = rule(s.Y.to_numpy())
        s = s[s.split != "none"].reset_index(drop=True)
        s.to_parquet("/tmp/_diag_samples.parquet")
        C.SAMPLES = "/tmp/_diag_samples.parquet"
        s, X = M.prepare(False)
        L4 = C.META + C.HEAT_TOPIC + C.HEAT_SUB + C.HEAT2 + PCS + C.COMP
        num = [c for c in L4 if c not in C.CATEG]
        tr, va = (s.split == "train").to_numpy(), (s.split == "val").to_numpy()
        z = drift(s, [c for c in num if not c.startswith("pc")], tr, va)
        # 与 05 / 06 相同的 MLP（选定参数），记录训练年与验证年的平均预测
        torch.manual_seed(C.SEED)
        dev = torch.device(f"cuda:{args.gpu}")
        A = s[num].astype(np.float32)
        miss = [c for c in num if A[c].isna().any()]
        Mi = A[miss].isna().astype(np.float32).add_suffix("_na")
        mu, sd = A[tr].mean(), A[tr].std().replace(0, 1)
        Az = ((A - mu) / sd).fillna(0)
        clip_share = float((Az[va].abs() > 10).to_numpy().mean())
        Az = Az.clip(-10, 10)
        O = pd.get_dummies(s[C.CATEG], dtype=np.float32)
        Xn = np.hstack([X / np.linalg.norm(X, axis=1, keepdims=True), Az.to_numpy(), Mi.to_numpy(), O.to_numpy()]).astype(np.float32)
        T = lambda a: torch.from_numpy(a).to(dev)
        Xt, yt = T(Xn), T(s.ly3.to_numpy(np.float32))
        itr, iva = np.where(tr)[0], np.where(va)[0]
        net = torch.nn.Sequential(torch.nn.Linear(Xn.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.3),
                                  torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(512, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        best = (1e9, None)
        for ep in range(100):
            net.train()
            perm = itr[np.random.default_rng(ep).permutation(len(itr))]
            for b in range(0, len(perm), 1024):
                i = T(perm[b:b + 1024])
                loss = torch.nn.functional.mse_loss(net(Xt[i]).squeeze(1), yt[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xt[T(iva)]).squeeze(1)
                v = (pv - yt[T(iva)]).abs().mean().item()
            if v < best[0] - 1e-4:
                best, bad = (v, {k: x.clone() for k, x in net.state_dict().items()}), 0
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(best[1])
        net.eval()
        with torch.no_grad():
            ptr = torch.cat([net(Xt[T(itr[b:b + 8192])]).squeeze(1) for b in range(0, len(itr), 8192)]).cpu().numpy()
            pva = torch.cat([net(Xt[T(iva[b:b + 8192])]).squeeze(1) for b in range(0, len(iva), 8192)]).cpu().numpy()
        ytr, yva = s.ly3.to_numpy()[itr], s.ly3.to_numpy()[iva]
        out.append(f"## {name}\n")
        out.append(f"- 训练年：平均预测 {ptr.mean():.3f}，真实均值 {ytr.mean():.3f}（偏差 {ptr.mean() - ytr.mean():+.3f}）")
        out.append(f"- 验证年：平均预测 {pva.mean():.3f}，真实均值 {yva.mean():.3f}（偏差 {pva.mean() - yva.mean():+.3f}），MALE {np.abs(pva - yva).mean():.4f}")
        out.append(f"- 验证年数值特征被截断（|z| > 10）的比例：{clip_share:.4f}")
        out.append("- 验证年相对训练年漂移最大的 10 个数值特征（验证年平均 z 值）：")
        out.append("  " + "；".join(f"{k} {v:+.2f}" for k, v in z.head(10).items()))
        out.append("")
    md = "# 诊断：前移设定下 MLP 的系统性高估\n\n" + "\n".join(out)
    open(os.path.join(C.RES_DIR, "08_shift_mlp_diag.md"), "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
