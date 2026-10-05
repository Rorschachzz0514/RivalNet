"""MPCC：DPPDCC 的主口径版本（data_source = mpcc_ai_all）。
图数据、快照、子图全部与 mpcc_ai 共享（符号链接），只把 split_data 的标签换成"全部来源被引"Y+1～Y+3 年之和（samples.parquet 的 y3），
用于与主表（MPC-Net 终版 0.6745 等）同口径比较。模型检查点与结果分别写到 checkpoints/mpcc_ai_all、results/mpcc_ai_all，不覆盖语料内口径。"""
import os, json, shutil
import pandas as pd, torch
S, D = "./data/mpcc_ai/", "./data/mpcc_ai_all/"
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f not in ("split_data",) and not os.path.exists(D + f):
        os.symlink(os.path.abspath(S + f), D + f)
y = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "y3"])
y = dict(zip(y.paper_id.astype(str), y.y3.astype(int)))
sp = torch.load(S + "split_data", weights_only=False)
for k in sp:
    sp[k][1] = [y[i] for i in sp[k][0]]
    print(k, len(sp[k][0]), "mean label", round(sum(sp[k][1]) / len(sp[k][1]), 3))
torch.save(sp, D + "split_data")
CS, CD = "./checkpoints/mpcc_ai/", "./checkpoints/mpcc_ai_all/"
os.makedirs(CD, exist_ok=True)
for f in os.listdir(CS):
    if not f.endswith(".pkl") and not os.path.exists(CD + f):
        os.symlink(os.path.abspath(CS + f), CD + f)
shutil.copy("./configs/mpcc_ai.json", "./configs/mpcc_ai_all.json")
print("ALL SPLIT DONE", sorted(os.listdir(CD)))
