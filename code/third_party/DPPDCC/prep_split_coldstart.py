"""MPCC：为 DPPDCC 重写 split_data（严格冷启动，避免看到预测窗口内的被引）。
DPPDCC 按阶段使用固定的快照窗口（train：2017–2019，val：2018–2020，test：2019–2021），每篇论文看到窗口最后一年为止的网络。
若训练集含 2017、2018 年论文，它们在 2019 年快照里已能看到 2018–2019 年收到的被引（正是预测目标的一部分）→ 泄漏。
因此：train = 2019 年论文、val = 2020 年论文、test = 2021 年论文（主分析集，与 MPC-Net 相同）；标签 = 语料内 Y+1～Y+3 年被引。
DPPDCC 只为 test 名单里的论文建索引并假设 test ⊇ train ∪ val，故 test 名单 = 2019–2021 年全部论文，评价时只取 2021 年。"""
import json, re
import numpy as np, pandas as pd, torch
P = "./data/mpcc_ai/"
s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "split", "Y"])
info = json.load(open(P + "sample_info_dict.json")); ab = json.load(open(P + "sample_abstract_dict.json"))
cy = json.load(open(P + "sample_cite_year_dict.json"))
def block(d):
    ids = [str(x) for x in d.paper_id]
    vals = [sum(int(c) for yy, c in cy.get(i, {}).items() if Y + 1 <= int(yy) <= Y + 3) for i, Y in zip(ids, d.Y)]
    cont = [re.sub(r"\s+", " ", str(info[i]["title"]) + ". " + ab[i]["abstract"]) for i in ids]
    return [ids, vals, cont, [int(y) for y in d.Y]]
cut = {"train": block(s[s.Y == 2019]), "val": block(s[s.Y == 2020]), "test": block(s[s.Y.between(2019, 2021)])}
for k, v in cut.items():
    print(k, len(v[0]), "mean label", round(float(np.mean(v[1])), 3))
torch.save(cut, P + "split_data")
print("SPLIT DONE")
