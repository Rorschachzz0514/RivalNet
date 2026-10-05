"""MPCC 第二步：① graph_sample → graph_sample.dgl；② 已有 SPECTER2 向量按节点顺序导出（DPPDCC 的 specter2 模式读取预算向量）；
③ 直接写 split_data，使训练 / 验证 / 测试与 MPCC 实验 2 完全一致（主分析集 2017–2019 / 2020 / 2021），
   标签 = 语料内三年被引（Y+1～Y+3 年被语料内论文引用的次数，DPPDCC 的被引口径）；④ 写 configs/mpcc_ai.json。"""
import json, re, time
import dgl, joblib, numpy as np, pandas as pd, torch
t0 = time.time()
P = "./data/mpcc_ai/"
g = torch.load(P + "graph_sample")
dgl.save_graphs(P + "graph_sample.dgl", [g]); print("dgl saved", g.num_nodes("paper"), round(time.time() - t0), flush=True)
trans = json.load(open(P + "sample_node_trans.json"))["paper"]
em = pd.read_parquet("/path/to/mpcc/subsets/exp0_v2/emb_meta.parquet", columns=["paper_id", "row"])
row = dict(zip(em.paper_id.astype(str), em.row))
E = np.load("/path/to/mpcc/subsets/exp0/emb_sample.npy", mmap_mode="r")
order = sorted(trans.items(), key=lambda x: x[1])
assert [v for _, v in order] == list(range(len(order)))
X = np.asarray(E[[row[k] for k, _ in order]], dtype=np.float32)
joblib.dump(X, P + "graph_sample_feature_specter2_embs"); print("embs", X.shape, round(time.time() - t0), flush=True)
s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "split", "Y"])
info = json.load(open(P + "sample_info_dict.json")); ab = json.load(open(P + "sample_abstract_dict.json"))
cy = json.load(open(P + "sample_cite_year_dict.json"))
cut = {}
for sp in ("train", "val", "test"):
    d = s[s.split == sp]
    ids = [str(x) for x in d.paper_id]
    vals = [sum(int(c) for yy, c in cy.get(i, {}).items() if Y + 1 <= int(yy) <= Y + 3) for i, Y in zip(ids, d.Y)]
    cont = [re.sub(r"\s+", " ", str(info[i]["title"]) + ". " + ab[i]["abstract"]) for i in ids]
    cut[sp] = [ids, vals, cont, [int(y) for y in d.Y]]
    print(sp, len(ids), "mean label", np.mean(vals), flush=True)
torch.save(cut, P + "split_data")
cfg = json.load(open("configs/computer_science.json"))
cfg["default"].update({"time": [2019, 2020, 2021], "cut_time": 2021, "time_length": 3, "graph_name": "graph_sample_feature_specter2",
                       "tokenizer_type": "bert", "tokenizer_path": "/path/to/mpcc/models/hub/models--allenai--specter2_base/snapshots/3447645e1def9117997203454fa4495937bfbd83/", "fixed_num": 10 ** 9})
for k in cfg:
    if k != "default" and "graph_name" in cfg[k]:
        cfg[k]["graph_name"] = "graph_sample_feature_specter2"
json.dump(cfg, open("configs/mpcc_ai.json", "w"), indent=1)
print("PREP2 DONE", round(time.time() - t0))
