import os, numpy as np, pandas as pd, torch
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from scipy.stats import spearmanr
P = "Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):"
d = pd.read_parquet("/path/to/mpcc/exp1/results/_naip_parts/_input.parquet")
s = pd.read_parquet("/path/to/mpcc/subsets/pred_v2/samples.parquet", columns=["paper_id", "y3"])
d = d[d.split == "val"].merge(s, on="paper_id").sample(800, random_state=0)
tok = AutoTokenizer.from_pretrained("/path/to/data/models/NAIP_full"); tok.pad_token = tok.eos_token; tok.padding_side = "right"
texts = [P.format(title=str(t).strip(), abstract=str(a).strip()) for t, a in zip(d.title.fillna(""), d.abstract.fillna(""))]
def score(m):
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), 16):
            e = tok(texts[b:b+16], max_length=512, padding=True, truncation=True, return_tensors="pt").to("cuda")
            out.append(torch.sigmoid(m(**e).logits.float().squeeze(-1)).cpu().numpy())
    return np.concatenate(out)
m = AutoModelForSequenceClassification.from_pretrained("/path/to/data/models/NAIP_full", num_labels=1, device_map={"": 0}).eval()
m.config.pad_token_id = tok.pad_token_id
a = score(m); print("完整权重:", spearmanr(a, d.y3).statistic, a.mean(), a.std(), flush=True)
pm = PeftModel.from_pretrained(m, "/path/to/data/models/NAIP").eval()
b = score(pm); print("完整权重 + adapter:", spearmanr(b, d.y3).statistic, b.mean(), b.std(), flush=True)
print("两者相关:", spearmanr(a, b).statistic)
