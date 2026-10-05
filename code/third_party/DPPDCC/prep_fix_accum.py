"""MPCC：语料内零被引的论文不在 sample_cite_year_dict 里，因此 get_citation_accum 没有为它们写累计被引序列，训练时 KeyError。
补上：序列中 -1（发表前）的位置只取决于发表年份，取同年已有论文的模板，其余位置全为 0。原文件备份为 sample_citation_accum.orig.json。"""
import json, torch
P = "./data/mpcc_ai/"
a = json.load(open(P + "sample_citation_accum.orig.json"))
info = json.load(open(P + "sample_info_dict.json"))
split = torch.load(P + "split_data", weights_only=False)
tmpl = {}
for p, v in a.items():
    tmpl.setdefault(int(info[p]["year"]), [-1 if x == -1 else 0 for x in v])
add = 0
for k in split:
    for p in split[k][0]:
        if p not in a:
            a[p] = list(tmpl[int(info[p]["year"])]); add += 1
assert all(len(v) == len(next(iter(a.values()))) for v in a.values())
json.dump(a, open(P + "sample_citation_accum.json", "w"))
print("added", add, "templates", {y: t for y, t in sorted(tmpl.items()) if y >= 2017})
