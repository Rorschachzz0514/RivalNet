"""MPCC：为 H2CGL 准备数据目录 data/mpcc_ai（目标 dp）与 data/mpcc_ai_all（目标 all）。
- 图快照 / 节点映射 / 论文信息：符号链接到 DPPDCC 已为本语料生成的文件（只读，不改 DPPDCC 目录）。
- split_data：train = 2019 年论文、val = 2020、test = 2021（严格冷启动，与 DPPDCC 相同的论文与标签）。
  与 DPPDCC 不同，test 只放 2021 年论文（H2CGL 只在自我中心子图索引处需要 test ⊇ train ∪ val，
  mpcc_prep_ego.py 改为对三个阶段的并集建索引，算法不变）。
- dp 标签 = DPPDCC split_data 的标签（语料内 Y+1～Y+3 被引）；all 标签 = samples.parquet 的 y3（全来源 Y+1～Y+3 被引）。
"""
import os
import numpy as np
import pandas as pd
import torch

SRC = '/path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/'
LINKS = ['sample_info_dict.json', 'sample_ref_dict.json', 'sample_node_trans.json'] + \
        ['graph_sample_feature_specter2_{}.dgl'.format(y) for y in range(2017, 2022)]

d = torch.load(SRC + 'split_data', weights_only=False)
tst = [list(x) for x in zip(*[r for r in zip(*d['test']) if r[3] == 2021])]
dp = {'train': [list(x) for x in d['train']], 'val': [list(x) for x in d['val']], 'test': tst}
for k, y in (('train', 2019), ('val', 2020), ('test', 2021)):
    assert set(dp[k][3]) == {y}
print({k: (len(v[0]), sorted(set(v[3])), round(float(np.mean(v[1])), 3)) for k, v in dp.items()})

s = pd.read_parquet('/path/to/mpcc/subsets/pred_v2/samples.parquet', columns=['paper_id', 'Y', 'split', 'y3'])
y3 = dict(zip(s.paper_id.astype(str), s.y3.astype(int)))
yy = dict(zip(s.paper_id.astype(str), s.Y.astype(int)))
al = {}
for k, v in dp.items():
    ids, _, cont, yrs = v
    assert all(yy[i] == y for i, y in zip(ids, yrs))
    al[k] = [list(ids), [y3[i] for i in ids], list(cont), list(yrs)]
print('all', {k: (len(v[0]), round(float(np.mean(v[1])), 3)) for k, v in al.items()})
assert len(dp['val'][0]) == 73400 and len(dp['test'][0]) == 84566 and len(dp['train'][0]) == 61238

for ds, data in (('mpcc_ai', dp), ('mpcc_ai_all', al)):
    p = './data/{}/'.format(ds)
    os.makedirs(p, exist_ok=True)
    os.makedirs('./results/{}/'.format(ds), exist_ok=True)
    os.makedirs('./checkpoints/{}/'.format(ds), exist_ok=True)
    for f in LINKS:
        if not os.path.lexists(p + f):
            os.symlink(SRC + f, p + f)
    torch.save(data, p + 'split_data')
print('PREP DATA DONE')
