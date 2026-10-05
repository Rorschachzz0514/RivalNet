"""MPCC：把 H2CGL 的逐篇预测（mpcc_predict 输出，模型原始输出 = log1p 尺度）整理为 baselines/h2cgl/preds_{dp,all}.parquet，并做自检。
pred = max(模型输出, 0)：H2CGL 以 MSE 拟合 log(1 + y)，评估时把 exp(输出) - 1 的负值截为 0，等价于 log 尺度截到 0。
自检：① 覆盖 2020（val）/ 2021（test）全部样本论文；② dp 的 'true' 与独立从 sample_cite_year_dict 重算的语料内三年被引逐篇一致；
all 的 'true' 与 samples.parquet 的 y3 逐篇一致；③ 打印预测与标签尺度。"""
import json
import os

import numpy as np
import pandas as pd

SEED = 123
OUT = '/path/to/mpcc/baselines/h2cgl/'
os.makedirs(OUT, exist_ok=True)
s = pd.read_parquet('/path/to/mpcc/subsets/pred_v2/samples.parquet', columns=['paper_id', 'Y', 'split', 'y3'])
s = s[s.split.isin(['val', 'test'])]
cy = json.load(open('/path/to/mpcc/third_party/DPPDCC/data/mpcc_ai/sample_cite_year_dict.json'))
s['y_dp'] = [sum(int(c) for yy, c in cy.get(str(p), {}).items() if Y + 1 <= int(yy) <= Y + 3) for p, Y in zip(s.paper_id, s.Y)]
summary = {}
for tgt, ds, col in (('dp', 'mpcc_ai', 'y_dp'), ('all', 'mpcc_ai_all', 'y3')):
    parts = []
    for split in ('val', 'test'):
        f = './results/{}/mpcc_pred_{}.csv'.format(ds, split)
        if not os.path.exists(f):
            print('missing', f); continue
        d = pd.read_csv(f)
        d['split'] = split
        parts.append(d)
    if not parts:
        continue
    d = pd.concat(parts)
    m = s.merge(d, on=['paper_id', 'split'], how='left', indicator=True)
    miss = m[m._merge != 'both']
    print(tgt, 'rows', len(d), 'dup', d.duplicated(['paper_id', 'split']).sum(), 'missing', len(miss),
          miss.groupby('split').size().to_dict())
    mm = m[m._merge == 'both']
    assert (mm.year == mm.Y).all()
    lab_ok = (mm['true'].astype(int) == mm[col].astype(int))
    print(tgt, 'label check (H2CGL true == ours):', int(lab_ok.sum()), '/', len(mm))
    mm = mm.assign(pred=np.clip(mm['out'], 0, None), ly=np.log1p(mm[col]))
    res = {}
    for split in ('val', 'test'):
        x = mm[mm.split == split]
        res[split] = {'n': len(x), 'MALE': float(np.mean(np.abs(x.pred - x.ly))),
                      'MALE_noclip': float(np.mean(np.abs(x.out - x.ly))),
                      'mean_pred': float(x.pred.mean()), 'mean_log1p_y': float(x.ly.mean()),
                      'share_out_neg': float((x.out < 0).mean())}
    print(tgt, json.dumps(res, indent=1))
    summary[tgt] = {'label_check_all_equal': bool(lab_ok.all()), 'missing': len(miss), **res}
    o = pd.DataFrame({'paper_id': mm.paper_id.astype('int64'), 'split': mm.split.astype(str),
                      'seed': np.int64(SEED), 'pred': mm.pred.astype(float)})
    o.to_parquet(OUT + 'preds_{}.parquet'.format(tgt), index=False)
    print('wrote', OUT + 'preds_{}.parquet'.format(tgt), len(o))
json.dump(summary, open(OUT + 'summary.json', 'w'), indent=1)
