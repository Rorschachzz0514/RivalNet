# -*- coding: utf-8 -*-
"""汇总 HINTS 各 seed 的预测 -> preds_{all,dp}.parquet, 并做自检 (标签对齐 / 泄漏 / 尺度)。用 MPCC 环境运行。"""
import json, glob, sys
import numpy as np, pandas as pd

R = '/path/to/mpcc/baselines/hints'
samp = pd.read_parquet('/path/to/mpcc/subsets/pred_v2/samples.parquet', columns=['paper_id', 'split', 'Y', 'y3'])
meta = pd.read_parquet('/path/to/mpcc/exp2/tensors_v6dp/meta.parquet', columns=['paper_id'])
meta['y_dp'] = np.load('/path/to/mpcc/exp2/tensors_v6dp/y.npy').sum(1)
truth = samp.merge(meta, on='paper_id', how='left', validate='1:1')
truth['all'] = np.log1p(truth.y3)
truth['dp'] = np.log1p(truth.y_dp)
report = {}
for target in ['all', 'dp']:
    files = sorted(glob.glob(f'{R}/runs/hints_{target}_seed*.npz'))
    if not files:
        continue
    rows = []
    for f in files:
        seed = int(f.split('seed')[-1].split('.')[0])
        z = np.load(f)
        info = json.load(open(f.replace('.npz', '.json')))
        for split in ['val', 'test']:
            pid = z[f'{split}_pid']
            pred = np.maximum(z[f'{split}_pred'][:, 2], 0.0)   # log(1+累计引用) 在 t=3 处 = log1p(Y+1..Y+3 之和), 截到 >=0 (同原 cal_metric)
            rows.append(pd.DataFrame({'paper_id': pid.astype(np.int64), 'split': split, 'seed': seed, 'pred': pred.astype(float)}))
        report.setdefault(target, {})[f'seed{seed}'] = {'best_epoch': info['best']['epoch'], 'minutes': round(info['minutes'], 1),
                                                        'stopped_epoch': info['history'][-1]['epoch'] if info['history'] else None}
    df = pd.concat(rows, ignore_index=True)
    # ---- 自检 1: 覆盖 & 标签对齐 (用 samples.parquet / y.npy 原始数据按 paper_id 重新 join, 与转换脚本无关) ----
    m = df.merge(truth[['paper_id', 'split', target]], on=['paper_id', 'split'], how='left', validate='m:1')
    assert m[target].notna().all(), 'paper_id/split 对不上'
    for split, n_exp in [('val', 73400), ('test', 84566)]:
        for seed in df.seed.unique():
            n = ((df.split == split) & (df.seed == seed)).sum()
            assert n == n_exp and df[(df.split == split) & (df.seed == seed)].paper_id.is_unique, (split, seed, n)
    m['ae'] = (m.pred - m[target]).abs()
    res = m.groupby(['split', 'seed']).agg(MALE=('ae', 'mean'), mean_pred=('pred', 'mean'), mean_true=(target, 'mean')).reset_index()
    ens = m.groupby(['split', 'paper_id']).agg(pred=('pred', 'mean'), y=(target, 'first')).reset_index()
    ens['ae'] = (ens.pred - ens.y).abs()
    e = ens.groupby('split').agg(MALE=('ae', 'mean'), mean_pred=('pred', 'mean'), mean_true=('y', 'mean')).reset_index()
    # 交叉验证: npz 里存的训练标签与原始数据一致 (index_YYYY.npz 的累计标签第 3 列)
    for Y, split in [(2020, 'val'), (2021, 'test')]:
        z = np.load(f'{R}/data/index_{Y}.npz')
        cum3 = z['all_cum'][:, 2] if target == 'all' else z['dp_cum'][:, 2]
        t = truth.set_index('paper_id').loc[z['paper_id'], target].values
        assert np.allclose(np.log1p(cum3), t), 'label mismatch'
    print('=' * 20, target)
    print(res.to_string(index=False))
    print('seed-ensemble'); print(e.to_string(index=False))
    report[target]['per_seed'] = res.to_dict('records')
    report[target]['ensemble'] = e.to_dict('records')
    report[target]['val_MALE_mean_over_seeds'] = float(res[res.split == 'val'].MALE.mean())
    report[target]['test_MALE_mean_over_seeds'] = float(res[res.split == 'test'].MALE.mean())
    report[target]['val_MALE_std'] = float(res[res.split == 'val'].MALE.std())
    report[target]['test_MALE_std'] = float(res[res.split == 'test'].MALE.std())
    df.to_parquet(f'{R}/preds_{target}.parquet', index=False)
    print('saved', f'{R}/preds_{target}.parquet', df.shape, df.dtypes.to_dict())
# ---- 自检 2: 泄漏 (测试集输入用到的快照里, 最大年份) ----
st = json.load(open(f'{R}/data/stats.json'))
report['leakage'] = {f'snapshot_{y}': {k: st[str(y)][k] for k in ['max_first_public_year_of_row_papers', 'max_year_of_cited_in_corpus']}
                     for y in [2018, 2019, 2020]}
print('test(2021) 输入快照 = 2018/2019/2020:', report['leakage'])
json.dump(report, open(f'{R}/report.json', 'w'), indent=1, ensure_ascii=False, default=float)
print('EXIT_OK')
