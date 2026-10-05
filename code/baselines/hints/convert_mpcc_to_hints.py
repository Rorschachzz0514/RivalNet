# -*- coding: utf-8 -*-
"""
把 MPCC 的 AI 语料 (OpenAlex) 转成 HINTS 需要的输入格式 (按 HINTS 官方 preprocess/data_aminer*.ipynb 的逻辑复刻):
  * 每年一个"individual"异质图快照 graph_y: 节点 = 当年(first_public_year==y)发表的论文 + 这些论文的所有对象
    (参考文献 P1P / 作者 P1A / 期刊会议 P1V / 关键词 P1K), 有向边 论文->对象 (行=论文), 外加自环 self。
  * 节点特征: 全局固定的 4 维随机向量 + 类型偏移 (与 data_aminernf.ipynb 一致: 关键词 rand, 作者 rand+1, venue rand+2, 论文 rand-1)。
  * 对齐: 相邻两年快照的共同节点下标 (data_aminer4.ipynb)。
  * 目标年 Y 的焦点论文的 index table: 在 Y-K..Y-1 每个快照中查找其参考文献(<=100)/作者(<=20)/venue(1)/关键词(<=15)
    的节点下标, 找不到的跳过, 不足补 -1 (data_aminer3.ipynb 的 construct_index_from_one_df)。
  * 标签: 累计引用 log(1+cum) 的 3 年序列 (官方为 5 年; 我们只有 Y+1..Y+3)。
只用 MPCC 环境 (duckdb/pandas/numpy) 运行; 输出为 npz (避免 scipy 版本间 pickle 不兼容)。
"""
import os, time, json
import numpy as np, pandas as pd, duckdb

OUT = '/path/to/mpcc/baselines/hints/data'
os.makedirs(OUT, exist_ok=True)
K = 3                       # 快照个数 (官方 5; 语料从 2016 年开始, 见 README)
SNAP_YEARS = list(range(2016, 2021))   # 2016..2020
TARGET_YEARS = {2019: 'train', 2020: 'val', 2021: 'test'}
MAXLEN = {'P': 100, 'A': 20, 'V': 1, 'K': 15}
TYPE_CODE = {'P': 1, 'A': 2, 'V': 3, 'K': 4}
OFF = {'P': -1.0, 'A': 1.0, 'V': 2.0, 'K': 0.0}
BASE = np.int64(10) ** 11   # OpenAlex 数字 id < 1e10 (已检查)
t0 = time.time()


def log(*a):
    print('[%.0fs]' % (time.time() - t0), *a, flush=True)


con = duckdb.connect()
con.execute("SET threads=32")
P = "read_parquet('/path/to/mpcc/subsets/exp0_v2/papers.parquet')"
S = "read_parquet('/path/to/mpcc/subsets/pred_v2/samples.parquet')"
pap = con.execute(f"""select p.paper_id, p.first_public_year fy, p.published_year py, p.source_id,
        s.paper_id is not null as in_samp, s.source_at_T,
        p.refs, p.author_ids, p.concept_names
     from {P} p left join {S} s using(paper_id)
     where p.first_public_year between 2016 and 2021""").df()
log('papers', len(pap))
# venue at end of the paper's own year (protocol: info at T):
#   样本论文用 samples.source_at_T; 其他论文: published_year<=first_public_year 时用 source_id, 否则 (当年只是预印本) 无 venue
pap['venue'] = np.where(pap.in_samp, pap.source_at_T,
                        np.where(pap.py <= pap.fy, pap.source_id, np.nan))
year_of = pd.Series(pap.fy.values, index=pap.paper_id.values)
# concept (keyword) name -> int code
allc = pd.Series([c for lst in pap.concept_names if lst is not None for c in lst]).unique()
cmap = {c: i for i, c in enumerate(sorted(allc))}
log('concepts', len(cmap))


def explode(df, col, typ):
    """返回 (row_idx_in_df, pos, key int64); key = type*1e11 + id; 保留原顺序"""
    lists = [x if x is not None else [] for x in df[col].values]
    lens = np.array([len(x) for x in lists])
    ridx = np.repeat(np.arange(len(df)), lens)
    pos = np.concatenate([np.arange(l) for l in lens]) if lens.sum() else np.zeros(0, int)
    vals = [v for x in lists for v in x]
    if typ == 'K':
        ids = np.array([cmap[v] for v in vals], dtype=np.int64)
    else:
        ids = np.array(vals, dtype=np.int64)
    return ridx, pos, ids + TYPE_CODE[typ] * BASE


def relations(df):
    """论文 df 的四种关系 (ridx,pos,key)"""
    rel = {}
    r, p, k = explode(df, 'refs', 'P')
    ref_ids = k - TYPE_CODE['P'] * BASE
    # 去掉"引用未来论文"的边: 被引论文在语料中且 first_public_year > 引用论文所在年
    ry = year_of.reindex(ref_ids).values
    citer_y = df.fy.values[r]
    keep = ~(ry > citer_y)          # NaN (语料外) 保留
    keep &= ref_ids != df.paper_id.values[r]
    rel['P'] = (r[keep], p[keep], k[keep])
    rel['A'] = explode(df, 'author_ids', 'A')
    v = df.venue.values.astype(float)
    ok = ~np.isnan(v)
    rel['V'] = (np.where(ok)[0], np.zeros(ok.sum(), int), v[ok].astype(np.int64) + TYPE_CODE['V'] * BASE)
    rel['K'] = explode(df, 'concept_names', 'K')
    return rel


# ---------- 1. 年度快照 ----------
snap_keys = {}
stats = {}
for y in SNAP_YEARS:
    df = pap[pap.fy == y].reset_index(drop=True)
    rel = relations(df)
    pkeys = df.paper_id.values.astype(np.int64) + TYPE_CODE['P'] * BASE
    keys = np.unique(np.concatenate([pkeys] + [rel[t][2] for t in 'PAVK']))
    snap_keys[y] = keys
    out = {'n': np.int64(len(keys)), 'keys': keys}
    prow = np.searchsorted(keys, pkeys)
    for t in 'PAVK':
        r, _, k = rel[t]
        e = np.unique(np.stack([prow[r], np.searchsorted(keys, k)], 1), axis=0)  # 去重
        out['P1' + t + '_row'] = e[:, 0].astype(np.int32)
        out['P1' + t + '_col'] = e[:, 1].astype(np.int32)
    stats[y] = {'papers': int(len(df)), 'nodes': int(len(keys)),
                **{'P1' + t: int(len(out['P1' + t + '_row'])) for t in 'PAVK'},
                'max_first_public_year_of_row_papers': int(df.fy.max()),
                'max_year_of_cited_in_corpus': float(np.nanmax(year_of.reindex(rel['P'][2] - TYPE_CODE['P'] * BASE).values))}
    np.savez(f'{OUT}/graph_{y}_raw.npz', **out)
    log('snapshot', y, stats[y])

# ---------- 2. 全局随机特征 (固定种子) ----------
allkeys = np.unique(np.concatenate(list(snap_keys.values())))
rs = np.random.RandomState(2021)
feat_all = rs.rand(len(allkeys), 4)
typ = allkeys // BASE
for t, c in TYPE_CODE.items():
    feat_all[typ == c] += OFF[t]
for y in SNAP_YEARS:
    d = dict(np.load(f'{OUT}/graph_{y}_raw.npz'))
    d['feature'] = feat_all[np.searchsorted(allkeys, d['keys'])].astype(np.float32)
    np.savez(f'{OUT}/graph_{y}.npz', **d)
    os.remove(f'{OUT}/graph_{y}_raw.npz')
log('features done', len(allkeys))

# ---------- 3. 对齐 ----------
for y in SNAP_YEARS[:-1]:
    _, i1, i2 = np.intersect1d(snap_keys[y], snap_keys[y + 1], assume_unique=True, return_indices=True)
    np.savez(f'{OUT}/align_{y}.npz', ind1=i1.astype(np.int32), ind2=i2.astype(np.int32))
    stats[f'align_{y}'] = int(len(i1))
log('alignment done')

# ---------- 4. 焦点论文 index table + 标签 ----------
samp = con.execute(f"select paper_id, Y, split, y1, y2, y3 from {S} order by paper_id").df()
meta = pd.read_parquet('/path/to/mpcc/exp2/tensors_v6dp/meta.parquet')
ydp = np.load('/path/to/mpcc/exp2/tensors_v6dp/y.npy').astype(np.float64)
dp = pd.DataFrame(ydp, columns=['d1', 'd2', 'd3'])
dp['paper_id'] = meta.paper_id.values
samp = samp.merge(dp, on='paper_id', how='left', validate='1:1')
assert samp[['d1', 'd2', 'd3']].notna().all().all()
pap_i = pap.set_index('paper_id')
for Y, split in TARGET_YEARS.items():
    s = samp[(samp.Y == Y) & (samp.split == split)].reset_index(drop=True)
    df = pap_i.loc[s.paper_id.values].reset_index()
    assert (df.paper_id.values == s.paper_id.values).all()
    assert (df.fy.values == Y).all()
    rel = relations(df)
    n = len(df)
    res = {}
    for t in 'PAVK':
        r, p, k = rel[t]
        arr = np.full((K, n, MAXLEN[t]), -1, dtype=np.int32)
        for j in range(K):
            y = Y - K + j
            keys = snap_keys[y]
            loc = np.searchsorted(keys, k)
            loc[loc >= len(keys)] = 0
            found = keys[loc] == k
            rr, ll = r[found], loc[found]            # 保持原顺序 (r 递增, 同一论文内按原列表顺序)
            rank = pd.Series(rr).groupby(rr).cumcount().values   # 只保留"找得到的"前 maxlen 个 (同官方 try/except)
            m = rank < MAXLEN[t]
            arr[j, rr[m], rank[m]] = ll[m]
        res[t] = arr
        stats[f'idx_{Y}_{t}_found_frac_per_snapshot'] = [float((arr[j] >= 0).any(1).mean()) for j in range(K)]
    all_cum = np.stack([s.y1.values, (s.y1 + s.y2).values, s.y3.values], 1)
    dp_cum = np.cumsum(s[['d1', 'd2', 'd3']].values, 1)
    np.savez(f'{OUT}/index_{Y}.npz', P=res['P'], A=res['A'], V=res['V'], K=res['K'],
             paper_id=s.paper_id.values.astype(np.int64), all_cum=all_cum, dp_cum=dp_cum)
    stats[f'n_{Y}'] = n
    log('index', Y, split, n)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
log('ALL DONE')
