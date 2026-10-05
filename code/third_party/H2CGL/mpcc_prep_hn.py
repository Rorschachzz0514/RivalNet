"""MPCC：官方 show_hn（data_processor.py，生成 hard_negative.csv / hard_negative_labeled.csv）的向量化等价实现。
官方写法对训练论文两两做 Python 集合求交（O(n²)，n = 61,238 时约 19 亿次），无法在合理时间内跑完；
这里用稀疏矩阵乘法得到同样的四类候选：
  co_cite：与本文被同一篇（≤ 训练年发表的）论文引用的训练论文；co_ref：与本文有共同参考文献的训练论文；
  in_cite：引用本文的训练论文；in_ref：本文引用的训练论文。
引用关系只用发表年 ≤ train_time（2019）的论文（与官方相同），因此只用到截至训练快照 T 的信息；
候选只在训练论文之间产生，验证 / 测试论文不参与。
labeled 版本只保留与本文标签（<10 / 10–99 / ≥100）不同的候选（与官方相同）。
与官方的唯一差别：每格候选列表最多保留 CAP 个（超过时固定随机种子均匀抽取，保持原顺序）。
co_ref 列表动辄上万（AI 论文普遍共引经典文献），完整写入 CSV 后官方 eval() 读取不可行；
训练时每篇论文每步只从中随机取 1–2 个，截断只降低跨轮次的候选多样性，不改变抽样规则。
用法：python mpcc_prep_hn.py <data_source> [--check]   （--check：在 3000 篇子集上与官方双重循环逐项比对）
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

CAP = 200
TRAIN_TIME = 2019


def get_label(v):
    return 0 if v < 10 else (2 if v >= 100 else 1)


def official_lists(train_ids, ref_dict, valid):
    """官方 show_hn 的核心双重循环（逐字照搬，只用于小子集比对）。"""
    from collections import defaultdict
    cite_dict = defaultdict(list)
    for paper in ref_dict:
        for ref_paper in ref_dict[paper]:
            cite_dict[ref_paper].append(paper)
    all_paper_citation, all_paper_ref = {}, {}
    for paper in train_ids:
        all_paper_citation[paper] = set([cite for cite in cite_dict.get(paper, []) if cite in valid])
        all_paper_ref[paper] = set([ref for ref in ref_dict.get(paper, []) if ref in valid])
    co_cite = {p: [] for p in train_ids}; co_ref = {p: [] for p in train_ids}
    in_ref = {p: [] for p in train_ids}; in_cite = {p: [] for p in train_ids}
    n = len(train_ids)
    for i in range(n):
        paper = train_ids[i]
        cur_cite, cur_ref = all_paper_citation[paper], all_paper_ref[paper]
        for j in range(i + 1, n):
            comp_paper = train_ids[j]
            comp_cite, comp_ref = all_paper_citation[comp_paper], all_paper_ref[comp_paper]
            if cur_cite & comp_cite:
                co_cite[paper].append(comp_paper); co_cite[comp_paper].append(paper)
            if cur_ref & comp_ref:
                co_ref[paper].append(comp_paper); co_ref[comp_paper].append(paper)
            if comp_paper in cur_ref:
                in_ref[paper].append(comp_paper); in_cite[comp_paper].append(paper)
            if comp_paper in cur_cite:
                in_cite[paper].append(comp_paper); in_ref[comp_paper].append(paper)
    return {'co_cite': co_cite, 'co_ref': co_ref, 'in_cite': in_cite, 'in_ref': in_ref}


def fast_lists(train_ids, ref_dict, valid):
    """返回 {列名: [每篇训练论文的候选下标数组（升序）]}。"""
    n = len(train_ids)
    idx = {p: i for i, p in enumerate(train_ids)}
    # C[r, c] = 1：训练论文 r 被有效论文 c 引用
    rows, cols, citer = [], [], {}
    for c, refs in ref_dict.items():
        if c not in valid:
            continue
        for r in set(refs):
            if r in idx:
                rows.append(idx[r]); cols.append(citer.setdefault(c, len(citer)))
    C = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, max(len(citer), 1)))
    # R[p, r] = 1：训练论文 p 引用有效论文 r
    rows, cols, refidx = [], [], {}
    for p in train_ids:
        for r in set(ref_dict.get(p, [])):
            if r in valid:
                rows.append(idx[p]); cols.append(refidx.setdefault(r, len(refidx)))
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, max(len(refidx), 1)))
    out = {}
    for name, M in (('co_cite', C), ('co_ref', R)):
        lists = []
        MT = M.T.tocsc()
        for s in range(0, n, 2000):
            P = (M[s:s + 2000] @ MT).tocsr()
            for k in range(P.shape[0]):
                cand = P.indices[P.indptr[k]:P.indptr[k + 1]]
                cand = np.sort(cand[cand != s + k])
                lists.append(cand)
        out[name] = lists
    # in_ref：p 引用的训练论文；in_cite：引用 p 的训练论文
    in_ref = [[] for _ in range(n)]; in_cite = [[] for _ in range(n)]
    for p in train_ids:
        i = idx[p]
        for r in set(ref_dict.get(p, [])):
            j = idx.get(r)
            if j is not None and j != i:
                in_ref[i].append(j); in_cite[j].append(i)
    out['in_ref'] = [np.array(sorted(x), dtype=np.int64) for x in in_ref]
    out['in_cite'] = [np.array(sorted(x), dtype=np.int64) for x in in_cite]
    return out


if __name__ == '__main__':
    ds = sys.argv[1]
    path = './data/{}/'.format(ds)
    t0 = time.time()
    split = torch.load(path + 'split_data', weights_only=False)
    ref_dict = json.load(open(path + 'sample_ref_dict.json'))
    info = json.load(open(path + 'sample_info_dict.json'))
    valid = set(p for p in info if int(info[p]['year']) <= TRAIN_TIME)
    train_ids, train_vals = list(split['train'][0]), list(split['train'][1])
    print('train', len(train_ids), 'valid', len(valid), round(time.time() - t0), flush=True)

    if '--check' in sys.argv:
        rng = np.random.RandomState(0)
        sub = [train_ids[i] for i in sorted(rng.choice(len(train_ids), 3000, replace=False))]
        a = official_lists(sub, ref_dict, valid)
        b = fast_lists(sub, ref_dict, valid)
        for col in a:
            same = all(a[col][p] == [sub[j] for j in b[col][i]] for i, p in enumerate(sub))
            print('check', col, 'identical' if same else 'MISMATCH', 'mean len', np.mean([len(a[col][p]) for p in sub]))
        print('CHECK DONE', round(time.time() - t0)); sys.exit(0)

    lists = fast_lists(train_ids, ref_dict, valid)
    labels = np.array([get_label(v) for v in train_vals])
    pub_time = [int(info[p]['year']) for p in train_ids]
    rng = np.random.RandomState(123)
    for tag, use_label in (('hard_negative', False), ('hard_negative_labeled', True)):
        data = {'pub_time': pub_time, 'label': labels.tolist()}
        for col in ('co_cite', 'co_ref', 'in_cite', 'in_ref'):
            cells, lens, capped = [], [], 0
            for i, cand in enumerate(lists[col]):
                if use_label:
                    cand = cand[labels[cand] != labels[i]]
                lens.append(len(cand))
                if len(cand) > CAP:
                    cand = np.sort(rng.choice(cand, CAP, replace=False)); capped += 1
                cells.append(str([train_ids[j] for j in cand]))
            data[col] = cells
            print(tag, col, 'mean len', round(float(np.mean(lens)), 1), 'median', float(np.median(lens)),
                  'share empty', round(float(np.mean(np.array(lens) == 0)), 3), 'capped', capped, flush=True)
        pd.DataFrame(data, index=train_ids).to_csv(path + tag + '.csv')
    print('label dist', np.bincount(labels), 'HN DONE', round(time.time() - t0), flush=True)
