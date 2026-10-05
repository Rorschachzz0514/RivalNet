"""MPCC：官方流程第 1 步（TSGCN.deal_graphs 的等价实现）。
对 split_data 中 train ∪ val ∪ test 全部论文，在第 <year> 年快照上用官方 get_paper_ego_subgraph（1 跳引用 / 被引，
各最多 100 篇，按 (时间, 被引) 截断；附作者与期刊）抽取自我中心子图，存为 checkpoints/<ds>/TSGCN_graphs_<year>.job，
索引存为 TSGCN_graphs_trans.json。唯一区别：官方只对 test 名单建索引（并假设 test ⊇ train ∪ val），这里对三个阶段的并集建索引。
可分块并行：python mpcc_prep_ego.py <year> <n_chunks> <chunk_idx>；n_chunks>1 时写 .part 文件，最后用 merge 合并。
用法：python mpcc_prep_ego.py <year> [n_chunks chunk_idx] | python mpcc_prep_ego.py merge <year> <n_chunks>
"""
import json
import os
import logging
import sys
import time

import dgl
import joblib
import torch

import our_models.TSGCN as TSGCN_module
from our_models.TSGCN import get_paper_ego_subgraph
from utilis.log_bar import TqdmToLogger
from mpcc_fast_ego import FastSingleSubgraph, canon

# 用提速等价实现替换官方 SingleSubgraph（get_paper_ego_subgraph 其余逻辑不变）；可用 verify 模式与官方实现逐项比对
OfficialSingleSubgraph = TSGCN_module.SingleSubgraph
TSGCN_module.SingleSubgraph = FastSingleSubgraph

DS = os.environ.get('MPCC_DS', 'mpcc_ai')
DATA = './data/{}/'.format(DS)
OUT = './checkpoints/{}/TSGCN_graphs'.format(DS)


def all_papers():
    split = torch.load(DATA + 'split_data', weights_only=False)
    trans = json.load(open(DATA + 'sample_node_trans.json'))['paper']
    seen, papers = set(), []
    for phase in ('test', 'val', 'train'):
        for p in split[phase][0]:
            if p not in seen:
                seen.add(p)
                papers.append(trans[p])
    return papers


def verify(year, n_rand=300):
    split = torch.load(DATA + 'split_data', weights_only=False)
    trans = json.load(open(DATA + 'sample_node_trans.json'))['paper']
    graph = dgl.load_graphs(DATA + 'graph_sample_feature_specter2_{}.dgl'.format(year))[0][0]
    oids = graph.nodes['paper'].data[dgl.NID].numpy().tolist()
    oc = dict(zip(oids, range(len(oids))))
    graph.nodes['paper'].data['citations'] = graph.in_degrees(etype='cites')
    ps = dgl.node_type_subgraph(graph, ['paper'])
    off, fast = OfficialSingleSubgraph(ps, graph, 1), FastSingleSubgraph(ps, graph, 1)
    rng = torch.Generator().manual_seed(0)
    cand = [oc.get(trans[p]) for k in ('train', 'val', 'test') for p in split[k][0]]
    sel = [cand[i] for i in torch.randperm(len(cand), generator=rng)[:n_rand].tolist()]
    deg = (ps.in_degrees(etype='cites') + ps.in_degrees(etype='is cited by')).numpy()
    sel += [int(i) for i in deg.argsort()[-30:]] + [None]  # 含被引 / 参考文献超过 100 的截断情形与空图
    bad = 0
    for p in sel:
        a, b = off.get_single_subgraph(p), fast.get_single_subgraph(p)
        for g in (a, b):
            for nt in g.ntypes:
                g.nodes[nt].data.pop('h', None)
        if canon(a) != canon(b):
            bad += 1
            ca, cb = canon(a), canon(b)
            print('MISMATCH', p, [k for k in ca if ca.get(k) != cb.get(k)])
    print('verify', year, 'checked', len(sel), 'mismatches', bad, 'max deg', int(deg.max()), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify(int(sys.argv[2]))
        sys.exit(0)
    if sys.argv[1] == 'merge':
        year, n = int(sys.argv[2]), int(sys.argv[3])
        out = []
        for i in range(n):
            out.extend(joblib.load(OUT + '_{}.part{}.job'.format(year, i)))
        papers = all_papers()
        assert len(out) == len(papers)
        joblib.dump(out, OUT + '_{}.job'.format(year))
        json.dump(dict(zip(papers, range(len(papers)))), open(OUT + '_trans.json', 'w+'))
        print('merged', year, len(out))
        sys.exit(0)
    year = int(sys.argv[1])
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    logging.basicConfig(level=logging.INFO, filename='./results/{}/ego_{}_{}.log'.format(DS, year, chunk),
                        filemode='w+', format='%(asctime)s - %(levelname)s: %(message)s', force=True)
    t0 = time.time()
    papers = all_papers()
    if n_chunks == 1 and year == 2021:  # 只由一个进程写索引，避免并行写同一文件
        json.dump(dict(zip(papers, range(len(papers)))), open(OUT + '_trans.json', 'w+'))
    size = (len(papers) + n_chunks - 1) // n_chunks
    sub = papers[chunk * size:(chunk + 1) * size]
    graph = dgl.load_graphs(DATA + 'graph_sample_feature_specter2_{}.dgl'.format(year))[0][0]
    print('loaded', year, graph.num_nodes('paper'), round(time.time() - t0), flush=True)
    res = get_paper_ego_subgraph(sub, 1, graph, batched=False, tqdm_log=TqdmToLogger(logging.getLogger(), level=logging.INFO))
    nonempty = sum(g.num_nodes('paper') > 0 for g in res)
    print('ego', year, chunk, len(res), 'nonempty', nonempty, round(time.time() - t0), flush=True)
    if n_chunks == 1:
        joblib.dump(res, OUT + '_{}.job'.format(year))
    else:
        joblib.dump(res, OUT + '_{}.part{}.job'.format(year, chunk))
    print('EGO DONE', year, chunk, round(time.time() - t0), flush=True)
