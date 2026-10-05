"""MPCC：官方 SingleSubgraph.get_single_subgraph（our_models/TSGCN.py）的提速等价实现。
官方实现在每篇论文上对整张快照图调用 dgl.in_subgraph / in_edges / node_subgraph，DGL 2.x 下每次都要扫描全图的 COO 边表
（本语料约 0.5–0.8 秒 / 篇，40 万个自我中心子图需数十小时）。这里预先建好 CSR / CSC 索引，按完全相同的规则选点：
  参考文献 = 'is cited by' 入边的源点，施引 = 'cites' 入边的源点；超过 100 篇时用同样的 pandas 排序（时间、被引降序）截断；
  作者 / 期刊 = 写作 / 发表这些论文的全部作者 / 期刊；子图 = 这些点在 (paper, journal, author) 图上的诱导子图。
输出图的点集、边集（含原边 ID）、点 / 边特征以及 is_ref / is_cite / is_target 与官方实现逐项相同（见 verify），
只有子图内部的点 / 边编号顺序可能不同（图神经网络对此不变）。
"""
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


class FastSingleSubgraph:
    def __init__(self, paper_subgraph, origin_graph, hop, citation='global'):
        self.paper_subgraph = paper_subgraph
        self.origin_graph = dgl.node_type_subgraph(origin_graph, ntypes=['paper', 'journal', 'author'])
        self.hop = 1
        self.max_nodes = 100
        self.citations = self.paper_subgraph.nodes['paper'].data['citations']
        self.time = self.paper_subgraph.nodes['paper'].data['time']
        g = self.origin_graph
        self.n = {nt: g.num_nodes(nt) for nt in g.ntypes}
        # 每种边：按 (dst, eid) 排序的入边表（CSC）与按 (src, eid) 排序的出边表（CSR），保留原边 ID
        self.csc, self.csr = {}, {}
        for et in g.canonical_etypes:
            src, dst, eid = [x.numpy() for x in g.edges(form='all', etype=et, order='eid')]
            st, _, dt = et
            o = np.lexsort((eid, dst))
            self.csc[et] = (np.searchsorted(dst[o], np.arange(self.n[dt] + 1)), src[o], eid[o])
            o = np.lexsort((eid, src))
            self.csr[et] = (np.searchsorted(src[o], np.arange(self.n[st] + 1)), dst[o], eid[o])
        self.ndata = {nt: {k: v for k, v in g.nodes[nt].data.items() if k != 'h'} for nt in g.ntypes}
        self.edata = {et: dict(g.edges[et].data) for et in g.canonical_etypes}
        self.etypes = g.canonical_etypes
        self.idtype = g.idtype

    def _in_src(self, et, nodes):
        ptr, src, _ = self.csc[et]
        return np.concatenate([src[ptr[v]:ptr[v + 1]] for v in nodes]) if len(nodes) else np.zeros(0, np.int64)

    def _truncate(self, papers):
        if len(papers) > self.max_nodes:
            temp_df = pd.DataFrame(data={'id': papers, 'citations': self.citations[papers].numpy(),
                                         'time': self.time[papers].squeeze(dim=-1).numpy()})
            papers = list(temp_df.sort_values(by=['time', 'citations'], ascending=False).head(self.max_nodes)['id'].tolist())
        return papers

    def _induced(self, nodes):
        num_nodes = {nt: len(nodes[nt]) for nt in self.n}
        pos = {}
        for nt, ids in nodes.items():
            m = np.full(self.n[nt], -1, dtype=np.int64)
            m[ids] = np.arange(len(ids))
            pos[nt] = m
        rel, eids = {}, {}
        for et in self.etypes:
            st, _, dt = et
            ptr, dst, eid = self.csr[et]
            s_ids = nodes[st]
            if len(s_ids):
                cnt = ptr[s_ids + 1] - ptr[s_ids]
                idx = np.concatenate([np.arange(ptr[v], ptr[v + 1]) for v in s_ids])
                src_rep = np.repeat(s_ids, cnt)
                keep = pos[dt][dst[idx]] >= 0
                e, s, d = eid[idx][keep], src_rep[keep], dst[idx][keep]
                o = np.argsort(e, kind='stable')
                e, s, d = e[o], s[o], d[o]
            else:
                e = s = d = np.zeros(0, np.int64)
            rel[et] = (torch.from_numpy(pos[st][s]), torch.from_numpy(pos[dt][d]))
            eids[et] = torch.from_numpy(e)
        g = dgl.heterograph(rel, num_nodes_dict=num_nodes, idtype=self.idtype)
        for nt, ids in nodes.items():
            t = torch.from_numpy(ids)
            for k, v in self.ndata[nt].items():
                g.nodes[nt].data[k] = v[t]
            g.nodes[nt].data[dgl.NID] = t
        for et in self.etypes:
            for k, v in self.edata[et].items():
                g.edges[et].data[k] = v[eids[et]]
            g.edges[et].data[dgl.EID] = eids[et]
        return g

    def get_single_subgraph(self, paper):
        if paper is not None:
            ref_hop_papers = self._truncate(self._in_src(('paper', 'is cited by', 'paper'), [paper]).tolist())
            cite_hop_papers = self._truncate(self._in_src(('paper', 'cites', 'paper'), [paper]).tolist())
            k_hop_papers = list(set(ref_hop_papers + cite_hop_papers + [paper]))
            authors = np.unique(self._in_src(('author', 'writes', 'paper'), k_hop_papers))
            journals = np.unique(self._in_src(('journal', 'publishes', 'paper'), k_hop_papers))
            cur_graph = self._induced({'paper': np.array(k_hop_papers, dtype=np.int64),
                                       'journal': journals.astype(np.int64), 'author': authors.astype(np.int64)})
            ref_set = set(ref_hop_papers)
            cite_set = set(cite_hop_papers)
            cur_graph.nodes['paper'].data['is_ref'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_cite'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            cur_graph.nodes['paper'].data['is_target'] = torch.zeros(len(k_hop_papers), dtype=torch.long)
            oids = cur_graph.nodes['paper'].data[dgl.NID].numpy().tolist()
            id_trans = dict(zip(oids, range(len(oids))))
            ref_idx = [id_trans[p] for p in ref_set]
            cite_idx = [id_trans[p] for p in cite_set]
            cur_graph.nodes['paper'].data['is_ref'][ref_idx] = 1
            cur_graph.nodes['paper'].data['is_cite'][cite_idx] = 1
            cur_graph.nodes['paper'].data['is_target'][id_trans[paper]] = 1
        else:
            cur_graph = self._induced({'paper': np.zeros(0, np.int64), 'journal': np.zeros(0, np.int64),
                                       'author': np.zeros(0, np.int64)})
            cur_graph.nodes['paper'].data['is_ref'] = torch.zeros(0, dtype=torch.long)
            cur_graph.nodes['paper'].data['is_cite'] = torch.zeros(0, dtype=torch.long)
            cur_graph.nodes['paper'].data['is_target'] = torch.zeros(0, dtype=torch.long)
        return cur_graph


def canon(g):
    """把子图转成与编号顺序无关的规范形式，用于和官方实现逐项比对。"""
    out = {'ntypes': g.ntypes, 'etypes': g.canonical_etypes, 'idtype': str(g.idtype)}
    for nt in g.ntypes:
        nid = g.nodes[nt].data[dgl.NID]
        o = torch.argsort(nid)
        out[nt] = {k: (str(v.dtype), tuple(v.shape[1:]), v[o].tolist()) for k, v in g.nodes[nt].data.items()}
    for et in g.canonical_etypes:
        st, _, dt = et
        s, d = g.edges(etype=et)
        eid = g.edges[et].data[dgl.EID]
        o = torch.argsort(eid)
        sn, dn = g.nodes[st].data[dgl.NID][s], g.nodes[dt].data[dgl.NID][d]
        out[et] = {'edges': list(zip(sn[o].tolist(), dn[o].tolist())),
                   **{k: (str(v.dtype), v[o].tolist()) for k, v in g.edges[et].data.items()}}
    return out
