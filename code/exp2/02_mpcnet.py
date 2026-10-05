"""
【02】实验 2 / 3 · MPC-Net 与消融：训练、验证集早停、验证 / 测试集预测（多 GPU 并行）

用途
  MPC-Net（预先登记见 实验2说明.md 第 3 节）：μ_{i,t} = Â_{g,t} × s_{i,t}，t = 第 1、2、3 年。
    ① 需求塔：格子特征 → 3 年 log 总量；损失 = 格子真实总被引的负二项 NLL（按格子数平均；偏置初始化为训练格子平均 log 总量）
    ② 论文编码器：SPECTER2 + 论文特征 → h（256 维）
    ③ 竞争模块：50 个对手（向量 + 关系特征）编码后，以 h 为查询做 4 头交叉注意力（另有一个可学习的"空"键，
       允许模型不看任何对手）→ z；对手效应不限符号
    ⑥ 份额头：[h, z] → 每年一个 logit，格内 softmax → 份额 p；格子浓度 α（由格子特征预测）；
       损失 = 以格子真实总数为条件的 Dirichlet-多项 NLL
  点预测：蒙特卡洛（32 次）——总数 ~ 负二项(Â)，份额 ~ Dirichlet(α·p)，论文被引 ~ Poisson(总数 × 份额)，
          pred = E[log1p(三年被引之和)]（与回归 log1p 的对比方法同口径）。
  消融（实验 3）：
    A1_direct          去掉分解：[h, z, 格子嵌入] → 直接输出逐年 log 均值，负二项 NLL
    A2_oracle_demand   需求塔换成格子真实总数（训练与预测都用；上帝视角）
    A3_no_rivals       去掉竞争模块（z = 0）
    A7_multinomial     Dirichlet-多项 → 多项（无 α）
    A11_meanpool       交叉注意力 → 对手编码的平均
    A13_mse            负二项 / DM 损失 → 对 log1p(μ) 与 log1p(y) 的均方误差（逐年）
    A17_coupled        只保留文献耦合对手（其余掩码）
    A17_top10          只保留前 10 名对手
  训练：AdamW（lr 1e-3 余弦退火，wd 1e-4），按格子组批（每批约 3,000 篇），最多 40 轮，验证集 MALE 连续 5 轮不降则停。
  需求塔 = 线性主干（便于对训练期之外的格子外推）+ 0.1 × MLP 残差（v2 迭代，见 实验2说明.md 第 7 节）。

输入
  config_exp2.TENSORS（【01】）、config_exp2.EMB

输出（runs/<配置>_s<种子>/）
  preds.parquet     paper_id, split, model, seed, pred（E[log1p y3]）, mu1–mu3（均值）, share_y3（格内份额）, A_y3（格子预测总量）
  attn_test.npy     测试集每篇论文对 [空, 对手 1..50] 的注意力（只在 full 配置保存，实验 4 用）
  log.csv           每轮训练损失与验证 MALE
  model.pt          最优权重

用法
  python 02_mpcnet.py --configs full --seeds 1 --gpus 0                     单个
  python 02_mpcnet.py --configs full,A1_direct,... --seeds 1,2,3,4,5 --gpus 0,1,2   任务轮流分配到各 GPU
  python 02_mpcnet.py --configs MPCNet5_l1 --seeds 1,2,3,4,5 --gpus 0,1 --placebo shuffle_cell
        实验 5-3：加载已训练权重，把测试集的竞争集合换成同格子另一篇论文的，输出 preds_placebo_<模式>.parquet
  python 02_mpcnet.py --configs MPCNet5_l1 --seeds 1,2,3,4,5 --gpus 4 --placebo cf:twins
        修改计划 A2：加载已训练权重，把 cf/twins.parquet 中列出的 (focal_id, comp_id) 对手从焦点论文的竞争集合中屏蔽，
        重新预测验证集与测试集，输出 preds_cf_twins.parquet（反事实：假如这个对手不存在）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_exp2 as C

CONFIGS = {
    "full": {}, "A1_direct": {"direct": True}, "A2_oracle_demand": {"oracle": True}, "A3_no_rivals": {"rivals": "none"},
    "A7_multinomial": {"share": "mult"}, "A11_meanpool": {"rivals": "mean"}, "A13_mse": {"loss": "mse"},
    "A17_coupled": {"mask": "coupled"}, "A17_top10": {"mask": "top10"},
    # 两阶段（设计文档 5.4 节）：份额头以真实格子总量为条件训练；预测时用第一阶段（LightGBM，只用训练格子拟合）的总量
    "two_stage": {"oracle": True, "stage1": True},
    # v4（log 空间分解）：log1p(ŷ3) = 格子水平 m_g（需求塔）+ 格内中心化的份额 logit；损失 = MSE + λ·DM 份额损失
    "logshare_l0": {"logshare": 0.0}, "logshare_l001": {"logshare": 0.01}, "logshare_l01": {"logshare": 0.1},
    "logshare_l03": {"logshare": 0.3}, "logshare_l1": {"logshare": 1.0},
}
# 定版（v4，验证集选择 λ = 0.3）与消融（实验 3）。原"计数空间 总量 × 份额"设计作为 A13_count_space 保留对照
FINAL = {
    "MPCNet": {"logshare": 0.3},
    "A1_direct": {"logshare": 0.3, "direct_log": True},
    "A2_oracle_level": {"logshare": 0.3, "level_oracle": True},
    "A3_no_rivals": {"logshare": 0.3, "rivals": "none"},
    "A7_no_share_loss": {"logshare": 0.0},
    "A11_meanpool": {"logshare": 0.3, "rivals": "mean"},
    "A13_count_space": {},
    "A17_coupled": {"logshare": 0.3, "mask": "coupled"},
    "A17_top10": {"logshare": 0.3, "mask": "top10"},
    "A18_no_text": {"logshare": 0.3, "no_text": True},
}
CONFIGS.update(FINAL)
# 定版 v5（验证集上"直接输出 + 格子需求上下文 + 份额辅助损失"优于显式分解：0.692 vs 0.702）及其消融
V5 = {
    "MPCNet5": {"logshare": 0.3, "direct_log": True},
    "MPCNet5_l1": {"logshare": 1.0, "direct_log": True},
    "B_decomp": {"logshare": 0.3},
    "B_no_cell": {"logshare": 0.3, "direct_log": True, "no_cell": True},
    "B_no_rivals": {"logshare": 0.3, "direct_log": True, "rivals": "none"},
    "B_no_share_loss": {"logshare": 0.0, "direct_log": True},
    "B_meanpool": {"logshare": 0.3, "direct_log": True, "rivals": "mean"},
    "B_coupled": {"logshare": 0.3, "direct_log": True, "mask": "coupled"},
    "B_top10": {"logshare": 0.3, "direct_log": True, "mask": "top10"},
    "B_no_text": {"logshare": 0.3, "direct_log": True, "no_text": True},
}
CONFIGS.update(V5)
# 发表 1 年后（实验 7-1 暴露的需求）：目标 = 第 2、3 年被引之和；论文输入加第 1 年与发表当年被引；格子输入加格内第 1 年被引的均值与总量
# 最终检查：验证集上的小规模超参数搜索（只看验证集）
CONFIGS.update({
    "HP_d512": {"logshare": 1.0, "direct_log": True, "d": 512},
    "HP_lr5e-4": {"logshare": 1.0, "direct_log": True, "lr": 5e-4},
    "HP_d512_lr5e-4": {"logshare": 1.0, "direct_log": True, "d": 512, "lr": 5e-4},
    "HP_drop0.3": {"logshare": 1.0, "direct_log": True, "dropout": 0.3},
})
# v6（第六轮迭代，只看验证集）：在定版 MPCNet5_l1 上逐项加改动
_B = {"logshare": 1.0, "direct_log": True}
CONFIGS.update({
    "V6_base_t6": {**_B, "t6": True},                                   # v6a：+ 参考文献质量特征
    "V6_cat": {**_B, "t6": True, "cat_emb": True},                      # v6b：+ 渠道 / 方向编号
    "V6_aux": {**_B, "t6": True, "aux_years": True},                    # v6c：+ 逐年辅助目标
    "V6_self": {**_B, "t6": True, "self_attn": True},                   # v6d：+ 对手之间自注意力
    "V6_all": {**_B, "t6": True, "cat_emb": True, "aux_years": True, "self_attn": True},
    "V6_aux_self": {**_B, "t6": True, "aux_years": True, "self_attn": True},   # 去掉在验证集上有害的类别编号
})
CONFIGS.update({
    "MPCNet5_after1y": {"logshare": 1.0, "direct_log": True, "after1y": True},
    "B_no_rivals_after1y": {"logshare": 1.0, "direct_log": True, "after1y": True, "rivals": "none"},
})


def stage1_demand(seed):
    """第一阶段需求塔：log(格子第 t 年总被引 + 1) = log(n_cell) + LightGBM(格子特征) 预测的人均水平；只用训练格子拟合。
    返回全部格子的 logÂ (C, 3) 与训练格子上的残差标准差 (3,)，预测时按对数正态抽样总量。"""
    import lightgbm as lgb
    L = lambda n: np.load(os.path.join(C.TENSORS, n))
    X, T, sp, ptr = L("cell_x.npy"), L("cell_tot.npy"), L("cell_split.npy"), L("cell_ptr.npy")
    n = np.diff(ptr).astype(float)
    tr = sp == "train"
    out, sd = np.zeros((len(n), 3)), np.zeros(3)
    for t in range(3):
        y = np.log(T[:, t] + 1)
        m = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=seed),
                      lgb.Dataset(X[tr], (y - np.log(n))[tr], weight=n[tr]), num_boost_round=600)
        out[:, t] = np.log(n) + m.predict(X)
        sd[t] = np.sqrt(np.average((out[tr, t] - y[tr]) ** 2, weights=n[tr]))
    return out, sd


def run(cfg_name, seed, gpu, max_epochs=40, patience=5, tag="", placebo=None, train_frac=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    if cfg_name in CONFIGS:
        cfg = CONFIGS[cfg_name]
    else:                                                  # 自动迭代（05_autosearch.py）生成的配置
        cfg = json.load(open(os.path.join(C.HERE, "autosearch", "configs", cfg_name + ".json")))
    patience = cfg.get("patience", patience)
    out_dir = os.path.join(C.RUNS, f"{cfg_name}{tag}_s{seed}")
    if placebo is None and os.path.exists(os.path.join(out_dir, "preds.parquet")):
        print(f"[{cfg_name} s{seed}] 已完成，跳过", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda")
    TD = C.TENSORS + cfg.get("tdir", "_v6" if cfg.get("t6") else "")   # 张量目录：v6 / v7 / v7k100 ...                  # v6 张量：多了参考文献质量特征与类别编号
    L = lambda n: np.load(os.path.join(TD, n), allow_pickle=False)
    meta = pd.read_parquet(os.path.join(TD, "meta.parquet"))
    E = torch.from_numpy(np.load(C.EMB)).to(dev)                                   # (732k, 768) fp16
    xp = torch.from_numpy(L("x_paper.npy")).to(dev)
    rrow = torch.from_numpy(L("r_row.npy")).to(dev)
    rf = torch.from_numpy(L("r_feat.npy")).to(dev)
    y = torch.from_numpy(L("y.npy")).to(dev)
    erow = torch.from_numpy(meta.emb_row.to_numpy()).to(dev)
    cx = torch.from_numpy(L("cell_x.npy")).to(dev)
    ctot = torch.from_numpy(L("cell_tot.npy")).to(dev)
    csplit = L("cell_split.npy")
    ptr = L("cell_ptr.npy")
    if cfg.get("sim_shift"):
        # 模拟"时间往前挪一年"：2017–2018 训练、2019 验证（早停）、2020 当作测试，用来检验"训练 + 验证一起重训"是否有益
        # （注：张量的标准化统计量来自 2017–2019，含 2019，只影响均值 / 方差，可忽略）
        cyear = meta.Y.to_numpy()[ptr[:-1]]
        csplit = np.where(cyear <= 2018, "train", np.where(cyear == 2019, "val", np.where(cyear == 2020, "test", "none")))
    if cfg.get("shuffle_train_y"):
        # 代码自查 C6：只在训练集内部随机打乱被引（格子总量随之重算）；若代码无泄漏，验证误差应退回"只猜平均"的水平
        tr_idx = np.where((meta.split == "train").to_numpy())[0]
        perm = torch.from_numpy(np.random.default_rng(777).permutation(tr_idx)).to(dev)
        y = y.clone()
        y[torch.from_numpy(tr_idx).to(dev)] = y[perm]
        ctot = torch.from_numpy(np.add.reduceat(y.cpu().numpy(), ptr[:-1], axis=0)).to(dev)
    if cfg.get("after1y"):
        # T' = 首次公开年份 Y+1 的年底：已知第 1 年（Y+1）与发表当年（Y）的被引；目标只剩第 2、3 年
        sm = pd.read_parquet(C.SAMPLES, columns=["paper_id", "y0", "y1"]).set_index("paper_id").reindex(meta.paper_id)
        ly = np.log1p(sm[["y1", "y0"]].to_numpy(np.float64))
        trm = (meta.split == "train").to_numpy()
        ly = (ly - ly[trm].mean(0)) / ly[trm].std(0)
        xp = torch.cat([xp, torch.from_numpy(ly.astype(np.float32)).to(dev)], 1)
        y1np = sm.y1.to_numpy(np.float64)
        cm = np.add.reduceat(np.log1p(y1np), ptr[:-1]) / np.diff(ptr)
        ct1 = np.log1p(np.add.reduceat(y1np, ptr[:-1]))
        cadd = np.stack([cm, ct1], 1)
        ctr_ = csplit == "train"
        cadd = (cadd - cadd[ctr_].mean(0)) / cadd[ctr_].std(0)
        cx = torch.cat([cx, torch.from_numpy(cadd.astype(np.float32)).to(dev)], 1)
        y = y.clone()
        y[:, 0] = 0                                        # 第 1 年已知，不再是目标
        ctot = ctot.clone()
        ctot[:, 0] = 0
    mask_all = torch.zeros(rrow.shape, dtype=torch.bool, device=dev)
    if cfg.get("mask") == "coupled":
        mask_all = ~torch.from_numpy(L("r_coupled.npy")).to(dev)
    elif cfg.get("mask") == "top10":
        mask_all = torch.from_numpy(L("r_rank.npy")).to(dev) > 10
    tpart = cfg.get("tpart", 0)
    if tpart:
        # 修改计划 A3：按时间划分双通道看到的对手（解决第七轮双通道不可识别、"替代"通道塌缩的问题）
        #   S（同期）= 发表不早于焦点论文 tpart 天之前（含晚于焦点论文的）→ 第一个通道（"抢先"，效应 ≤ 0）
        #   其余（更早的前作）→ 第二个通道（"带火"，效应 ≥ 0）；天数缺失时按年份差：同年 → S
        #   对手特征列：3 = dyear_0，4 = ddays_yr（对手日期 − 焦点日期，/365，缺失填 0），5 = ddays_na
        _na, _y0 = rf[:, :, 5] > 0.5, rf[:, :, 3] > 0.5
        S_all = torch.where(_na, _y0, rf[:, :, 4].float() >= -tpart / 365.0)
    dp, dr, dc, d = xp.shape[1], rf.shape[2], cx.shape[1], cfg.get("d", 256)
    if cfg.get("cat_emb"):
        cat_sizes = json.load(open(os.path.join(TD, "cat_sizes.json")))
        cat_t = {c: torch.from_numpy(L(f"cat_{c}.npy")).to(dev) for c in cat_sizes}
    else:
        cat_sizes, cat_t = {}, {}
    dce = 32 * len(cat_sizes)
    use_cell = (cfg.get("direct") or cfg.get("direct_log")) and not cfg.get("no_cell")
    # ---------- 第七轮（由研究设想推出的结构，见 实验2说明.md 第 8 节；默认全部关闭，旧配置结果不变）
    #   rel_rival  相对竞争位置：对手编码里加入 [对手 − 自己, 对手 ⊙ 自己]（竞争取决于和对手相比的强弱，而非对手的绝对强弱）
    #   two_chan   双通道竞争："替代"（抢引用，效应 ≤ 0）与"互补 / 可见度"（带引用，效应 ≥ 0）两个注意力通道；
    #              "strict" = 两个通道只通过带符号的标量效应进入预测（可解释的分解）；"plus" = 通道向量也进入预测头
    #   luce       吸引力份额（Luce / 多项 logit 市场份额模型）：显式项 −γ·log(1 + Σ_j σ_ij·exp(q_j − q_i))，
    #              q = 吸引力，σ_ij = 可替代程度（由相似度、年份差、文献耦合等关系特征决定），γ ≥ 0
    #   cohort     同期竞争层：同一格子（同子课题、同一年）内的论文互相注意（它们分的是同一份需求）
    #   xdom       跨域需求：除自身子课题外，再看向量最接近的 xdom 个其他子课题同年的需求，按相似度加权（论文同时参与多个方向的竞争）
    two_chan, luce, n_coh, xd = cfg.get("two_chan"), cfg.get("luce"), cfg.get("cohort", 0), cfg.get("xdom", 0)
    if xd:
        # 子课题中心 = 训练期论文 SPECTER2 向量的均值（只用文本，无未来信息）；每篇论文取最接近的 xd 个其他子课题、同一年的格子
        cl = torch.from_numpy((meta.cell.to_numpy() // 10000).astype(np.int64)).to(dev)
        yr = torch.from_numpy(meta.Y.to_numpy().astype(np.int64)).to(dev)
        trm_ = torch.from_numpy((meta.split == "train").to_numpy()).to(dev)
        En = F.normalize(E[erow].float(), dim=1)
        ncl = int(cl.max()) + 1
        cen = torch.zeros(ncl, 768, device=dev).index_add_(0, cl[trm_], En[trm_])
        cen = F.normalize(cen, dim=1)
        xd_sim, xd_cl = [], []
        for b in range(0, len(cl), 50000):
            s_ = En[b:b + 50000] @ cen.T
            s_[torch.arange(len(s_), device=dev), cl[b:b + 50000]] = -2.0              # 去掉自身子课题
            v_, i_ = s_.topk(xd, dim=1)
            xd_sim.append(v_)
            xd_cl.append(i_)
        xd_sim, xd_cl = torch.cat(xd_sim), torch.cat(xd_cl)
        code2cell = {int(c): k for k, c in enumerate(meta.cell.to_numpy()[ptr[:-1]])}
        lut = torch.full((ncl, 2030 - 2010), -1, dtype=torch.long, device=dev)
        for c_, k_ in code2cell.items():
            lut[c_ // 10000, c_ % 10000 - 2010] = k_
        xd_cell = lut[xd_cl, (yr - 2010)[:, None].expand_as(xd_cl)]                    # (N, xd)，−1 = 该子课题当年无样本
        del En
    H = cfg.get("heads", 4)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_emb = nn.ModuleDict({c: nn.Embedding(n, 32) for c, n in cat_sizes.items()})
            self.paper = nn.Sequential(nn.Linear(768 + dp + dce, 2 * d), nn.GELU(), nn.Dropout(cfg.get("dropout", 0.2)), nn.Linear(2 * d, d), nn.GELU())
            self.rival = nn.Sequential(nn.Linear(768 + dr, d), nn.GELU(), nn.Linear(d, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.attn = nn.MultiheadAttention(d, H, batch_first=True, dropout=0.1)
            n_self = cfg.get("n_self", 1 if cfg.get("self_attn") else 0)
            if n_self > 0:                                 # v6d：对手之间先互相"看"若干层
                self.rself = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 2 * d, dropout=0.1, batch_first=True), n_self)
            z_in = 0 if two_chan == "strict" else (2 * d if two_chan == "plus" else d)
            hin = d + z_in + (128 if use_cell else 0) + (128 if xd else 0) + (d if n_coh else 0)
            if cfg.get("aux_years") or cfg.get("aux_w", 0) > 0:   # v6c：逐年辅助目标（输入 = 预测头的输入，另外总是含格子嵌入）
                self.aux = nn.Linear(hin + (0 if use_cell else 128), 3)
            self.cell = nn.Sequential(nn.Linear(dc, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
            self.demand = nn.Linear(128, 3)
            self.demand_lin = nn.Linear(dc, 3)                 # 线性主干：对训练期之外的格子特征可以外推
            self.alpha = nn.Linear(128, 3)
            self.head = nn.Sequential(nn.Linear(hin, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 3))
            self.level_lin = nn.Linear(dc, 1)
            self.level = nn.Linear(128, 1)
            self.log_r_dem = nn.Parameter(torch.zeros(3))
            self.log_r_pap = nn.Parameter(torch.zeros(3))
            # 第七轮新增模块放在最后创建：不改变旧配置的参数初始化顺序（同一种子下旧配置结果逐位不变）
            if cfg.get("rel_rival"):
                self.rel = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            if two_chan:
                self.null2 = nn.Parameter(torch.zeros(1, 1, d))
                self.attn2 = nn.MultiheadAttention(d, H, batch_first=True, dropout=0.1)
                self.e_sub, self.e_com = nn.Linear(d, 1), nn.Linear(d, 1)
            if luce:
                self.q_self, self.q_riv = nn.Linear(d, 1), nn.Linear(d, 1)
                self.subst = nn.Sequential(nn.Linear(dr, 32), nn.GELU(), nn.Linear(32, 1))
                self.luce_g = nn.Parameter(torch.zeros(()))
            if n_coh:
                self.coh_in = nn.Linear(2 * d, d)
                self.coh = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 2 * d, dropout=0.1, batch_first=True), n_coh,
                                                 enable_nested_tensor=False)
            if xd:
                self.xd_tau = nn.Parameter(torch.tensor(3.0))     # 相似度 → 权重的温度（log 尺度）

        def forward(self, idx, cid, cells, want_attn=False):
            tz = 0.0 if cfg.get("no_text") else 1.0
            parts = [E[erow[idx]].float() * tz, xp[idx]] + [self.cat_emb[c](cat_t[c][idx]) for c in cat_sizes]
            h = self.paper(torch.cat(parts, 1))
            w = None
            extra = torch.zeros(len(idx), device=dev)              # 第七轮：直接加到 log 预测上的可解释项
            self._parts = {}
            zs = []
            if cfg.get("rivals") == "none":
                zs = [torch.zeros_like(h)]
            else:
                rraw = rf[idx].float()
                rv = self.rival(torch.cat([E[rrow[idx]].float() * tz, rraw], 2))      # (n, K, d)
                m = mask_all[idx]
                if self.training and cfg.get("rdrop", 0) > 0:      # 训练时随机遮掉一部分对手（正则化）
                    m = m | (torch.rand(m.shape, device=dev) < cfg["rdrop"])
                if hasattr(self, "rself"):
                    rv = self.rself(rv, src_key_padding_mask=m)
                if hasattr(self, "rel"):                           # 相对竞争位置
                    hb = h.unsqueeze(1).expand_as(rv)
                    rv = rv + self.rel(torch.cat([rv - hb, rv * hb], 2))
                if cfg.get("rivals") == "mean":
                    keep = (~m).float().unsqueeze(2)
                    zs = [(rv * keep).sum(1) / keep.sum(1).clamp(min=1)]
                else:
                    kv = torch.cat([self.null.expand(len(idx), 1, d), rv], 1)
                    km = torch.cat([torch.zeros(len(idx), 1, dtype=torch.bool, device=dev), m], 1)
                    km1 = km2 = km
                    if tpart:                                      # A3：两个通道各看一部分对手（空对手两边都可见）
                        S = S_all[idx]
                        z0 = torch.zeros(len(idx), 1, dtype=torch.bool, device=dev)
                        km1, km2 = torch.cat([z0, m | ~S], 1), torch.cat([z0, m | S], 1)
                    z, w = self.attn(h.unsqueeze(1), kv, kv, key_padding_mask=km1, need_weights=want_attn)
                    zs = [z.squeeze(1)]
                    if two_chan:                                   # 第二个通道：互补 / 可见度
                        kv2 = torch.cat([self.null2.expand(len(idx), 1, d), rv], 1)
                        z2, _ = self.attn2(h.unsqueeze(1), kv2, kv2, key_padding_mask=km2, need_weights=False)
                        zs.append(z2.squeeze(1))
                        e_s = F.softplus(self.e_sub(zs[0]).squeeze(1).float())       # 替代：抢走的引用（≥ 0，取负号）
                        e_c = F.softplus(self.e_com(zs[1]).squeeze(1).float())       # 互补 / 可见度：带来的引用（≥ 0）
                        extra = extra - e_s + e_c
                        self._parts.update(subst=e_s, compl=e_c)
                        if two_chan == "strict":
                            zs = []
                if luce:                                           # 吸引力份额：log(自身吸引力 / 竞争集合总吸引力)
                    qi = self.q_self(h).float()                                  # (n, 1)
                    qj = self.q_riv(rv).squeeze(2).float()                       # (n, K)
                    ls = F.logsigmoid(self.subst(rraw).squeeze(2).float())       # log σ_ij ∈ (−∞, 0)
                    t = (ls + qj - qi).masked_fill(m, float("-inf"))
                    comp = -torch.logsumexp(torch.cat([torch.zeros_like(qi), t], 1), 1)   # −log(1 + Σ σ e^{q_j − q_i}) ≤ 0
                    extra = extra + F.softplus(self.luce_g) * comp
                    self._parts["luce"] = comp
            ce = self.cell(cx[cells])
            inp = [h] + zs + ([ce[cid]] if use_cell else [])
            if xd:                                                 # 跨域需求：相近子课题同年的格子嵌入，按相似度加权
                ok = xd_cell[idx] >= 0
                cex = self.cell(cx[xd_cell[idx].clamp(min=0)])                   # (n, xd, 128)
                wx = torch.softmax((xd_sim[idx] * torch.exp(self.xd_tau)).masked_fill(~ok, -1e4), 1) * ok
                inp.append((wx.unsqueeze(2) * cex).sum(1))
            if n_coh:                                              # 同期竞争层：同格子论文互相注意
                u = self.coh_in(torch.cat([h, zs[0] if zs else torch.zeros_like(h)], 1))
                nc = len(cells)
                cnt = torch.bincount(cid, minlength=nc)
                start = torch.cumsum(cnt, 0) - cnt
                pos = torch.arange(len(idx), device=dev) - start[cid]
                X = torch.zeros(nc, int(cnt.max()), d, device=dev, dtype=u.dtype)
                X[cid, pos] = u
                pm = torch.ones(nc, X.shape[1], dtype=torch.bool, device=dev)
                pm[cid, pos] = False
                inp.append(self.coh(X, src_key_padding_mask=pm)[cid, pos])
            inp = torch.cat(inp, 1)
            self._aux_out = self.aux(inp if use_cell else torch.cat([inp, ce[cid]], 1)).float() if hasattr(self, "aux") else None
            logA = self.demand_lin(cx[cells]) + 0.1 * self.demand(ce)
            if "logshare" in cfg:
                logA = (self.level_lin(cx[cells]) + 0.1 * self.level(ce)).expand(-1, 3)       # 格子水平（log1p 空间）
                if cfg.get("level_oracle"):
                    logA = cell_meanlog[cells][:, None].expand(-1, 3)
            o = self.head(inp).float() + extra[:, None] / 3          # 3 个逐年 logit 之和 = log1p(ŷ3)，可解释项均分到 3 列
            return o, logA.float(), F.softplus(self.alpha(ce).float()) + 1e-3, w

    net = Net().to(dev)
    _ly3 = np.log1p(L("y.npy")[:, (1 if cfg.get("after1y") else 0):].sum(1))
    cell_meanlog = torch.from_numpy(np.add.reduceat(_ly3, ptr[:-1]) / np.diff(ptr)).float().to(dev)
    if cfg.get("stage1"):
        a1, sd1 = stage1_demand(seed)
        s1A, s1sd = torch.from_numpy(a1).float().to(dev), torch.from_numpy(sd1).float().to(dev)
    with torch.no_grad():                                   # 需求塔偏置初始化为训练格子的平均 log 总量，避免开头预测量级相差上百倍
        trc = torch.from_numpy(np.where(csplit == "train")[0]).to(dev)
        net.demand_lin.bias.copy_(torch.log(ctot[trc] + 1).mean(0))
        net.demand.weight.mul_(0.1)
        net.level_lin.bias.fill_(float(np.log1p(L("y.npy").sum(1))[np.repeat(csplit == "train", np.diff(ptr))].mean()))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4))
    ema = None
    if cfg.get("ema", 0) > 0:                              # 权重滑动平均：评价与保存都用平均后的权重，减小种子间波动
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema"]))
    if cfg.get("t_max"):
        # 第七轮：固定训练 t_max 轮（余弦降到最低），最后 swa_k 轮的权重取平均作为最终模型，不再按验证集早停
        # （验证误差逐轮跳动很大，早停的"运气"会盖过结构之间的真实差别）
        max_epochs, patience = cfg["t_max"], 10 ** 6
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    swa = None

    def nb_nll(k, mu, r):
        return -(torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1) + r * torch.log(r / (r + mu)) + k * torch.log(mu / (r + mu) + 1e-12))

    def seg_logsoftmax(o, cid, nc):
        mx = torch.full((nc, o.shape[1]), -1e30, device=dev).scatter_reduce(0, cid[:, None].expand_as(o), o, "amax")
        ex = torch.exp(o - mx[cid])
        sm = torch.zeros(nc, o.shape[1], device=dev).index_add_(0, cid, ex)
        return o - mx[cid] - torch.log(sm[cid])

    def batch_tensors(cells):
        st, en = ptr[cells], ptr[cells + 1]
        idx = torch.from_numpy(np.concatenate([np.arange(a, b) for a, b in zip(st, en)])).to(dev)
        cid = torch.from_numpy(np.repeat(np.arange(len(cells)), en - st)).to(dev)
        return idx, cid, torch.from_numpy(cells).to(dev)

    def losses(cells):
        idx, cid, ct = batch_tensors(cells)
        o, logA, alpha, _ = net(idx, cid, ct)
        yy = y[idx]
        if cfg.get("direct"):
            mu = torch.exp(o.clamp(-10, 12))
            return nb_nll(yy, mu, torch.exp(net.log_r_pap)).sum() / len(idx)
        if "logshare" in cfg:
            lam = cfg["logshare"]
            o3 = o.sum(1)                                                    # 3 个 logit 合成一个（份额针对三年之和）
            mean_c = torch.zeros(len(cells), device=dev).index_add_(0, cid, o3) / torch.bincount(cid, minlength=len(cells))
            pred = o3 if cfg.get("direct_log") else logA[:, 0][cid] + (o3 - mean_c[cid])
            lossf = F.huber_loss if cfg.get("loss_fn") == "huber" else F.mse_loss
            loss = lossf(pred, torch.log1p(yy.sum(1)))
            if net._aux_out is not None:
                loss = loss + cfg.get("aux_w", 0.3) * lossf(net._aux_out, torch.log1p(yy))
            if lam > 0:
                tot3, y3 = ctot[ct].sum(1), yy.sum(1)
                lp = seg_logsoftmax(o3[:, None], cid, len(cells))[:, 0]
                al = alpha[:, 0]
                a = al[cid] * torch.exp(lp)
                ll = (torch.lgamma(al) - torch.lgamma(tot3 + al)).sum() + (torch.lgamma(y3 + a) - torch.lgamma(a)).sum()
                loss = loss - lam * ll / len(idx)
            return loss
        tot = ctot[ct]
        dem = 0.0 if cfg.get("oracle") else nb_nll(tot, torch.exp(logA.clamp(-5, 18)), torch.exp(net.log_r_dem)).sum() / len(cells)
        lp = seg_logsoftmax(o, cid, len(cells))
        if cfg.get("loss") == "mse":
            A = tot if cfg.get("oracle") else torch.exp(logA.clamp(-5, 18)).detach()
            mu = A[cid] * torch.exp(lp)
            return dem + F.mse_loss(torch.log1p(mu), torch.log1p(yy))
        if cfg.get("share") == "mult":
            return dem - (yy * lp).sum() / len(idx)
        a = alpha[cid] * torch.exp(lp)
        ll_c = torch.lgamma(alpha) - torch.lgamma(tot + alpha)
        ll_i = torch.zeros_like(tot).index_add_(0, cid, torch.lgamma(yy + a) - torch.lgamma(a))
        return dem - (ll_c + ll_i).sum() / len(idx)

    @torch.no_grad()
    def predict(split, n_mc=32, want_attn=False):
        net.eval()
        cells = np.where(csplit == split)[0]
        res = {k: [] for k in ("idx", "pred", "mu", "share", "A")}
        parts = {}
        attn = []
        g = torch.Generator(device=dev).manual_seed(seed * 1000 + 7)
        for b in range(0, len(cells), 64):
            cs = cells[b:b + 64]
            idx, cid, ct = batch_tensors(cs)
            o, logA, alpha, w = net(idx, cid, ct, want_attn=want_attn)
            for k_, v_ in net._parts.items():
                parts.setdefault(k_, []).append(v_.float().cpu().numpy())
            if "logshare" in cfg:
                o3 = o.sum(1)
                cnt = torch.bincount(cid, minlength=len(cs)).float()
                mean_c = torch.zeros(len(cs), device=dev).index_add_(0, cid, o3) / cnt
                pr_log = o3 if cfg.get("direct_log") else logA[:, 0][cid] + (o3 - mean_c[cid])
                lp = seg_logsoftmax(o3[:, None], cid, len(cs))[:, 0]
                res["idx"].append(idx.cpu().numpy())
                res["pred"].append(pr_log.cpu().numpy())
                res["mu"].append(torch.expm1(pr_log).clamp(min=0)[:, None].expand(-1, 3).cpu().numpy() / 3)
                res["share"].append(torch.exp(lp).cpu().numpy())
                res["A"].append(torch.exp(logA[:, 0][cid]).cpu().numpy())
                if want_attn and w is not None:
                    attn.append(w.squeeze(1).half().cpu().numpy())
                continue
            if cfg.get("direct"):
                mu = torch.exp(o.clamp(-10, 12))
                r = torch.exp(net.log_r_pap)
                lam = torch.distributions.Gamma(r.expand(n_mc, *mu.shape), (r / mu).expand(n_mc, *mu.shape)).sample()
                ys = torch.poisson(lam, generator=g)
                Ac, share = mu.sum(1), torch.ones(len(idx), device=dev)
            else:
                lp = seg_logsoftmax(o, cid, len(cs))
                p = torch.exp(lp)
                if cfg.get("stage1"):
                    A = torch.exp(s1A[ct])
                else:
                    A = ctot[ct] if cfg.get("oracle") else torch.exp(logA.clamp(-5, 18))
                mu = A[cid] * p
                if cfg.get("loss") == "mse":
                    ys = mu.unsqueeze(0)
                else:
                    if cfg.get("stage1"):
                        Ns = torch.round(torch.exp(s1A[ct].unsqueeze(0) + s1sd * torch.randn(n_mc, *A.shape, device=dev, generator=g)) - 1).clamp(min=0)
                    elif cfg.get("oracle"):
                        Ns = A.unsqueeze(0).expand(n_mc, *A.shape)
                    else:
                        r = torch.exp(net.log_r_dem)
                        Ns = torch.poisson(torch.distributions.Gamma(r.expand(n_mc, *A.shape), (r / A).expand(n_mc, *A.shape)).sample(), generator=g)
                    if cfg.get("share") == "mult":
                        pi = p.unsqueeze(0).expand(n_mc, *p.shape)
                    else:
                        a = (alpha[cid] * p).unsqueeze(0).expand(n_mc, *p.shape)
                        G = torch.distributions.Gamma(a, torch.ones_like(a)).sample()
                        sg = torch.zeros(n_mc, len(cs), 3, device=dev).index_add_(1, cid, G)
                        pi = G / sg[:, cid].clamp(min=1e-20)
                    ys = torch.poisson(Ns[:, cid] * pi, generator=g)
                y3w = (mu.sum(1) / torch.zeros(len(cs), device=dev).index_add_(0, cid, mu.sum(1))[cid])
                share, Ac = y3w, A.sum(1)[cid]
            pred = torch.log1p(ys.sum(2)).mean(0)
            res["idx"].append(idx.cpu().numpy())
            res["pred"].append(pred.cpu().numpy())
            res["mu"].append(mu.cpu().numpy())
            res["share"].append(share.cpu().numpy())
            res["A"].append((Ac if Ac.shape[0] == len(idx) else Ac[cid]).cpu().numpy())
            if want_attn and w is not None:
                attn.append(w.squeeze(1).half().cpu().numpy())
        net.train()
        out = {k: np.concatenate(v) for k, v in res.items()}
        out.update({"part_" + k: np.concatenate(v) for k, v in parts.items()})
        return out, (np.concatenate(attn) if attn else None)

    if placebo is not None and placebo.startswith("cf:"):
        # 修改计划 A2：反事实屏蔽。模型参数不变，把 cf/<名称>.parquet 中的 (focal_id, comp_id) 对手从焦点论文的竞争集合中屏蔽
        # （掩码同时作用于对手自注意力与交叉注意力），重新预测验证集与测试集
        cfn = placebo[3:]
        pairs = pd.read_parquet(os.path.join(C.HERE, "cf", cfn + ".parquet"))
        pos = pd.Series(np.arange(len(meta)), index=meta.paper_id.to_numpy())
        erm = pd.Series(meta.emb_row.to_numpy(), index=meta.paper_id.to_numpy())
        fi, ce_ = pos.reindex(pairs.focal_id).to_numpy(), erm.reindex(pairs.comp_id).to_numpy()
        ok = ~(np.isnan(fi) | np.isnan(ce_))
        fi_t = torch.from_numpy(fi[ok].astype(np.int64)).to(dev)
        ce_t = torch.from_numpy(ce_[ok].astype(np.int64)).to(dev)
        hit = rrow[fi_t] == ce_t[:, None]                                          # (对数, K)：对方在不在自己的竞争集合里
        Kr = rrow.shape[1]
        cnt = torch.zeros(rrow.shape, dtype=torch.int32, device=dev)
        cnt.index_put_((fi_t.repeat_interleave(Kr), torch.arange(Kr, device=dev).repeat(len(fi_t))), hit.flatten().int(), accumulate=True)
        mask_all |= cnt > 0
        hp = pairs[ok].assign(hit=hit.any(1).cpu().numpy(), rank=torch.where(hit.any(1), hit.int().argmax(1) + 1, 0).cpu().numpy())
        hp.to_parquet(os.path.join(out_dir, f"cf_{cfn}_hits.parquet"), index=False)
        net.load_state_dict(torch.load(os.path.join(out_dir, "model.pt")))
        outs = []
        for sp in ("val", "test"):
            pr, _ = predict(sp, n_mc=32)
            outs.append(pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[pr["idx"]], "split": sp, "seed": seed, "pred": pr["pred"]}))
        pd.concat(outs).to_parquet(os.path.join(out_dir, f"preds_cf_{cfn}.parquet"), index=False)
        print(f"[{cfg_name} s{seed}] cf {cfn} DONE：{int(ok.sum())} 对可定位，其中 {int(hp.hit.sum())} 对的对方在竞争集合内并已屏蔽", flush=True)
        return
    if placebo is not None:
        # 实验 5-3：模型参数不变，把测试集每篇论文的竞争集合（对手行号、关系特征、掩码）整体换成另一篇论文的
        rng = np.random.default_rng(seed + 99)
        te_cells = np.where(csplit == "test")[0]
        src = np.arange(rrow.shape[0])
        if placebo == "shuffle_cell":
            for c in te_cells:
                a, b = ptr[c], ptr[c + 1]
                if b - a > 1:
                    src[a:b] = a + rng.permutation(b - a)
                    same = src[a:b] == np.arange(a, b)
                    if same.any():                                   # 尽量避免换回自己
                        src[a:b][same] = np.roll(src[a:b], 1)[same]
        elif placebo == "shuffle_all":
            te = np.concatenate([np.arange(ptr[c], ptr[c + 1]) for c in te_cells])
            src[te] = rng.permutation(te)
        si = torch.from_numpy(src).to(dev)
        rrow.copy_(rrow[si])
        rf.copy_(rf[si])
        mask_all.copy_(mask_all[si])
        net.load_state_dict(torch.load(os.path.join(out_dir, "model.pt")))
        pr, _ = predict("test", n_mc=32)
        pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[pr["idx"]], "split": "test", "model": f"MPC-Net[{cfg_name}{tag}]|{placebo}",
                      "seed": seed, "pred": pr["pred"]}).to_parquet(os.path.join(out_dir, f"preds_placebo_{placebo}.parquet"), index=False)
        print(f"[{cfg_name} s{seed}] placebo {placebo} DONE (换掉自己对手集合的比例 {(src != np.arange(len(src))).mean():.3f})", flush=True)
        return
    ytrue3 = np.log1p(L("y.npy")[:, (1 if cfg.get("after1y") else 0):].sum(1))
    cell_tot3 = np.repeat(L("cell_tot.npy").sum(1), np.diff(ptr))                     # 每篇论文所在格子的真实三年总量
    tr_cells = np.where(csplit == "train")[0]
    if cfg.get("only_year"):                               # 只用某一年的训练格子（与 DPPDCC 的训练数据相同时用）
        cyear_ = meta.Y.to_numpy()[ptr[:-1]]
        tr_cells = tr_cells[cyear_[tr_cells] == cfg["only_year"]]
    if cfg.get("trainval"):                                # 训练 + 验证一起训练，固定轮数，不用验证集早停
        tr_cells = np.where((csplit == "train") | (csplit == "val"))[0]
        max_epochs, patience = cfg["fixed_ep"], 10 ** 6
    if train_frac < 1.0:                                   # 实验 9：只用一部分训练格子，测规模与耗时的关系
        tr_cells = np.sort(np.random.default_rng(seed).choice(tr_cells, int(len(tr_cells) * train_frac), replace=False))
    sizes = ptr[tr_cells + 1] - ptr[tr_cells]
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, bad, logs = 1e9, -1, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        net.train()
        perm = np.random.default_rng(seed * 100 + ep).permutation(len(tr_cells))
        batches, cur, n = [], [], 0
        for j in perm:
            cur.append(tr_cells[j])
            n += sizes[j]
            if n >= cfg.get("bs", 3000):
                batches.append(np.array(cur))
                cur, n = [], 0
        if cur:
            batches.append(np.array(cur))
        tl = 0.0
        for bc in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = losses(bc)
            opt.zero_grad()
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if ema is not None:
                ema.update_parameters(net)
            tl += loss.item()
        sched.step()
        if cfg.get("t_max") and ep + 1 > max_epochs - cfg.get("swa_k", 1):
            sd_ = net.state_dict()                         # 逐轮等权平均（手写：模型上挂着中间张量，不能深拷贝）
            if swa is None:
                swa, n_swa = {k: v.detach().float().clone() for k, v in sd_.items()}, 1
            else:
                n_swa += 1
                for k, v in sd_.items():
                    swa[k] += (v.detach().float() - swa[k]) / n_swa
        if ema is not None:                                # 用滑动平均权重评价
            live = {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema.module.state_dict())
        pv, _ = predict("val", n_mc=16)
        male = float(np.abs(pv["pred"] - ytrue3[pv["idx"]]).mean())
        dem_err = float(np.abs(np.log1p(pv["A"]) - np.log1p(cell_tot3[pv["idx"]])).mean())
        bias = float((pv["pred"] - ytrue3[pv["idx"]]).mean())          # 整体水平偏差（预测均值 − 真实均值，log 尺度）
        logs.append({"epoch": ep + 1, "train_loss": tl / len(batches), "val_MALE": male, "val_bias": bias, "val_demand_err": dem_err,
                     "sec": round(time.time() - t0)})
        print(f"[{cfg_name} s{seed} gpu{gpu}] ep {ep + 1}: loss {tl / len(batches):.4f} val MALE {male:.4f} 偏差 {bias:+.3f} ({time.time() - t0:.0f}s)", flush=True)
        if not np.isfinite(male):
            break
        improved = (male < best - 1e-4) or bool(cfg.get("trainval"))   # trainval：验证集已在训练中，保存最后一轮
        if improved:                                       # 有 EMA 时此刻载入的是平均权重，保存的也是平均权重
            best, best_ep, bad = male, ep + 1, 0
            torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
        if ema is not None:
            net.load_state_dict(live)
        if not improved:
            bad += 1
            if bad >= patience and ep + 1 >= cfg.get("min_ep", 0):   # 至少训练 min_ep 轮才允许早停（避免撞上前期偶然的低点）
                break
    if swa is not None:                                    # 最终模型 = 最后 swa_k 轮权重的平均
        net.load_state_dict({k: v.to(net.state_dict()[k].dtype) for k, v in swa.items()})
        pv, _ = predict("val", n_mc=16)
        best, best_ep = float(np.abs(pv["pred"] - ytrue3[pv["idx"]]).mean()), max_epochs
        logs.append({"epoch": "swa", "val_MALE": best, "val_bias": float((pv["pred"] - ytrue3[pv["idx"]]).mean())})
        torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))
        print(f"[{cfg_name} s{seed}] SWA（最后 {cfg.get('swa_k', 1)} 轮平均）val MALE {best:.4f}", flush=True)
    pd.DataFrame(logs).to_csv(os.path.join(out_dir, "log.csv"), index=False)
    net.load_state_dict(torch.load(os.path.join(out_dir, "model.pt")))
    rows = []
    pred_sec = {}
    for sp in ("val", "test"):
        tp = time.time()
        pr, at = predict(sp, n_mc=32, want_attn=(cfg_name == "full" and sp == "test"))
        torch.cuda.synchronize()
        pred_sec[sp] = round(time.time() - tp, 2)
        if at is not None:
            np.save(os.path.join(out_dir, "attn_test.npy"), at)
            np.save(os.path.join(out_dir, "attn_test_idx.npy"), pr["idx"])
        rows.append(pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[pr["idx"]], "split": sp, "model": f"MPC-Net[{cfg_name}{tag}]",
                                  "seed": seed, "pred": pr["pred"], "mu1": pr["mu"][:, 0], "mu2": pr["mu"][:, 1], "mu3": pr["mu"][:, 2],
                                  "share_y3": pr["share"], "A_y3": pr["A"],
                                  **{k: v for k, v in pr.items() if k.startswith("part_")}}))
    pd.concat(rows).to_parquet(os.path.join(out_dir, "preds.parquet"), index=False)
    if cfg.get("rdrop", 0) > 0:                            # 代码自查 C8：评价模式下再预测一次，应与上面完全相同
        pr2, _ = predict("test", n_mc=32)
        pd.DataFrame({"paper_id": meta.paper_id.to_numpy()[pr2["idx"]], "pred": pr2["pred"]}).to_parquet(
            os.path.join(out_dir, "preds_eval_twice.parquet"), index=False)
    ep_secs = np.diff([0] + [r_["sec"] for r_ in logs if "sec" in r_])
    json.dump({"config": cfg_name, "seed": seed, "best_epoch": best_ep, "best_val_MALE": best, "sec": round(time.time() - t0),
               "n_train_papers": int(sizes.sum()), "sec_per_epoch": float(np.mean(ep_secs)) if len(ep_secs) else None,
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "predict_sec": pred_sec,
               "n_params": sum(p_.numel() for p_ in net.parameters())},
              open(os.path.join(out_dir, "summary.json"), "w"))
    print(f"[{cfg_name} s{seed}] DONE best ep {best_ep} val MALE {best:.4f} ({time.time() - t0:.0f}s)", flush=True)


def gpu_worker(gpu, tasks, tag, max_epochs, placebo=None, train_frac=1.0):
    for cfg_name, seed in tasks:
        try:
            run(cfg_name, seed, gpu, max_epochs=max_epochs, tag=tag, placebo=placebo, train_frac=train_frac)
        except Exception as e:  # 单个任务失败不影响其他任务，记录后继续
            import traceback
            print(f"[{cfg_name} s{seed} gpu{gpu}] FAILED: {e}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="full")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--placebo", default=None, help="实验 5-3：shuffle_cell / shuffle_all；修改计划 A2：cf:<名称>（只预测，不训练）")
    ap.add_argument("--train_frac", type=float, default=1.0, help="实验 9：只用这一比例的训练格子")
    args = ap.parse_args()
    tasks = [(c, int(s)) for s in args.seeds.split(",") for c in args.configs.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=gpu_worker, args=(g, tasks[i::len(gpus)], args.tag, args.max_epochs, args.placebo, args.train_frac)) for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
