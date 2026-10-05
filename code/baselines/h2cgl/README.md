# H2CGL 基线（IP&M 2023）在 MPCC 引用预测数据上的运行记录

- 官方仓库：github.com/ECNU-Text-Computing/H2CGL；代码位置 /path/to/mpcc/third_party/H2CGL（兼容性修改见同目录 MPCC_COMPAT_PATCHES.md）。
- 环境：/path/to/envs/dppdcc（Python 3.11、torch 2.4 cu124、DGL 2.4），单张 RTX 4090。
- 输出：本目录 preds_dp.parquet、preds_all.parquet（列 paper_id int64、split 'val'/'test'、seed int、pred = log1p 尺度预测），summary.json（自检与指标）。

## 1. 任务设定（与 MPC-Net / DPPDCC 冷启动设定一致）

| 项 | 设定 |
|---|---|
| 预测时点 T | 发表年 Y 年末；输入只来自 ≤ Y 的快照图，焦点论文 Y 年之后收到的引用一律不可见 |
| 训练集 | 2019 年论文 61,238 篇（快照 2017–2019） |
| 验证集（选检查点） | 2020 年论文 73,400 篇（快照 2018–2020） |
| 测试集（只评估一次） | 2021 年论文 84,566 篇（快照 2019–2021） |
| 目标 dp | 语料内三年被引（Y+1～Y+3 年被语料内论文引用次数），即 DPPDCC split_data 的标签 |
| 目标 all（主目标） | 全来源三年被引 = pred_v2/samples.parquet 的 y3；单独的数据源 mpcc_ai_all（同一套图，只换标签与按标签分组的困难负样本表） |

为什么只用 2019 年论文训练：H2CGL 每个阶段用固定的快照窗口（训练阶段截止 2019 年快照）。若训练集含 2017/2018 年论文，它们在 2019 年快照中已能看到 2018–2019 年收到的被引（属于预测窗口）→ 泄漏。因此与 DPPDCC 相同，训练集只放 2019 年论文。

## 2. 运行内容（官方流程 + 默认超参数）

命令（README 默认配置）：`--phase H2CGL --cl_type label_aug_hard_negative --aug_type cg --encoder_type CGIN+RGAT --n_layers 4 --hn 2 --hn_method co_cite`，另加 `--graph_type specter2`。
其余超参数取官方 configs/dblp.json（计算机领域数据集，与本 AI 语料最接近）：lr 1e-4、batch 32、epochs 20、weight_decay 1e-4、dropout 0.3、hidden/out 256、tau 0.4、cl_weight 0.5、aug_rate 0.1、aux_weight 0.5、辅助分类阈值 [10, 100]、损失 MSE（目标 log(1+y)）、gcn_out mean、pred_type snapshot、种子 123。

流程：
1. 困难负样本表（mpcc_prep_hn.py，官方 show_hn 的向量化等价实现）：co_cite / co_ref / in_cite / in_ref 只在训练论文之间、只用发表年 ≤ 2019 的论文的引用关系。
2. 每篇论文每年的 1 跳自我中心子图（mpcc_prep_ego.py，官方 get_paper_ego_subgraph；提速实现与官方逐项比对一致）。
3. 组合快照图（官方 CTSGCN.deal_graphs）与固定的 cg 增强训练图（官方 aug_graphs_plus）。
4. 训练 20 轮；每轮在验证集上算 MALE，按官方规则保存验证集 MALE 最小的"全局最优"检查点 H2CGL_aux.pkl。
5. 只加载该检查点，对验证集、测试集逐篇预测（mpcc_predict）；**不运行**官方 get_test_results（它在测试集上评估所有检查点，属于用测试集选模）。

## 3. 与官方 / 论文设定的偏差

| 偏差 | 原因与影响 |
|---|---|
| 节点特征用 SPECTER2（768 维，DPPDCC 已为本语料生成的快照图），embed_dim 300→768 | 与 DPPDCC、MPC-Net 用同一文本表示；官方为 GloVe 均值 300 维 |
| time_length 5→3（快照窗口 Y-2～Y） | 语料从 2016 年开始，没有 2015 年快照。冷启动下焦点论文在发表前的快照里不存在，官方 deal_graphs 会把开头的空快照删掉（"only valid snapshots"），每篇论文实际只有发表当年这一个有效快照，所以窗口长度 3 与 5 的输入完全相同 |
| 只用 2019 年论文训练 | 见第 1 节（防泄漏，同 DPPDCC） |
| 困难负样本候选表每格最多 200 个（随机均匀抽取） | 官方 CSV 中 co_ref 列表平均 191 个、最多上万，完整写出后无法读取；每步只从中随机抽 1–2 个，抽样规则不变，只降低跨轮次多样性。dp：61,238 篇中 13,294 篇的 co_ref 被截断 |
| 单一随机种子（123） | 单次训练约 7 小时 |
| 其余均为不改算法的兼容性修改 | 见 MPCC_COMPAT_PATCHES.md |

困难负样本是否使用未来信息：co-citation 候选只为训练论文（2019 年）构造，只用发表年 ≤ 2019 的施引论文，即截至训练快照 T = 2019 年末的信息；验证 / 测试论文在推断时不使用困难负样本（只在训练前向中使用）。标签分组（<10 / 10–99 / ≥100）用的是训练论文的训练标签。**无测试期信息泄漏。**

## 4. 预测尺度与检查点选择

- H2CGL 以 MSE 拟合 log(1 + y)（dataloader.log = True），模型输出即 log1p 尺度；官方评估把 exp(输出) − 1 的负值截为 0，等价于 log 尺度截到 0。
  因此 pred = max(模型输出, 0)。实际 dp 没有负输出；all 只有 3 篇验证、4 篇测试论文输出为负（截断后 MALE 变化 < 0.0001）。
- 检查点选择：官方 train_batch 的全局最优检查点 = 验证集（2020）MALE 最小的轮次。测试集（2021）只在最后用这一个检查点评估一次。
  mpcc_predict 重算的验证集 MALE 与训练日志中该轮的验证集 MALE 完全相同（dp 第 3 轮 0.678502；all 第 15 轮 0.783711），说明加载的确是该检查点。

## 5. 自检

- id 映射：mpcc_predict 中每个样本的内部节点号经 sample_node_trans.json 反查回 paper_id，与数据加载器给出的 paper_id 逐条断言一致；
  dp 的 'true' 标签与独立从 DPPDCC 的 sample_cite_year_dict.json 重算的语料内三年被引逐篇一致（157,966 / 157,966）；
  all 的 'true' 与 samples.parquet 的 y3 逐篇一致（见下表）。
- 覆盖：两个目标都覆盖全部 2020（73,400）与 2021（84,566）样本论文，无缺失、无重复。
- 尺度：预测与 log1p(y) 同尺度（均值见下表）。

## 6. 结果（MALE = mean |pred − log1p(y)|）

| 目标 | 选中轮次（验证集 MALE 最小） | 验证集 2020 MALE | 测试集 2021 MALE | 测试集 均值 pred / 均值 log1p(y) | 标签核对 |
|---|---|---|---|---|---|
| dp（语料内三年被引） | 第 3 轮 / 20 | **0.6785** | **0.7337** | 1.422 / 1.018 | 157,966 / 157,966 一致 |
| all（全来源三年被引，主目标） | 第 15 轮 / 20 | **0.7837** | **0.8404** | 1.868 / 1.536 | 157,966 / 157,966 一致 |

- 不截断（直接用模型输出）时：dp 完全相同；all 验证 0.78377、测试 0.84049（all 只有 3 篇验证、4 篇测试论文输出为负）。
- 验证集 MALE 随轮次波动较大（dp 0.68–0.86、all 0.78–1.00），主要是整体水平（偏置）的摆动；测试集上两个目标都明显高估（2021 届论文实际被引水平低于 2020 届），这与 MPC-Net 等其他模型观察到的 2022 年后被引水平下降一致。
- 逐轮记录：results/mpcc_ai/H2CGL_aux_records.csv、results/mpcc_ai_all/H2CGL_aux_records.csv（test_* 列全为 0：训练中未评估测试集）。

## 7. 运行时间（共享服务器，CPU 负载高）

| 步骤 | 墙钟时间 | 说明 |
|---|---|---|
| 困难负样本表 | 约 20 秒 / 目标 | 稀疏矩阵实现（官方双重循环估计需数天） |
| 自我中心子图（2017–2021 共 5 个快照，各 219,204 篇） | 约 56 分钟 | 5 进程并行；抽取约 8–18 分钟，其余为 joblib 序列化 |
| 组合快照图（train / val / test） | 约 1 小时 49 分钟 | 3 进程并行，主要耗时在读入子图文件 |
| cg 增强训练图 | 约 46 分钟 | |
| 训练 dp（20 轮） | 24,848 秒（6.9 小时） | 每轮约 11 分钟训练 + 8 分钟验证；GPU 0 |
| 训练 all（20 轮） | 27,009 秒（7.5 小时） | GPU 1，与 dp 并行 |
| 预测（验证 + 测试） | 约 38 分钟 / 目标 | 含读入图数据 |
| 合计 | 约 12 小时（2026-10-04 06:10–18:15 UTC） | |

## 8. 文件

- /path/to/mpcc/third_party/H2CGL/：data/mpcc_ai、data/mpcc_ai_all（split_data、困难负样本表；图与元数据为指向 DPPDCC 的只读符号链接）；
  checkpoints/mpcc_ai（自我中心子图、组合图、增强图、检查点 H2CGL_aux.pkl；mpcc_ai_all 的图为符号链接，检查点独立）；
  results/<数据源>/H2CGL_aux_records.csv（逐轮训练 / 验证指标）、mpcc_pred_{val,test}.csv（逐篇原始输出与标签）；
  脚本 mpcc_prep_data.py、mpcc_prep_hn.py、mpcc_prep_ego.py、mpcc_fast_ego.py、mpcc_export.py、run_prep_graphs.sh、run_train.sh。
