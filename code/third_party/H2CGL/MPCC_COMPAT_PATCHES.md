# H2CGL 在 MPCC 中运行所做的兼容性 / 工程修改

原则：只改与新版本库不兼容、或在本语料规模下跑不完的地方；**不改 H2CGL 的模型、损失、对比学习、增强、困难负样本规则与训练 / 选模流程**。
原始代码备份：同目录未改动文件与 GitHub 官方仓库一致（ECNU-Text-Computing/H2CGL）。

环境：/path/to/envs/dppdcc（Python 3.11、torch 2.4.0+cu124、DGL 2.4.0+cu124、numpy 1.26）；官方要求 DGL 0.9.1 / torch 1.10，不支持 RTX 4090 的 CUDA 版本。

| # | 文件 | 修改 | 原因 / 是否影响算法 |
|---|---|---|---|
| 1 | data_processor.py（导入） | torchtext 改为可选导入；缺失时用与 torchtext `basic_english` 规则相同的替代分词函数（同 DPPDCC 补丁） | torchtext 已停止维护，无 torch 2.4 版本。本实验用 BERT（SPECTER2）分词器，文本不进入模型（H2CGL 的 vocab_size = 0）。不影响算法 |
| 2 | data_processor.py（DataProcessor.__init__） | 数据源名不以 pubmed / dblp 结尾时，数据目录同样取 `./data/<data_source>/` | 官方只为 pubmed / dblp 设路径。不影响算法 |
| 3 | data_processor.py（load_graphs） | 快照图优先读官方 `.job`，不存在时读 DGL 2.x 的 `.dgl`（DPPDCC 为本语料生成的同结构快照图） | 复用 DPPDCC 已生成的 SPECTER2 特征快照（同一课题组、同一 get_feature_graph 流程）。不影响算法 |
| 4 | main.py（get_model） | 把配置中的 `max_time_length` 传给编码器（默认仍为 5） | 语料从 2016 年开始，time_length = 3（见 README 偏差说明）；官方把快照窗口上限写死为 5，会去找不存在的 2015 年快照。不影响算法 |
| 5 | main.py（train_single） | 环境变量 `MPCC_NO_TEST=1` 时训练结束后不调用官方 `get_test_results` | 官方函数在**测试集**上评估全部检查点；我们只用按验证集选出的全局最优检查点。防止测试集参与选择 |
| 6 | main.py（新增 3 个 phase） | `mpcc_deal_graphs`：按原作者 CTSGCN 的路径约定调用官方 `CTSGCN.deal_graphs`（一次一个阶段，可并行）；`mpcc_cl_data`：官方 `get_cl_data` 的等价实现；`mpcc_predict`：加载全局最优检查点，对验证 / 测试集逐篇输出 | 官方 `get_graphs` / `get_fixed_cl_data` 对改名后的 H2CGL 路径映射有误（会读 `CTSGCN_graphs_trans.json`、写 `CCTSGCN_graphs_*`、读不存在的 `H2CGL_graphs_train.job`）。只做流程编排，调用的都是官方函数 |
| 7 | 新增 mpcc_prep_ego.py + mpcc_fast_ego.py | 官方第 1 步（TSGCN.deal_graphs：每篇论文每年的 1 跳自我中心子图）。`SingleSubgraph.get_single_subgraph` 用预建 CSR / CSC 索引的等价实现替换 | 官方实现在 DGL 2.x 下每次调用都扫描全图 COO 边表（约 0.5–0.8 秒 / 篇，40 万子图需数十小时）。选点规则（参考文献 / 施引各最多 100 篇，按 (时间, 被引) 降序截断；作者、期刊；诱导子图）完全相同；`verify` 模式与官方实现逐项比对点集、边集（含原边 ID）、全部点 / 边特征、is_ref / is_cite / is_target：一致。只有子图内部编号顺序可能不同（GNN 不变） |
| 8 | mpcc_prep_ego.py | 对 train ∪ val ∪ test 的并集建子图索引 | 官方只对 test 名单建索引并假设 test ⊇ train ∪ val；我们的 test 只放 2021 年论文。不影响算法 |
| 9 | 新增 mpcc_prep_hn.py | 官方 `show_hn`（困难负样本候选表）的稀疏矩阵等价实现 | 官方为训练论文两两 Python 集合求交（n = 61,238 → 约 19 亿次）。`--check` 在 3000 篇子集上与官方双重循环逐项比对四列（co_cite / co_ref / in_cite / in_ref）：完全一致。**唯一差别**：每格候选最多保留 200 个（超出时固定种子均匀抽取）——co_ref 列表平均上百、最多上万，完整写入 CSV 后官方 `eval()` 读取不可行；训练时每篇每步只从中随机取 1–2 个，抽样规则不变，仅降低跨轮次的候选多样性 |
| 10 | 新增 mpcc_prep_data.py、configs/mpcc_ai*.json | 数据目录（符号链接到 DPPDCC 的只读文件）、split_data（dp / all 两套标签）、配置（以官方 dblp.json 为底） | 对齐任务 |

| 11 | our_models/CTSGCN.py、our_models/CL.py | 注释掉 5 处调试用 print（如 `print(all_graphs[0].nodes['snapshot'].data.keys())`） | 训练时 cg 增强的训练图是 [g1, g2] 列表，官方这一行对列表取 `.nodes` 直接报错；其余几处每个批次打印整张批图，日志过大。只删输出，不影响计算 |
| 12 | 新增 mpcc_export.py、run_*.sh、smoke*.sh、configs/mpcc_smoke.json | 结果整理与自检、后台运行脚本、300/200/200 篇小规模端到端冒烟测试 | 工程脚本 |

说明：
- CTSGCN.load_single_graph 对官方发布的 h_pubmed / h_dblp 数据会交换 't_cites' 与 'is t_cited by' 两类快照边的权重（修正其旧版数据的方向约定）。
  我们的组合图由当前版本的 deal_graphs 新生成，两类边的权重已按目的快照归一化、方向一致，因此不做交换（保持官方代码对非 h_ 数据源的默认行为）。
- 运行过程中的一次事故：冒烟测试目录 data/mpcc_smoke 用符号链接指向 data/mpcc_ai 的文件，冒烟测试的困难负样本脚本经链接覆盖了
  data/mpcc_ai/hard_negative*.csv。发现后立即停止刚启动 1 分钟（尚在加载数据）的 dp 训练，用同一脚本重新生成该文件（脚本确定性、固定随机种子，标签分布与最初一致）后重新训练；冒烟目录中的这两个链接已删除。
  mpcc_ai_all 的文件与所有图文件未受影响。
