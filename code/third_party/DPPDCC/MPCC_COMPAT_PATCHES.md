# DPPDCC 在 MPCC 中运行所做的兼容性修改（只改与新版本库不兼容的地方，不改模型逻辑）

环境：/path/to/envs/dppdcc（Python 3.11、torch 2.4.0+cu124、DGL 2.4.0+cu124、torch_geometric 2.8；官方要求的 torch 1.12 / cu113 不支持 RTX 4090）

| 文件 | 修改 | 原因 |
|---|---|---|
| utilis/scripts.py | `from torch._six import inf` → `from math import inf` | torch 2.x 删除了 torch._six；inf 的值相同 |
| data_processor.py（get_feature_graph） | 作者 / 渠道特征的逐个循环 → 按节点分组求平均（torch index_add） | 原写法为 O(节点数 × 边数)，72 万篇论文的图上跑不完；结果与原循环相同 |
| utilis/tokenizers.py | torchtext 导入改为可选 | torchtext 已停止维护、无 torch 2.4 版本；只在 GloVe / 词表模式使用，本实验用 specter2 预算向量模式 |
| 新增 prep_mpcc.py、prep_mpcc2.py、configs/mpcc_ai.json | 数据准备：官方 s2orc 预处理函数 + 自写 split_data（与 MPCC 实验 2 的训练 / 验证 / 测试完全一致） | 对齐任务 |
| data_processor.py（add_co_strength） | 只对实际存在的边 (u, v) 计算 (A·Aᵀ)[u, v]（u、v 出边邻居交集大小，scipy 稀疏行相乘，分块） | DGL 2.x 的 adj() 返回 SparseMatrix（无 .values()）；原写法先算全量 A·Aᵀ（枢纽论文使非零元素爆炸）再逐边 Python 循环（千万级）；在这些边上的取值与原算法相同 |
| utilis/tokenizers.py | 无 torchtext 时用与其 basic_english 规则相同的替代分词函数 | 准备阶段仍会构造分词器 |
| 新增 prep_split_coldstart.py（替换 prep_mpcc2.py 写的 split_data） | train = 2019 年论文、val = 2020、test = 2021（test 名单含 2019–2021 以满足代码"test ⊇ train ∪ val"的假设，评价只取 2021） | DPPDCC 按阶段使用固定快照窗口；若训练集含 2017–2018 年论文，会在 2019 快照中看到预测窗口内的被引（泄漏）。与 MPC-Net 的冷启动设定一致，代价是 DPPDCC 只能用 2019 一年的论文训练 |

## 5. numpy 2 删除的类型别名（2026-10-03）
our_models/DDHGCN.py、utilis/collate_batch.py：np.long → np.int64（及同类 np.int / np.float / np.bool 别名），语义不变。

## 6. 零被引论文的累计被引序列（2026-10-03，prep_fix_accum.py）
get_citation_accum 只为语料内至少被引一次的论文写序列；训练时 get_pop_indicators 对零被引论文 KeyError。
补全为同年模板（发表前 -1，其余 0）。原文件备份 data/mpcc_ai/sample_citation_accum.orig.json。

## 7. mm_norm 的空列（2026-10-03）
严格冷启动下训练论文（2019 年发表）在 2017、2018 年没有被引历史，流行度序列的这些位置全是 -1，mm_norm 对空数组取最大值报错。
修改：整列无效时原样返回（保持 -1，后续按 -1 掩码）；max = min 时分母取 1e-12。

## 8. 空子图的残差变形（2026-10-03）
layers/GCN/custom_gcn.py：res_fc(h_dst).view(n, -1, out) 在某类节点数 n = 0 时无法推断 -1；改为显式 view(n, 最后一维 // out, out)，含义不变。

## 9. 测试阶段 DataLoader 句柄（2026-10-03）
测试时 4 个读数据进程共享大量张量，超过默认文件句柄上限 1024，报 "received 0 items of ancdata"，官方 get_test_results 吞掉异常后 argmin 空序列。
修改：main.py 设 torch.multiprocessing 共享方式为 file_system；run_train.sh / run_test.sh 中 ulimit -n 1048576。只重跑测试（run_test.sh），不重训。

## 10. 主口径版本 mpcc_ai_all（2026-10-04，prep_split_all.py）
与 mpcc_ai 共享全部图数据（符号链接），只把 split_data 的标签换成全部来源被引（samples.parquet 的 y3），检查点 / 结果写到独立目录。不改模型。

## 11. 额外随机种子（2026-10-04，prep_seed_copy.py / run_seed.sh）
每个种子一个独立数据源目录（mpcc_ai_s2 / _s3、mpcc_ai_all_s2 / _s3，数据与图全部符号链接共享），通过 --model_seed 2 / 3 训练，避免检查点同名覆盖。种子 1 = 原 mpcc_ai / mpcc_ai_all（默认 model_seed 123）。

## 12. 测试依次运行（2026-10-04，run_tests_seq.sh）
容器的 /dev/shm 只有 32 GB；file_system 共享方式下 4 个测试并行会写满共享内存（No space left on device），官方 get_test_results 吞掉异常后 argmin 空序列。改为依次运行，不改模型。
