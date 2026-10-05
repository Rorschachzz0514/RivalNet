# HINTS 原始代码的兼容性修改记录 (MPCC 对比实验)

原始代码备份在 `orig_src_backup/`（main.py, model*.py, rgcn.py, utils.py 未改动的版本）。
所有改动处在代码里都带 `[MPCC_COMPAT]` 注释，可用 `diff orig_src_backup/X.py src/X.py` 查看。
**以下改动均不改变 HINTS 的算法**（RGCN 结构、维度、对齐损失、插补模块、GRU+MLP 解码器、
对数正态累计引用曲线 `exp(eta*Phi((ln t-mu)/(1+sigma)))-1`、MSE 损失、Adam lr=0.01、beta=0.5、
embedding 128、`tf.nn.dropout(H_1, 0.2)`（TF1 中第二个位置参数是 keep_prob，原样保留，推断时也生效）全部保持原样）。

运行环境：`/path/to/envs/hints`（python 3.7 + tensorflow 1.15.5 CPU，numpy 1.18.5，scipy 1.4.1）。
TF1.15 的 GPU 版（CUDA 10.0）在 RTX 4090 (sm_89) 上需要 PTX JIT，实测卡死且内存暴涨到 150GB，已放弃；全部在 CPU 上运行。

## 1. `src/model_imputed.py`：越界下标查表（CPU 兼容）
* 原代码用 `-1` 给 index table 补位，直接 `tf.nn.embedding_lookup(emb, ids)`。在 GPU 上 TF 的 gather 对越界下标返回 0 向量，
  在 CPU 上则直接报错 (InvalidArgumentError)。
* 改为 `_lookup_pad(emb, ids) = embedding_lookup(emb, max(ids,0)) * 1[ids>=0]`，与 GPU 上的原始行为数值完全一致；
  均值的分母 `sum(ids>=0)+0.001` 未改。

## 2. `src/model.py`、`src/model_ts.py`：快照个数与预测年数参数化
* 原代码硬编码 `self.train_year = 5`（5 个年度快照）和解码器输出 `t = 1..5`（5 年累计引用）。
* 增加参数 `train_year`（快照数）与 `n_out`（预测年数），**默认值仍为 5**，原 main.py 调用行为不变。
* MPCC 实验中用 `train_year=3, n_out=3`：
  - 语料（exp0_v2）最早只到 2016 年，测试年 2021 之前最多 5 个快照、验证年 2020 之前只有 4 个、训练年 2019 之前只有 3 个；
    为了让训练/验证/测试使用相同位置含义的快照（每个快照位置有自己的 RGCN 权重），统一取 Y-3..Y-1 三个快照。
  - 标签只有 Y+1..Y+3（测试年 2021 的 Y+5=2026 尚不可观测），所以累计引用曲线只拟合 t=1,2,3。
  - 曲线公式、解码器结构没有任何变化，只是 `range(1,6)` → `range(1,n_out+1)`。

## 3. 新增 `src/hints_mpcc.py`（替代 main.py 作为入口，main.py 本身未改）
模型直接 `from model import Model`，数据处理复用 `utils.normalize / convert_sparse_matrix_to_sparse_tensor / label_recover`。
与 main.py 的差别：
1. 数据读取：读 `/path/to/mpcc/baselines/hints/convert_mpcc_to_hints.py` 生成的 npz（格式与官方 pkl 等价：
   每快照 4 个关系的 论文→对象 有向邻接 + 自环，行归一化；4 维随机特征；相邻快照对齐下标；index table；累计对数标签）。
2. 验证与早停：原代码固定训练 700 epoch、无验证集。这里每 20 个 epoch 在 2020 验证集上算 MALE，
   保留验证最好的 epoch 的 val/test 预测，连续 10 次评估不提升则停止，最多 800 epoch。测试集只在最后报告，不参与任何选择。
3. **反归一化用训练集的 label_max/label_min**：原代码对测试集重新做 min-max（用测试集自己的标签 max/min），
   再用它来反归一化预测——这是标签信息泄漏。这里统一用训练集的统计量，属于去泄漏修正。
4. batch：保持官方 README 的设定 “batch_size = 训练论文数，每个 epoch 只有一次迭代（full batch）”。
   GRU 初始状态和 reshape 都依赖固定 batch_size，预测时最后一个不满的 batch 用循环补齐后再丢弃补齐部分
   （原代码 generate_batch 直接丢弃不足一个 batch 的论文，这样会有论文没有预测值）。
5. 训练时 `sess.run([optimizer, loss])` 一次取出（原代码在 optimizer 之后再前向一次只为了打印 loss，不影响参数更新）。
6. 设置随机种子 `tf.set_random_seed(seed)`、`np.random.seed(seed)`（原代码不设种子），用于多 seed 实验。
7. 预测取 `model.cvae.citation_pred_test`（与 main.py 保存的 `pred_test` 相同），评估时按原 `cal_metric` 把预测截到 >=0。

## 4. 未修改的文件
`rgcn.py`、`model_embedding.py`、`utils.py`、`main.py` 完全未改。
