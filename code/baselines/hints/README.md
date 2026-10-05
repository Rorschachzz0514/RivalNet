# HINTS 对比实验（WWW 2021, Jiang et al.）

在 MPCC 的 AI 语料（OpenAlex, exp0_v2）上运行 **HINTS 原始 TensorFlow 代码**，与 MPC-Net 对比。

## 输出
| 文件 | 说明 |
|---|---|
| `preds_all.parquet` | 目标 "all"（y3 = Y+1..Y+3 全来源引用之和）的预测；列 `paper_id(int64), split('val'/'test'), seed(int), pred(float)`，`pred` = 预测的 log1p(3 年引用和) |
| `preds_dp.parquet` | 目标 "dp"（语料内引用，`tensors_v6dp/y.npy` 三年之和）的预测，格式同上 |
| `report.json` | 每个 seed 的 MALE、预测均值、seed 集成、最佳 epoch、运行时间、泄漏检查 |
| `runs/hints_{all,dp}_seed{0,1,2}.{npz,json}` | 每次运行的原始输出（val/test 的 3 年累计对数预测）和验证曲线 |
| `data/` | 转换后的 HINTS 输入（年度快照图、对齐、index table、标签），由 `convert_mpcc_to_hints.py` 生成 |

覆盖：val 73,400 篇、test 84,566 篇，每个 seed 全部有预测（无缺失）。seed = 0,1,2；seed 集成 = 3 个 seed 的 `pred` 按论文取平均（未写入 parquet，结果见下表）。

## 结果（MALE = mean |pred − log1p(y)|）

| 目标 | seed | 最佳 epoch (val 选) | val MALE | test MALE |
|---|---|---|---|---|
| all | 0 | 460（660 早停） | 0.9216 | 0.9405 |
| all | 1 | 740（到 800 上限） | 0.9149 | 0.9197 |
| all | 2 | 720（到 800 上限） | 0.9120 | 0.9128 |
| all | **均值 ± std** | | **0.9162 ± 0.0049** | **0.9244 ± 0.0144** |
| all | **seed 集成** | | **0.9046** | **0.9136** |
| dp | 0 | 420（620 早停） | 0.7439 | 0.7393 |
| dp | 1 | 760（到 800 上限） | 0.7385 | 0.7031 |
| dp | 2 | 760（到 800 上限） | 0.7354 | 0.7247 |
| dp | **均值 ± std** | | **0.7393 ± 0.0043** | **0.7224 ± 0.0182** |
| dp | **seed 集成** | | **0.7298** | **0.7133** |

参考：用训练集 log1p(y) 中位数作常数预测，all 的 val/test MALE = 1.0476 / 1.0076，dp 的 = 0.9458 / 0.8597。

## 数据转换（`convert_mpcc_to_hints.py`，逻辑照搬官方 `src/preprocess/data_aminer*.ipynb`）
* **年度快照（"individual" 图）** graph_y：节点 = first_public_year == y 的语料论文 + 它们的全部对象；有向边 论文→对象（行 = 论文），
  4 种关系 + 自环，每个关系行归一化（同官方 `normalize`）。
  - P1P：论文的 `refs`（OpenAlex 参考文献，**包括语料外的被引文献**，与 AMiner 用全部 DBLP 引用一致）；
    去掉了"引用了 first_public_year 更晚的语料论文"的边（OpenAlex 日期噪声，防止未来信息）。
  - P1A：`author_ids`。
  - P1V：venue。样本论文用 `samples.source_at_T`（T 时刻的 venue）；其余论文若 published_year ≤ first_public_year 用 `source_id`，
    否则（当年只是预印本、正式发表在之后）不连 venue 边，避免使用 T 之后才知道的 venue。
  - P1K（关键词）：AMiner 的 keyword 实际是 MAG 的 FOS（fields of study）；OpenAlex 的 `concept_names` 正是 MAG FOS 的延续，
    因此用 `concept_names` 作为关键词节点（全语料 3.3 万个 concept）。
  - 规模：2016–2020 每年 10.3–14.0 万篇论文，快照节点 92–145 万，P1P 边 143–287 万。
* **节点特征**：同官方 `data_aminernf.ipynb`，每个实体一个全局固定的 4 维随机向量 + 类型偏移（论文 −1，关键词 0，作者 +1，venue +2），种子 2021。
* **对齐**：相邻两年快照的共同节点下标（同 `data_aminer4.ipynb`）。
* **焦点论文 index table**：对目标年 Y 的每篇样本论文，在 Y−3、Y−2、Y−1 三个快照里查它的参考文献（最多 100 个）、作者（20）、venue（1，用 source_at_T）、
  concept（15，按得分顺序），找不到的跳过、不足用 −1 补（同 `construct_index_from_one_df`）。焦点论文本身从不作为被查节点，其被引信息不进入输入。
* **标签**：累计对数引用 log(1 + 累计引用) 在 t = 1,2,3 年（官方是 t = 1..5，见下文偏差 2）。
  - all：[y1, y1+y2, y3]（已核对 y3 = y1+y2+y3a 对所有样本成立）；
  - dp：`y.npy` 按 `meta.parquet.paper_id` 对齐后逐年累加。
  - **评估量**：HINTS 第 3 年的累计对数预测 = log(1 + Y+1..Y+3 引用和) = log1p(y3)，正好是评估目标；按原 `cal_metric` 截到 ≥0。无需额外换算。

## 训练设置
* 原始 TF 代码（`/path/to/mpcc/third_party/HINTS_code/src`，入口 `hints_mpcc.py`，模型直接 import 原 `model.py`），环境 `/path/to/envs/hints`
  （python 3.7 + tensorflow 1.15.5，CPU）。超参数保持原值：embedding 128，RGCN 64/128，GRU 50，lr 0.01 Adam，beta（对齐损失权重）0.5，
  dropout 照原代码 `tf.nn.dropout(H_1, 0.2)`（TF1 中即 keep_prob = 0.2，训练和推断都生效）。
* full batch：同官方 README，batch_size = 训练论文数（61,238），每个 epoch 一次迭代。
* 训练 2019 年论文（split = train 中 Y = 2019 的 61,238 篇），每 20 epoch 在 2020（val）上算 MALE，保存 val 最好的 epoch 的 val/test 预测；
  连续 10 次评估（200 epoch）不提升则早停，最多 800 epoch（官方 AMiner 为固定 700 epoch）。test 只在最后统计一次，不参与选择。
* 2 个目标 × 3 个 seed，各自独立训练（标签不同）。

## 与论文 / 原代码的偏差（全部列出）
1. **只用 2019 年训练，快照数 K = 3（原 5）**。语料最早只有 2016 年论文：2021（test）之前有 5 年快照，2020（val）之前只有 4 年，
   2019 之前只有 3 年，2017/2018 更少。HINTS 每个快照位置有独立的 RGCN 权重，训练/预测必须使用同样多的、位置含义一致的快照，
   因此统一取 Y−3..Y−1 三个快照，并且只能用 2019 年论文训练（2017、2018 年论文之前没有足够快照；与 DPPDCC 的做法相同）。
2. **预测序列长度 3（原 5）**：只有 Y+1..Y+3 的标签（2021 + 5 = 2026 不可观测）。累计引用曲线公式不变，只拟合 t = 1,2,3。
3. **反归一化用训练集的 label max/min**：原代码对测试集用测试集自身标签的 max/min 做 min-max 再反归一化，属于标签泄漏，已修正。
4. **增加验证集早停**（原代码无验证、固定 700 epoch）；最多 800 epoch。注意 all 和 dp 各有 2 个 seed 在 800 上限时验证集仍在缓慢下降
   （最佳 epoch 720–760），受 CPU 时间所限没有继续加大上限。
5. **CPU 运行**：TF1.15 的 GPU 版（CUDA 10）在 RTX 4090 上需要 PTX JIT，实测卡死且内存涨到 150 GB，放弃；
   因此对 `model_imputed.py` 做了 CPU 兼容补丁（−1 补位查表返回 0 向量，与 GPU 上原行为完全一致）。
6. 预测时最后一个不满 batch 循环补齐后丢弃补齐部分（原代码会丢掉不足一个 batch 的论文）；训练时 loss 与 optimizer 同一次 sess.run 取出；设置随机种子。
7. 数据层面：关键词用 OpenAlex concepts（AMiner 用 MAG FOS）；venue 用 T 时刻 venue；删掉"引用更晚论文"的少量边。
所有代码改动见 `/path/to/mpcc/third_party/HINTS_code/MPCC_COMPAT_PATCHES.md`（原始文件备份在 `orig_src_backup/`），均不改变算法。

## 运行时间
* 数据转换约 1 分钟（MPCC 环境）。
* 训练在共享服务器 CPU 上 6 个任务并行（每个 8 线程；服务器负载 100–500），单个 epoch 约 15–45 秒；
  每次运行 274–347 分钟（all: 294 / 347 / 329 min；dp: 274 / 346 / 341 min），总墙钟时间约 5.8 小时。

## 自检
* **标签对齐**：`assemble_hints.py` 用 `samples.parquet`（y3）和 `y.npy`+`meta.parquet`（dp）按 `paper_id`+`split` 重新 join 计算 MALE，
  与转换时写入的标签逐篇一致（`np.allclose` 通过）；每个 seed 的 val/test 论文数分别为 73,400 / 84,566，paper_id 无重复。
* **无未来信息**：test（Y = 2021）的输入只用 2018、2019、2020 三个快照；这些快照中作为"行"（有出边）的论文 first_public_year 最大为 2020，
  被引用的语料论文年份最大为 2020；焦点论文自身的引用（任何年份）都不在输入中，只用其参考文献/作者/T 时刻 venue/concepts。
  val（2020）只用到 2019 年为止的快照，train（2019）只用到 2018 年为止。
* **尺度**：val 上 mean(pred) vs mean(log1p(y))：all 1.708 / 1.714 / 1.724 vs 1.719；dp 1.134 / 1.045 / 1.113 vs 1.164。
  test 上 all 预测均值 1.73 高于真实 1.54（2021 年论文的 log1p(y3) 均值明显低于 2019–2020，模型从 2019 年训练集学到的水平偏高），dp 预测 1.09 vs 真实 1.02。
