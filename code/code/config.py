"""数据准备的统一配置 (服务器路径). 本地只编辑, 在服务器 /path/to/mpcc/code/ 运行.

思路: 先建一个覆盖所有实验的总库 (union), 每个实验再从总库里切自己的子集, 不再重扫原始快照.
"""
import os

# ---- 输入: OpenAlex 全量快照 (RELEASE 2024-12-31), 一行一条 JSON ----
OA_DIR = "/data/openalex/jsonl"
WORKS_TXT = os.path.join(OA_DIR, "works.txt")

# ---- 输出根目录 ----
ROOT = "/path/to/mpcc"
UNION_DIR = os.path.join(ROOT, "union")      # 总库: union/detail, union/lite, union/meta
SUBSET_DIR = os.path.join(ROOT, "subsets")   # 各实验从总库切出的子集
TMP_DIR = os.path.join(ROOT, "tmp")          # DuckDB 溢写目录
LOG_DIR = os.path.join(ROOT, "logs")

# ---- detail: 保存完整字段的论文范围 (按 primary_topic 的 field 判定) ----
# 17 Computer Science (含 AI 子领域 1702)   主学科, 实验 0-9
# 27 Medicine / 26 Mathematics / 16 Chemistry  实验 6 跨学科, 5-5 新冠
# 31 Physics and Astronomy / 23 Environmental Science  实验 6-2 与 Funding 六学科对照
DETAIL_FIELDS = {17, 27, 26, 16, 31, 23}
DETAIL_YEAR_MIN, DETAIL_YEAR_MAX = 2016, 2024
AI_SUBFIELD = int(os.environ.get("MPCC_SUBFIELD", "1702"))   # 实验 6 跨学科：用环境变量换成其他子领域
SUBSET_TAG = os.environ.get("MPCC_TAG", "")                    # 子集目录后缀，例如 "_sf2730"；AI（默认）为空
EXP0 = "exp0" + SUBSET_TAG
EXP0_V2 = "exp0" + SUBSET_TAG + "_v2"
PRED_V2 = "pred" + SUBSET_TAG + "_v2"

# ---- lite: 全库每篇论文都保存精简字段 (方向/年份/逐年被引/作者); ----
# 参考文献列表只对这之后发表的论文保存 (引用边只看 2016 年及以后的施引论文, 留一年余量)
REFS_YEAR_MIN = 2015

# ---- 并行: 实测 16 进程 185 MB/s, 32 进程 314 MB/s ----
N_WORKERS = 32
FLUSH_ROWS = 100_000   # 每个 worker 每攒这么多行写一个 parquet 分片
