"""
实验 0 的统一配置：数据路径、结果路径、阈值与时间窗口。所有实验 0 脚本都 import 这个文件。

  DATA_DIR  实验 0 子集（服务器 /path/to/mpcc/subsets/exp0；本地副本 MPCC/data/exp0）
            可用环境变量 EXP0_DATA 覆盖，例如在本地运行时
  RES_DIR   结果输出目录（脚本所在目录下的 results/）；可用环境变量 EXP0_RES 覆盖
  子样本变体（【25】生成）：EXP0_DATA=/path/to/mpcc/subsets/exp0_jc EXP0_RES=results_jc EXP0_LABEL="只保留期刊 + 会议"
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("EXP0_DATA", "/path/to/mpcc/subsets/exp0")
RES_DIR = os.path.join(HERE, os.environ.get("EXP0_RES", "results"))
LABEL = os.environ.get("EXP0_LABEL", "全部论文")
TMP_DIR = os.environ.get("EXP0_TMP", "/path/to/mpcc/tmp")

FOCAL_YEARS = (2017, 2021)       # 焦点论文首次公开年份
Y_WINDOW = (1, 3)                # 因变量: 发表后第 1-3 个完整年份的被引
PRIOR_DAYS = 365                 # "发表前"的时间窗 (天)
TAU = 0.95                       # 主结果的 SPECTER2 相似度阈值 (约为随机同方向同时期论文对的 p99.9)
TAU_GRID = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
TWIN_SIM = 0.97                  # 撞车论文对: 相似度下限
TWIN_DAYS = 183                  # 撞车论文对: 首次公开相差天数上限
N_PLACEBO = 200                  # 假对手置换次数
THREADS = 128

# ---------------- 第二版（v2，预先登记见 实验0说明.md 第 7 节）----------------
V2_DATA = os.environ.get("EXP0_V2_DATA", "/path/to/mpcc/subsets/exp0_v2")
V2_RES = os.path.join(HERE, os.environ.get("EXP0_V2_RES", "results_v2"))
V2_MAIN_VENUES = ("journal", "conference", "book series", "arxiv")   # 焦点论文主分析集
V2_JC_VENUES = ("journal", "conference")
V2_K = int(os.environ.get("EXP0_V2_K", "2000"))   # 子课题数（主）；实验 6 应用数学用 1000
V2_KS = (1000, 2000, 5000)
V2_TAU = "0.95"                  # 主阈值（列名中的写法）
V2_TAUS = ["0.90", "0.91", "0.92", "0.93", "0.94", "0.95", "0.96", "0.97", "0.98"]
