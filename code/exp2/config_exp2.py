"""
实验 2 的统一配置：路径、张量目录、超参数。所有实验 2 脚本都 import 这个文件。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("EXP2_TAG", "")              # 实验 6 跨学科：例如 "_sf2730"；AI 为空
SAMPLES = f"/path/to/mpcc/subsets/pred{TAG}_v2/samples.parquet"
RIVALS = f"/path/to/mpcc/subsets/pred{TAG}_v2/rivals.parquet"
EMB = f"/path/to/mpcc/subsets/exp0{TAG}/emb_sample.npy"
EMB_META = f"/path/to/mpcc/subsets/exp0{TAG}_v2/emb_meta.parquet"
TENSORS = f"/path/to/mpcc/exp2/tensors{TAG}"
RUNS = os.path.join(HERE, f"runs{TAG}")
RES_DIR = os.path.join(HERE, f"results{TAG}")
EXP1_RES = f"/path/to/mpcc/exp1/results{TAG}"       # 实验 1 的对比方法预测
SEEDS = [1, 2, 3, 4, 5]
K_RIVALS = 50

# 论文特征（与实验 1 的 L1 + 第二方向热度一致；方向 / 子课题热度进入需求塔，不进入论文编码器）
PAPER_NUM = ["n_authors", "n_institutions", "n_countries", "n_refs", "any_oa", "has_abstract", "has_funding", "n_grants",
             "pub_month", "has_preprint_at_T", "topic_score", "topic2_score", "auth_cites_max", "auth_cites_mean",
             "auth_npap_max", "auth_npap_mean", "first_auth_cites", "first_auth_npap", "share_new_authors",
             "venue_age1_mean", "venue_n_prev", "venue_y3_mean", "topic2_supply_Y", "topic2_inflow_Y",
             "topic_age1_mean", "topic_y3_mean", "topic_inflow_Y", "topic_supply_Y"]
PAPER_CAT = ["date_precision", "venue_at_T"]
LOG_COLS = ["n_authors", "n_institutions", "n_countries", "n_refs", "n_grants", "auth_cites_max", "auth_cites_mean",
            "auth_npap_max", "auth_npap_mean", "first_auth_cites", "first_auth_npap", "venue_n_prev", "topic2_supply_Y",
            "topic2_inflow_Y", "topic_inflow_Y", "topic_supply_Y"]
# 格子（训练期子课题 × 年份）特征
CELL_LOG = ["sub_supply_Y", "sub_supply_Ym1", "sub_supply_Ym2", "sub_inflow_Y", "sub_inflow_Ym1", "sub_inflow_Ym2", "sub_refs_Y",
            "topic_supply_Y", "topic_supply_Ym1", "topic_supply_Ym2", "topic_inflow_Y", "topic_inflow_Ym1", "topic_inflow_Ym2",
            "topic_dem_Y", "topic_dem_Ym2"]
CELL_RAW = ["sub_age1_mean", "sub_y3_mean", "topic_age1_mean", "topic_y3_mean"]
RIVAL_VENUES = ["journal", "conference", "book series", "arxiv", "repository", "ebook", "none", "preprint_other"]
