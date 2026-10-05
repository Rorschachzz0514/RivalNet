"""
实验 1 的统一配置：数据路径、结果路径、特征分组。所有实验 1 脚本都 import 这个文件。

  SAMPLES   预测样本表（data_pipeline【30】）；可用环境变量 EXP1_SAMPLES 覆盖
  RES_DIR   结果目录（脚本所在目录下的 results/）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.environ.get("EXP1_SAMPLES", "/path/to/mpcc/subsets/pred_v2/samples.parquet")
EMB = os.environ.get("EXP1_EMB", "/path/to/mpcc/subsets/exp0/emb_sample.npy")
NAIP_DIR = "/path/to/data/models/NAIP_full"   # 只含完整权重的链接目录（去掉 adapter 文件，否则 transformers 会去下载基座模型）
PAPERS = os.environ.get("EXP1_PAPERS", "/path/to/mpcc/subsets/exp0_v2/papers.parquet")
RES_DIR = os.path.join(HERE, os.environ.get("EXP1_RES", "results"))   # 实验 6：results_sf<子领域>
THREADS = 128
SEED = 2026

# 特征分组（预先登记，见 实验1说明.md 第 3 节）
META = ["n_authors", "n_institutions", "n_countries", "n_refs", "any_oa", "has_abstract", "has_funding", "n_grants",
        "pub_month", "date_precision", "venue_at_T", "has_preprint_at_T", "topic_score", "topic2_score",
        "auth_cites_max", "auth_cites_mean", "auth_npap_max", "auth_npap_mean", "first_auth_cites", "first_auth_npap",
        "share_new_authors", "venue_age1_mean", "venue_n_prev", "venue_y3_mean"]
HEAT_TOPIC = ["topic_age1_mean", "topic_n_prev", "topic_y3_mean", "topic_supply_Y", "topic_supply_Ym1", "topic_supply_Ym2",
              "topic_dem_Y", "topic_dem_Ym2", "topic_inflow_Y", "topic_inflow_Ym1", "topic_inflow_Ym2", "topic_age",
              "g_topic_inflow", "g_topic_supply"]
HEAT_SUB = ["sub_age1_mean", "sub_n_prev", "sub_y3_mean", "sub_supply_Y", "sub_supply_Ym1", "sub_supply_Ym2", "sub_refs_Y",
            "sub_inflow_Y", "sub_inflow_Ym1", "sub_inflow_Ym2", "g_sub_inflow", "g_sub_supply"]
HEAT2 = ["topic2_supply_Y", "topic2_inflow_Y"]
COMP = ["c_m2_95", "c_m1_95", "c_m2_90", "c_m1_90", "c_pb365_95", "rival_cites_m1", "rival_cites_m1_max",
        "n_preempted", "preempt_min_days", "s3_m1", "n_cand_ym1"]
CATEG = ["date_precision", "venue_at_T"]
N_PCA = 64
NAIP_ADAPTER = "/path/to/data/models/NAIP"         # LoRA adapter（q/v 投影 + score 头）所在目录
