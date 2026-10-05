"""MPCC：用 DPPDCC 官方预处理函数（utilis/s2orc.py）从 data/mpcc_ai/all_*.json 生成样本文件与异构图。
time_point = 2021：样本图只含 ≤ 2021 年的论文（DPPDCC 的设计）；2022–2024 年的被引仍由完整的 all_cite_dict 统计进标签；累计被引按 cut_time = 2021、time_length = 3。"""
import sys, time
sys.path.insert(0, ".")
from utilis.s2orc import get_sample_data, get_citation_accum, get_input_data
t0 = time.time()
P = "./data/mpcc_ai/"
get_sample_data(P, time_point=2021); print("sample_data", round(time.time() - t0), flush=True)
get_citation_accum(P, time_point=2021, time_length=3); print("citation_accum", round(time.time() - t0), flush=True)
get_input_data(P, subset=True); print("input_data (graph_sample)", round(time.time() - t0), flush=True)
print("PREP DONE")
