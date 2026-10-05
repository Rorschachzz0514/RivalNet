#!/bin/bash
# 【22】收尾 (跳过已完成的 GPU 块, 只做校准与合并) -> 同时启动【24】(8 GPU) 与【23】(CPU)
cd /path/to/mpcc
PY=/path/to/conda/envs/mpcc/bin/python
$PY -u code/22_exp0_pairs_knn.py --gpus 0,1,2,3,4,5,6,7 --k 100 || { echo "22 失败"; exit 1; }
grep -q . subsets/exp0/pairs_knn.parquet || { echo "pairs_knn 缺失"; exit 1; }
code/start_bg.sh 24_sim_counts $PY -u /path/to/mpcc/code/24_exp0_sim_counts.py --gpus 0,1,2,3,4,5,6,7
$PY -u code/23_exp0_pairs_merge.py --threads 160
