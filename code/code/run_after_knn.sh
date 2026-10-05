#!/bin/bash
# 等【22】S2 近邻结束后, 同时启动【23】合并 (CPU) 与【24】相似计数 (8 GPU)
cd /path/to/mpcc
while kill -0 $(cat logs/02c_knn.pid) 2>/dev/null; do sleep 10; done
grep -q "KNN DONE" logs/02c_knn.log || { echo "S2 未成功结束, 不继续"; exit 1; }
code/start_bg.sh 24_sim_counts /path/to/conda/envs/mpcc/bin/python -u /path/to/mpcc/code/24_exp0_sim_counts.py --gpus 0,1,2,3,4,5,6,7
/path/to/conda/envs/mpcc/bin/python -u /path/to/mpcc/code/23_exp0_pairs_merge.py --threads 160
