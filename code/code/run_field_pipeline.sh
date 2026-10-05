#!/bin/bash
# 用法: run_field_pipeline.sh <子领域编号> <目录后缀>     实验 6：在其他子领域上重建实验 0 v2 与预测样本的全部数据
set -e
export MPCC_SUBFIELD=$1 MPCC_TAG=$2
cd /path/to/mpcc/code
PY=/path/to/conda/envs/mpcc/bin/python
G=0,1,2,3,4,5,7
for step in "20_exp0_base.py" "21_exp0_pairs_coupling.py" "22_exp0_pairs_knn.py --gpus $G" "23_exp0_pairs_merge.py" \
            "26_exp0_v2_clean.py" "27_exp0_v2_clusters.py --gpu 0" "28_exp0_v2_sim_counts.py --gpus $G" \
            "29_exp0_v2_sim_pairs.py --gpus $G" "30_pred_samples.py --gpu 0" "31_pred_rivals.py --gpus 1,2,3"; do
  echo "=== $(date +%H:%M:%S) $step"
  $PY -u $step 2>&1 | grep -v -E "^\[gpu.*(year|focal/s)" | tail -4
done
echo "PIPELINE EXIT=0"
