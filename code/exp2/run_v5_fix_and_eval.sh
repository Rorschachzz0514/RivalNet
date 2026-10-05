#!/bin/bash
cd /path/to/mpcc/exp2
PY=/path/to/conda/envs/mpcc/bin/python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u 02_mpcnet.py --configs MPCNet5,MPCNet5_l1,B_decomp,B_no_cell,B_no_rivals,B_no_share_loss,B_meanpool,B_coupled,B_top10,B_no_text --seeds 1,2,3,4,5 --gpus 0,1,2,3,4,5,7
echo "TRAIN_EXIT=$?"
n=$(ls runs/*/preds.parquet | wc -l); echo "preds files: $n"
$PY -u 03_evaluate.py --version v4 > logs/03_v4.log 2>&1 &
$PY -u 03_evaluate.py --version v5 > logs/03_v5.log 2>&1
wait
echo "EXIT=$?"
