#!/bin/bash
cd /path/to/mpcc/exp2
PY=/path/to/conda/envs/mpcc/bin/python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u 02_mpcnet.py --configs MPCNet5_after1y,B_no_rivals_after1y --seeds 1,2,3,4,5 --gpus 0,1,2,3,4
echo "TRAIN FAILED=$(grep -c FAILED /path/to/mpcc/logs/exp2_after1y.log)"
$PY -u 03_evaluate.py --version after1y > logs/03_after1y.log 2>&1
echo "EXIT=$?"
