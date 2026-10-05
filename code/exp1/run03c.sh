#!/bin/bash
cd /path/to/mpcc/exp1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /path/to/conda/envs/mpcc/bin/python -u 03_naip.py --gpus 0,1,2,1,2,0,7 --batch 24
echo "EXIT=$?"
