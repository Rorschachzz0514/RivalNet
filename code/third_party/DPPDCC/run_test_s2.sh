#!/bin/bash
cd /path/to/mpcc/third_party/DPPDCC
ulimit -n 1048576
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=3
/path/to/envs/dppdcc/bin/python -u main.py --phase test_results --model DPPDCC --data_source mpcc_ai_s2 --graph_type specter2 --model_seed 2
echo "EXIT=$?"
