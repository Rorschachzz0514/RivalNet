#!/bin/bash
# DPPDCC 训练：自动选空闲显存最多的 GPU
cd /path/to/mpcc/third_party/DPPDCC
ulimit -n 1048576
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "GPU $G"
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/envs/dppdcc/bin/python -u main.py --phase DPPDCC --data_source mpcc_ai_all --graph_type specter2
echo "EXIT=$?"
