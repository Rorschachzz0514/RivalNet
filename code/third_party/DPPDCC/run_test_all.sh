#!/bin/bash
# DPPDCC 只做测试（加载按验证集选出的全局最优检查点）
cd /path/to/mpcc/third_party/DPPDCC
ulimit -n 1048576
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "GPU $G"
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/envs/dppdcc/bin/python -u main.py --phase test_results --model DPPDCC --data_source mpcc_ai_all --graph_type specter2
echo "EXIT=$?"
