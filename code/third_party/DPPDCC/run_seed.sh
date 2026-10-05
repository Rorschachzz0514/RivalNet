#!/bin/bash
# 用法：run_seed.sh <数据源> <模型种子> <GPU>：训练 + 测试（按验证集选出的全局最优检查点）
cd /path/to/mpcc/third_party/DPPDCC
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=$3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/envs/dppdcc/bin/python -u main.py --phase DPPDCC --data_source $1 --graph_type specter2 --model_seed $2
echo "TRAIN EXIT=$?"
/path/to/envs/dppdcc/bin/python -u main.py --phase test_results --model DPPDCC --data_source $1 --graph_type specter2 --model_seed $2
echo "EXIT=$?"
