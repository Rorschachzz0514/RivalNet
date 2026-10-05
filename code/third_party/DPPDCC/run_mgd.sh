#!/bin/bash
cd /path/to/mpcc/third_party/DPPDCC
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/envs/dppdcc/bin/python -u main.py --phase get_model_graph_data --data_source mpcc_ai --model DDHGCNSCL --graph_type specter2
echo "STEP1 EXIT=$?"
/path/to/envs/dppdcc/bin/python -u main.py --phase get_model_graph_data --data_source mpcc_ai --model DPPDCC --graph_type specter2
echo "EXIT=$?"
