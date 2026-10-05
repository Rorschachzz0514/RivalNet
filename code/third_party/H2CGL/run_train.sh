#!/bin/bash
# 用法：run_train.sh <data_source: mpcc_ai | mpcc_ai_all> [GPU]
# H2CGL 训练（README 默认超参数），按验证集 MALE 保存全局最优检查点；不跑官方 test_results（它在测试集上评估所有检查点）
# 随后只用该检查点对验证集 / 测试集逐篇预测（mpcc_predict）
cd /path/to/mpcc/third_party/H2CGL
DS=$1
G=${2:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}
echo "DS $DS GPU $G $(date)"
export CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPCC_NO_TEST=1
PY=/path/to/envs/dppdcc/bin/python
FLAGS="--cl_type label_aug_hard_negative --aug_type cg --encoder_type CGIN+RGAT --n_layers 4 --hn 2 --hn_method co_cite --graph_type specter2"
$PY -u main.py --phase H2CGL --data_source $DS $FLAGS || { echo "EXIT=1"; exit 1; }
echo "train done $(date)"
$PY -u main.py --phase mpcc_predict --model H2CGL --data_source $DS $FLAGS
echo "EXIT=$?"
