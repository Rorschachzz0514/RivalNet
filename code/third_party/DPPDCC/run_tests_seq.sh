#!/bin/bash
# 额外种子的测试依次运行（/dev/shm 只有 32 GB，并行测试会写满共享内存）
cd /path/to/mpcc/third_party/DPPDCC
ulimit -n 1048576
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for x in "mpcc_ai_s2 2" "mpcc_ai_s3 3" "mpcc_ai_all_s2 2" "mpcc_ai_all_s3 3"; do
  set -- $x
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
  CUDA_VISIBLE_DEVICES=$G /path/to/envs/dppdcc/bin/python -u main.py --phase test_results --model DPPDCC --data_source $1 --graph_type specter2 --model_seed $2
  echo "TEST $1 EXIT=$?"
done
echo "EXIT=0"
