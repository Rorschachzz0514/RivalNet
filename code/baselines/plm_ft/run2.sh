#!/bin/bash
# PLM-FT：3 个种子分在 3 张卡上并行；每张卡先跑主口径，再跑语料内口径
P=/path/to/conda/envs/mpcc/bin/python
S=/path/to/mpcc/exp2/11_plm_finetune.py
L=/path/to/mpcc/baselines/plm_ft
for pair in "4 2" "5 0"; do
  set -- $pair
  ( $P -u $S --target all --seed $1 --gpu $2 > $L/run_all_s$1.log 2>&1; $P -u $S --target dp --seed $1 --gpu $2 > $L/run_dp_s$1.log 2>&1 ) &
done
wait
echo "EXIT=$?"
