#!/bin/bash
# PLM-FT 其他学科：每张卡先肿瘤学再应用数学，三个种子分在三张卡上
P=/path/to/conda/envs/mpcc/bin/python
S=/path/to/mpcc/exp2/11_plm_finetune.py
L=/path/to/mpcc/baselines/plm_ft
for pair in "1 0" "2 1" "3 3"; do
  set -- $pair
  ( $P -u $S --target all --tag _sf2730 --seed $1 --gpu $2 > $L/run_sf2730_s$1.log 2>&1; $P -u $S --target all --tag _sf2604 --seed $1 --gpu $2 > $L/run_sf2604_s$1.log 2>&1 ) &
done
wait
echo "EXIT=$?"
