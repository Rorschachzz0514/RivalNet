#!/bin/bash
# 【01】实验 8 R7：用剔除自引的被引口径重跑实验 0 v2 的 11–13、15、16（结果在 exp0_analysis/results_v2_noself/）
export EXP0_V2_Y=noself EXP0_V2_RES=results_v2_noself
cd /path/to/mpcc/exp0_analysis && mkdir -p results_v2_noself
for s in 11_v2_build_tables 12_v2_test_macro 13_v2_test_micro 15_v2_test_twins 16_v2_test_crossdomain; do
  /path/to/conda/envs/mpcc/bin/python -u $s.py > logs/v2/${s}_noself.log 2>&1; echo "$s exit=$?"
done
/path/to/conda/envs/mpcc/bin/python -u /path/to/mpcc/exp8/02_compare.py
echo "EXIT=$?"
