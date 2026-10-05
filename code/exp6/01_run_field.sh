#!/bin/bash
# 【01】实验 6：在一个学科上依次运行 实验 0 v2 核心检验（11–13、15、16、18）、实验 1 对比方法（01–02）、
#      实验 2 定版结构与去掉对手的消融（01–03，3 个种子），全部用环境变量指向该学科的数据与结果目录。
# 用法: 01_run_field.sh <目录后缀，如 _sf2730> <子课题数 K>
# 前提: data_pipeline/run_field_pipeline.sh 已为该学科生成数据
TAG=$1; K=$2
PY=/path/to/conda/envs/mpcc/bin/python
echo "=== $(date +%H:%M:%S) 实验 0 v2（$TAG，K=$K）"
export EXP0_V2_DATA=/path/to/mpcc/subsets/exp0${TAG}_v2 EXP0_V2_RES=results_v2${TAG} EXP0_V2_K=$K
cd /path/to/mpcc/exp0_analysis && mkdir -p results_v2${TAG}
for s in 11_v2_build_tables 12_v2_test_macro 13_v2_test_micro 15_v2_test_twins 16_v2_test_crossdomain; do
  $PY -u $s.py > logs/v2/${s}${TAG}.log 2>&1; echo "$s exit=$?"
done
$PY -u 18_v2_posthoc_wave.py --workers 64 > logs/v2/18${TAG}.log 2>&1; echo "18 exit=$?"
echo "=== $(date +%H:%M:%S) 实验 1 对比方法"
export EXP1_SAMPLES=/path/to/mpcc/subsets/pred${TAG}_v2/samples.parquet EXP1_EMB=/path/to/mpcc/subsets/exp0${TAG}/emb_sample.npy
export EXP1_PAPERS=/path/to/mpcc/subsets/exp0${TAG}_v2/papers.parquet EXP1_RES=results${TAG}
cd /path/to/mpcc/exp1 && mkdir -p results${TAG}
$PY -u 01_variance.py > logs/01${TAG}.log 2>&1; echo "exp1 01 exit=$?"
$PY -u 02_models.py --gpu 0 > logs/02${TAG}.log 2>&1; echo "exp1 02 exit=$?"
echo "=== $(date +%H:%M:%S) 实验 2"
export EXP2_TAG=$TAG
cd /path/to/mpcc/exp2 && mkdir -p results${TAG} runs${TAG}
$PY -u 01_build_tensors.py > logs/01${TAG}.log 2>&1; echo "exp2 01 exit=$?"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u 02_mpcnet.py --configs MPCNet5_l1,B_no_rivals --seeds 1,2,3 --gpus 0,1,2,3,4,5 > logs/02${TAG}.log 2>&1; echo "exp2 02 exit=$? FAILED=$(grep -c FAILED logs/02${TAG}.log)"
$PY -u 03_evaluate.py --version field > logs/03${TAG}.log 2>&1; echo "exp2 03 exit=$?"
echo "FIELD EXIT=0"
