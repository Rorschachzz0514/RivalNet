#!/bin/bash
# H2CGL 图数据准备：① 各年自我中心子图 ② 各阶段组合快照图 ③ 固定的 cg 增强训练图；mpcc_ai_all 共用同一套图（符号链接）
cd /path/to/mpcc/third_party/H2CGL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
PY=/path/to/envs/dppdcc/bin/python
FLAGS="--model H2CGL --cl_type label_aug_hard_negative --aug_type cg --encoder_type CGIN+RGAT --n_layers 4 --hn 2 --hn_method co_cite --graph_type specter2"
for y in 2017 2018 2019 2020 2021; do $PY -u mpcc_prep_ego.py $y > logs/ego_$y.log 2>&1 & done
wait
for y in 2017 2018 2019 2020 2021; do grep -q "EGO DONE" logs/ego_$y.log || { echo "ego $y failed"; echo "EXIT=1"; exit 1; }; done
echo "ego done $(date)"
for ph in train val test; do MPCC_PHASE=$ph $PY -u main.py --phase mpcc_deal_graphs --data_source mpcc_ai $FLAGS > logs/deal_$ph.log 2>&1 & done
wait
for ph in train val test; do grep -q "Done main" logs/deal_$ph.log || { echo "deal $ph failed"; echo "EXIT=1"; exit 1; }; done
echo "deal done $(date)"
$PY -u main.py --phase mpcc_cl_data --data_source mpcc_ai $FLAGS > logs/cl_data.log 2>&1 || { echo "cl failed"; echo "EXIT=1"; exit 1; }
echo "cl done $(date)"
for f in CTSGCN_graphs CTSGCN_graphs_train.job CTSGCN_graphs_val.job CTSGCN_graphs_test.job CTSGCN_graphs_cl_cg_0.1.job; do
  ln -sf /path/to/mpcc/third_party/H2CGL/checkpoints/mpcc_ai/$f checkpoints/mpcc_ai_all/$f
done
ls -la checkpoints/mpcc_ai checkpoints/mpcc_ai_all
echo "EXIT=0"
