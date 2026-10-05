#!/bin/bash
# 【01】实验 9 · 效率与可扩展性：定版 MPC-Net 在 10% / 25% / 50% / 100% 的训练格子上各训练 5 轮（单卡、独占时测量），
#      记录每轮耗时、显存峰值、验证 / 测试集预测耗时（写在 exp2/runs/MPCNet5_l1_eff*/summary.json）
# 用法: 01_efficiency.sh <GPU 编号>
G=${1:-0}
cd /path/to/mpcc/exp2
for f in 0.1 0.25 0.5 1.0; do
  rm -rf runs/MPCNet5_l1_eff${f}_s1
  /path/to/conda/envs/mpcc/bin/python -u 02_mpcnet.py --configs MPCNet5_l1 --seeds 1 --gpus $G --tag _eff$f --max_epochs 5 --train_frac $f 2>&1 | grep -E "DONE|FAILED"
done
/path/to/conda/envs/mpcc/bin/python -u /path/to/mpcc/exp9/02_summarize.py
echo "EXIT=$?"
