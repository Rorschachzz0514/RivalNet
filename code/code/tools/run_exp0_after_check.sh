#!/bin/bash
# 等完整检查结束后再建实验 0 子集 (两者都大量读盘, 不同时跑)
cd /path/to/mpcc
while kill -0 $(cat logs/check_full.pid) 2>/dev/null; do sleep 10; done
grep -E "RESULT" logs/check_full.log
/path/to/conda/envs/mpcc/bin/python -u /path/to/mpcc/code/02_build_exp0.py --threads 160
