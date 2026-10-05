#!/bin/bash
# 用法: run_hints.sh <hints_mpcc.py 参数...>
cd /path/to/mpcc/third_party/HINTS_code/src
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=8
/path/to/envs/hints/bin/python -u hints_mpcc.py "$@"
echo "EXIT=$?"
