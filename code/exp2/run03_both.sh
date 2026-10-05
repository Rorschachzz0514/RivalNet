#!/bin/bash
cd /path/to/mpcc/exp2
/path/to/conda/envs/mpcc/bin/python -u 03_evaluate.py --version v4 > logs/03_v4.log 2>&1 &
/path/to/conda/envs/mpcc/bin/python -u 03_evaluate.py --version v5 > logs/03_v5.log 2>&1
wait
echo "EXIT=$?"
