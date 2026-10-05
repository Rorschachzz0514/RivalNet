#!/bin/bash
cd /path/to/mpcc/exp1

/path/to/conda/envs/mpcc/bin/python -u 03_naip.py --gpus 0,1,2,3,4,5,7 --batch 12
echo "EXIT=$?"
