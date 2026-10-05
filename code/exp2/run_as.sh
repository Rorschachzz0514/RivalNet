#!/bin/bash
cd /path/to/mpcc/exp2
/path/to/conda/envs/mpcc/bin/python -u 05_autosearch.py --rounds $1 --space space_r4.json
echo "EXIT=$?"
