#!/bin/bash
cd /path/to/mpcc/third_party/H2CGL
/path/to/envs/dppdcc/bin/python -u mpcc_prep_hn.py mpcc_ai && /path/to/envs/dppdcc/bin/python -u mpcc_prep_hn.py mpcc_ai_all
echo "EXIT=$?"
