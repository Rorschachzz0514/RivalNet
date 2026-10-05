#!/bin/bash
cd /path/to/mpcc/exp2
/path/to/conda/envs/mpcc/bin/python -u 02_mpcnet.py --configs MPCNet5,MPCNet5_l1,B_decomp,B_no_cell,B_no_rivals,B_no_share_loss,B_meanpool,B_coupled,B_top10,B_no_text --seeds 1,2,3,4,5 --gpus 0,1,2
echo "EXIT=$?"
