#!/bin/bash
cd /path/to/mpcc/third_party/DPPDCC
/path/to/envs/dppdcc/bin/python -u data_processor.py --phase make_data_graph --data_source mpcc_ai --graph specter2
echo "EXIT=$?"
