cd /path/to/mpcc/exp2
P=/path/to/conda/envs/mpcc/bin/python
$P -u 02_mpcnet.py --configs IN_r1_rel,IN_r1_tc_strict,IN_r1_tc_plus,IN_r1_luce --seeds 1 --gpus 0,1,2,6 --max_epochs 1 --tag _smoke &
$P -u 02_mpcnet.py --configs IN_r1_coh1,IN_r1_xd3,IN_r1_all --seeds 1 --gpus 0,1,2 --max_epochs 1 --tag _smoke2 &
$P -u 02_mpcnet.py --configs AS_r06_1 --seeds 1 --gpus 4 --max_epochs 2 --tag _eqchk &
wait
echo EXIT
