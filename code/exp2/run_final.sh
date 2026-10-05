#!/bin/bash
# 最终方案：A 组 = AS_r06_1（2017–2019 训练、2020 早停）10 个种子；B 组 = 2017–2020 重训，轮数照搬 A 组同种子的最佳轮
cd /path/to/mpcc/exp2
PY=/path/to/conda/envs/mpcc/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -u 02_mpcnet.py --configs AS_r06_1 --seeds 4,5,6,7,8,9,10 --gpus 0,1,2,3,4,5,7
$PY -c "
import json
a = json.load(open(\"autosearch/configs/AS_r06_1.json\"))
for s in range(1, 11):
    ep = json.load(open(f\"runs/AS_r06_1_s{s}/summary.json\"))[\"best_epoch\"]
    json.dump({**a, \"trainval\": True, \"fixed_ep\": ep}, open(f\"autosearch/configs/FINAL_B{s}.json\", \"w\"))
    print(s, ep)
"
for s in 1 2 3 4 5 6 7 8 9 10; do
  g=$(( (s - 1) % 7 )); [ $g -eq 6 ] && g=7
  $PY -u 02_mpcnet.py --configs FINAL_B$s --seeds $s --gpus $g &
done
wait
echo "EXIT=$?"
