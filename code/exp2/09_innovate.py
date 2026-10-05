"""
【09】实验 2 · 第七轮：由研究设想推出的新结构，在验证集上逐个检验（测试集不看）

用途
  在当前最好配置 AS_r06_1 上加一个或几个新结构（02_mpcnet.py 的 rel_rival / two_chan / luce / cohort / xdom），
  每个变体若干种子；与 AS_r06_1 同种子配对比较验证集 MALE 与子课题内 Spearman。
  GPU 调度：按显存余量给每张卡分若干任务位，任务结束就补下一个（动态队列）；显存不足失败的任务换卡重跑（最多 3 次）。

输入
  innov/<轮名>.json   {变体名: {在 AS_r06_1 上改动的配置项}}；可选键 "_base"：换一个基准配置（autosearch/configs 中的名字）
输出
  autosearch/configs/IN_<轮名>_<变体>.json、runs/IN_<轮名>_<变体>_s<种子>/
  innov/rounds.md（每轮追加一张表）、innov/<轮名>_summary.csv

用法
  python 09_innovate.py --round r1 --seeds 1,2,3,4
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from importlib import util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_exp2 as C

AS = os.path.join(HERE, "autosearch")
IN = os.path.join(HERE, "innov")
PY = "/path/to/conda/envs/mpcc/bin/python"
spec = util.spec_from_file_location("ev1", "/path/to/mpcc/exp1/04_evaluate.py")
EV = util.module_from_spec(spec)
spec.loader.exec_module(EV)


def free_slots(gb_per_task, max_per_gpu=4):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    g = {}
    for line in out:
        i, tot, used = [int(x) for x in line.split(",")]
        g[i] = max(0, min(max_per_gpu, int(((tot - used) / 1024 - 1) // gb_per_task)))
    return g


def done(name, s):
    return os.path.exists(os.path.join(C.RUNS, f"{name}_s{s}", "preds.parquet"))


def run_queue(tasks, gb):
    """动态队列：每张卡的任务位数按启动时的显存余量定；一个任务结束就在同一张卡上补下一个。"""
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    slots = free_slots(gb)
    print(f"  任务位：{slots}（共 {sum(slots.values())}）", flush=True)
    if sum(slots.values()) == 0:
        slots = {max(free_slots(1).items(), key=lambda kv: kv[1])[0]: 1}
    queue, running = list(tasks), []
    while queue or running:
        for g, n in slots.items():
            while queue and sum(1 for r in running if r[0] == g) < n:
                name, s = queue.pop(0)
                log = open(os.path.join(IN, "logs", f"{name}_s{s}.log"), "a")
                p = subprocess.Popen([PY, "-u", os.path.join(HERE, "02_mpcnet.py"), "--configs", name, "--seeds", str(s), "--gpus", str(g)],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
                running.append((g, name, s, p))
        time.sleep(10)
        running = [r for r in running if r[3].poll() is None]


def val_metrics(name, s, st):
    f = os.path.join(C.RUNS, f"{name}_s{s}", "preds.parquet")
    if not os.path.exists(f):
        return None
    p = pd.read_parquet(f, columns=["paper_id", "split", "pred"])
    p = p[p.split == "val"].set_index("paper_id").pred
    d = st.assign(pred=p.reindex(st.index)).reset_index()
    m = EV.metrics(d.assign(y=d.y3, ly=np.log1p(d.y3)))
    e = d.pred - np.log1p(d.y3)
    sm = json.load(open(os.path.join(C.RUNS, f"{name}_s{s}", "summary.json")))
    return {"MALE": m["MALE"], "bias": e.mean(), "MALE_db": (e - e.mean()).abs().mean(), "rho": m["Spearman_子课题内"], "ndcg": m["NDCG@10_子课题内"], "ep": sm["best_epoch"],
            "mem": sm.get("peak_mem_gb"), "sec": sm.get("sec")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--gb", type=float, default=7.0)
    args = ap.parse_args()
    os.makedirs(os.path.join(IN, "logs"), exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    spec_ = json.load(open(os.path.join(IN, f"{args.round}.json")))
    base_name = spec_.pop("_base", "AS_r06_1")
    base = json.load(open(os.path.join(AS, "configs", base_name + ".json")))
    names = {}
    for v, ov in spec_.items():
        name = f"IN_{args.round}_{v}"
        json.dump({**base, **ov}, open(os.path.join(AS, "configs", name + ".json"), "w"))
        names[v] = (name, ov)
    tasks = [(n, s) for s in seeds for n, _ in names.values() if not done(n, s)]
    tasks += [(base_name, s) for s in seeds if not done(base_name, s)]
    t0 = time.time()
    print(f"[{args.round}] {len(tasks)} 个任务", flush=True)
    run_queue(tasks, args.gb)
    for attempt in range(3):                                  # 显存不足（同卡其他用户的任务变大）而失败：换卡重跑
        oom = [(n, s) for n, s in tasks if not done(n, s) and
               "out of memory" in open(os.path.join(IN, "logs", f"{n}_s{s}.log"), errors="ignore").read()]
        if not oom:
            break
        print(f"  显存不足重跑（第 {attempt + 1} 次）：{oom}", flush=True)
        run_queue(oom, args.gb + 2)
    # ---------- 汇总：与基准同种子配对
    s = pd.read_parquet(C.SAMPLES, columns=["paper_id", "split", "y3", "topic", "eval_c2000"])
    st = s[s.split == "val"].set_index("paper_id")
    B = {sd: val_metrics(base_name, sd, st) for sd in seeds}
    rows = []
    for v, (name, ov) in [("（基准）", (base_name, {}))] + list(names.items()):
        R = {sd: val_metrics(name, sd, st) for sd in seeds}
        ok = [sd for sd in seeds if R[sd] and B[sd]]
        if not ok:
            rows.append({"变体": v, "改动": json.dumps(ov, ensure_ascii=False), "成功种子": 0})
            continue
        dm = np.array([R[sd]["MALE"] - B[sd]["MALE"] for sd in ok])
        dr = np.array([R[sd]["rho"] - B[sd]["rho"] for sd in ok])
        rows.append({"变体": v, "改动": json.dumps(ov, ensure_ascii=False), "成功种子": len(ok),
                     "验证 MALE": np.mean([R[sd]["MALE"] for sd in ok]), "各种子": " / ".join(f"{R[sd]['MALE']:.4f}" for sd in ok),
                     "整体偏差": np.mean([R[sd]["bias"] for sd in ok]),
                     "去偏 MALE": np.mean([R[sd]["MALE_db"] for sd in ok]),
                     "Δ去偏 MALE（配对）": np.mean([R[sd]["MALE_db"] - B[sd]["MALE_db"] for sd in ok]),
                     "ΔMALE（配对）": dm.mean(), "ΔMALE 标准误": dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan,
                     "子课题内 Spearman": np.mean([R[sd]["rho"] for sd in ok]), "ΔSpearman（配对）": dr.mean(),
                     "NDCG@10": np.mean([R[sd]["ndcg"] for sd in ok]), "最佳轮": "/".join(str(R[sd]["ep"]) for sd in ok),
                     "显存 GB": max(R[sd]["mem"] or 0 for sd in ok), "秒": int(np.mean([R[sd]["sec"] for sd in ok]))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(IN, f"{args.round}_summary.csv"), index=False)
    md = [f"\n## {args.round}（{time.strftime('%m-%d %H:%M')}，基准 {base_name}，种子 {seeds}，用时 {time.time() - t0:.0f} 秒）\n",
          res.round(4).to_markdown(index=False), "\n"]
    open(os.path.join(IN, "rounds.md"), "a", encoding="utf-8").write("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
