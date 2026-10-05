"""
【05】实验 2 · 自动迭代搜索（只看验证集；测试集结果虽然随每次训练一起写出，但搜索与选择从不读取）

用途
  每一轮：
    1. 按当前空闲 GPU 数决定本轮任务数（每张卡一个任务，显存余量 ≥ 13 GB 才用）；
    2. 生成候选配置：当前最好的几个配置各"变异"1–2 个可调项，外加少量随机配置；每个配置跑 2 个种子；
    3. 所有任务同时在各 GPU 上训练（调用 02_mpcnet.py）；
    4. 读取每个任务的验证集 MALE，按 2 个种子平均排名，写本轮总结：本轮结果、当前前五名、各可调项每个取值的平均表现
       （用来判断"哪个方向在起作用"）。
  状态保存在 autosearch/state.json，中断后再运行会接着上一轮继续。

输入
  02_mpcnet.py、v6 张量（参考文献质量特征）
输出（autosearch/）
  configs/AS_r<轮>_<序号>.json、state.json、rounds.md（每轮追加总结）

用法
  python 05_autosearch.py --rounds 5 [--space space.json]
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AS = os.path.join(HERE, "autosearch")
PY = "/path/to/conda/envs/mpcc/bin/python"
FIXED = {"direct_log": True, "t6": True}
SPACE = {
    "logshare": [0.3, 1.0, 3.0], "aux_w": [0.0, 0.3, 1.0], "n_self": [0, 1, 2], "heads": [4, 8], "d": [256, 384, 512],
    "dropout": [0.1, 0.2, 0.3], "lr": [5e-4, 1e-3, 2e-3], "wd": [1e-4, 1e-3, 1e-2], "bs": [1500, 3000, 6000],
    "patience": [5, 8], "ema": [0.0, 0.99, 0.995], "rdrop": [0.0, 0.1, 0.3], "loss_fn": ["mse", "huber"],
}
BASE = {"logshare": 1.0, "aux_w": 0.3, "n_self": 1, "heads": 4, "d": 256, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4,
        "bs": 3000, "patience": 5, "ema": 0.0, "rdrop": 0.0, "loss_fn": "mse", "tdir": "_v6", "min_ep": 0}   # = V6_aux_self


def free_gpus(gb_per_task=7.0, max_per_gpu=4):
    """按显存余量往每张卡上装多个任务（单个任务峰值约 4.6 GB，按 6 GB 计）；返回可重复的 GPU 编号列表。"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    g = []
    for line in out:
        i, tot, used = [int(x) for x in line.split(",")]
        g += [i] * min(max_per_gpu, int(((tot - used) / 1024 - 1) // gb_per_task))
    return g


def fill(cfg):
    """旧试验没有后来新增的可调项（如 tdir）：按起点 BASE 的取值补齐"""
    return {**BASE, **cfg}


def key(cfg):
    cfg = fill(cfg)
    return json.dumps({k: cfg[k] for k in sorted(SPACE)}, sort_keys=True)


def load_state():
    p = os.path.join(AS, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {"round": 0, "trials": []}


def save_state(st):
    json.dump(st, open(os.path.join(AS, "state.json"), "w"), indent=1)


def ranked(st):
    rows = [t for t in st["trials"] if len(t["val"]) >= 2]   # 至少 2 个种子成功才参与排名
    return sorted(rows, key=lambda t: np.mean(t["val"]))


def propose(st, n, rng):
    seen = {key(t["cfg"]) for t in st["trials"]}
    out = []
    if not st["trials"]:
        out.append(dict(BASE))
    top = [fill(t["cfg"]) for t in ranked(st)[:3]] or [BASE]
    # 定向试验：当前最好的配置，把"张量版本"等离散的数据 / 结构选项逐一换成其他取值（变异很少碰到它们）
    for k in ("tdir",):
        if k in SPACE:
            for v in SPACE[k]:
                c = dict(top[0])
                c[k] = v
                if len(out) < n and key(c) not in seen and key(c) not in {key(x) for x in out}:
                    out.append(c)
    tries = 0
    while len(out) < n and tries < 2000:
        tries += 1
        if rng.random() < 0.8:                       # 变异：从前三名之一出发，改 1–2 个可调项
            c = dict(top[rng.randrange(len(top))])
            mutable = [k for k in SPACE if any(v != c.get(k) for v in SPACE[k])]   # 只有一个取值的可调项不能变异
            for k in rng.sample(mutable, min(len(mutable), rng.choice([1, 1, 2]))):
                c[k] = rng.choice([v for v in SPACE[k] if v != c.get(k)])
            for k in SPACE:                              # 起点的取值若不在当前搜索空间里，换成空间内的值
                if c.get(k) not in SPACE[k]:
                    c[k] = rng.choice(SPACE[k])
        else:                                        # 随机探索
            c = {**BASE, **{k: rng.choice(v) for k, v in SPACE.items()}}
        if key(c) not in seen and key(c) not in {key(x) for x in out}:
            out.append(c)
    return out


def run_round(st, rng, seeds=(1, 2, 3)):
    gpus = free_gpus()
    n_cfg = max(1, len(gpus) // len(seeds))
    st["round"] += 1
    r = st["round"]
    cands = propose(st, n_cfg, rng)
    tasks = []
    for i, c in enumerate(cands):
        name = f"AS_r{r:02d}_{i}"
        json.dump({**FIXED, **c}, open(os.path.join(AS, "configs", name + ".json"), "w"))
        for s in seeds:
            tasks.append((name, s, c))
    t0 = time.time()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    def launch(name, s, g, mode="w"):
        log = open(os.path.join(AS, "logs", f"{name}_s{s}.log"), mode)
        return subprocess.Popen([PY, "-u", os.path.join(HERE, "02_mpcnet.py"), "--configs", name, "--seeds", str(s),
                                 "--gpus", str(g)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)

    procs = [(name, s, launch(name, s, g)) for (name, s, c), g in zip(tasks, itertools.cycle(gpus))]
    for _, _, p in procs:
        p.wait()
    # 显存不足（同卡上其他用户的任务突然变大）而失败的任务：换到当前空闲显存最多的卡上重跑，最多 3 次
    for attempt in range(3):
        oom = [(n, s) for n, s, _ in procs if not os.path.exists(os.path.join(HERE, "runs", f"{n}_s{s}", "summary.json"))
               and "out of memory" in open(os.path.join(AS, "logs", f"{n}_s{s}.log"), errors="ignore").read()]
        if not oom:
            break
        fg = free_gpus()
        gs = sorted(set(fg), key=lambda i: -fg.count(i)) or [0]
        print(f"  显存不足重跑（第 {attempt + 1} 次）：{len(oom)} 个任务 → GPU {gs}", flush=True)
        procs = [(n, s, launch(n, s, g, "a")) for (n, s), g in zip(oom, itertools.cycle(gs))]
        for _, _, p in procs:
            p.wait()
    rows = []
    for i, c in enumerate(cands):
        name = f"AS_r{r:02d}_{i}"
        val, ep, mem = [], [], []
        for s in seeds:
            f = os.path.join(HERE, "runs", f"{name}_s{s}", "summary.json")
            if os.path.exists(f):
                d = json.load(open(f))
                val.append(d["best_val_MALE"])
                ep.append(d["best_epoch"])
                mem.append(d.get("peak_mem_gb"))
        st["trials"].append({"name": name, "round": r, "cfg": c, "val": val, "epochs": ep})
        c = fill(c)
        rows.append({"配置": name, "改动（相对 v6 起点）": ", ".join(f"{k}={c[k]}" for k in SPACE if c[k] != BASE[k]) or "（起点）",
                     "验证 MALE（各种子）": " / ".join(f"{v:.4f}" for v in val) or "失败", "平均": np.mean(val) if val else np.nan,
                     "最佳轮": "/".join(map(str, ep)), "显存 GB": max([m for m in mem if m] or [np.nan])})
    save_state(st)
    # ---------- 本轮总结
    rk = ranked(st)
    top5 = pd.DataFrame([{"配置": t["name"], "平均验证 MALE": np.mean(t["val"]),
                          "改动": ", ".join(f"{k}={fill(t['cfg'])[k]}" for k in SPACE if fill(t["cfg"])[k] != BASE[k]) or "（起点）"} for t in rk[:5]])
    allt = pd.DataFrame([{**fill(t["cfg"]), "m": np.mean(t["val"])} for t in rk])
    marg = []
    for k, vals in SPACE.items():
        for v in vals:
            sub = allt[allt[k] == v].m if len(allt) else pd.Series(dtype=float)
            if len(sub):
                marg.append({"可调项": k, "取值": str(v), "试验数": len(sub), "平均": sub.mean(), "最好": sub.min()})
    marg = pd.DataFrame(marg)
    best = rk[0] if rk else None
    md = [f"\n## 第 {r} 轮（{time.strftime('%H:%M')}，{len(tasks)} 个任务 / {len(gpus)} 张 GPU，用时 {time.time() - t0:.0f} 秒）\n",
          pd.DataFrame(rows).round(4).to_markdown(index=False), "\n",
          f"**当前最好**：{best['name']}，平均验证 MALE {np.mean(best['val']):.4f}（共 {len(rk)} 个有效配置）\n" if best else "",
          "当前前五：\n", top5.round(4).to_markdown(index=False), "\n",
          "各可调项取值的表现（全部试验）：\n", marg.round(4).to_markdown(index=False) if len(marg) else "", "\n"]
    open(os.path.join(AS, "rounds.md"), "a", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:5]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--space", default=None, help="可选：json 文件，覆盖搜索空间（反思后缩小 / 扩展）")
    args = ap.parse_args()
    for d in ("configs", "logs"):
        os.makedirs(os.path.join(AS, d), exist_ok=True)
    global SPACE
    if args.space:
        SPACE = json.load(open(args.space))
    st = load_state()
    rng = random.Random(1000 + st["round"])
    for _ in range(args.rounds):
        run_round(st, rng)
    print("AUTOSEARCH BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
