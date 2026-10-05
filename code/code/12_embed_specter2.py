"""
【12】为总库 detail 的全部论文计算 SPECTER2 向量（多 GPU，边扫边算）               阶段 1：总库

用途
  论文的语义向量，用于找内容相似的对手（S2）、实验 2 的论文编码器、R1 向量聚类。

模型与输入文本
  allenai/specter2_base + proximity 适配器 allenai/specter2（经 hf-mirror.com 下载，缓存在 /path/to/mpcc/models）
  文本 = 标题 + [SEP] + 摘要；无摘要只用标题（has_abstract=False，日后可用 Semantic Scholar 摘要重算）；
  标题为空但有摘要只用摘要；两者都没有的不计算

输入
  union/detail/*.parquet（只读 id、title、abstract）

输出（/path/to/mpcc/union/specter2/）
  {与 detail 同名的分片}.parquet   列 id、has_abstract、emb（768 维 float16）；每个 detail 分片对应一个文件
  _stats/{分片}.json              计算时当场检查的统计：行数、NaN/Inf 个数、全零向量个数、向量长度范围

机制
  每张 GPU 一个进程，按分片名哈希分工；只处理写完超过 60 秒的 detail 分片；先写临时文件再改名，可断点续跑；
  显存不足自动减半批大小；扫描结束且所有分片算完后自动退出

用法
  python 12_embed_specter2.py --gpus 0,1,2,3,4,5,6,7

运行记录
  2026-10-02  8 张 GPU，单卡约 730 篇/秒，277 个分片全部完成，总计 2,600 余万个向量
  发现的问题：标题为空字符串但有摘要的论文被漏算 → 改规则后重启；早期分片的遗漏由【13】补算
  原文件名：03_embed_specter2.py
"""
import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

DETAIL = os.path.join(C.UNION_DIR, "detail")
OUT = os.path.join(C.UNION_DIR, "specter2")
SCAN_LOG = os.path.join(C.LOG_DIR, "01_scan.log")
MODEL_HOME = os.path.join(C.ROOT, "models")


def scan_finished():
    try:
        with open(SCAN_LOG) as f:
            return any(line.startswith("DONE") for line in f)
    except FileNotFoundError:
        return False


def my_pending(k, n):
    """本进程负责且尚未计算、且已写完的分片."""
    out = []
    now = time.time()
    for name in sorted(os.listdir(DETAIL)):
        if not name.endswith(".parquet") or zlib.crc32(name.encode()) % n != k:
            continue
        if os.path.exists(os.path.join(OUT, name)):
            continue
        if now - os.path.getmtime(os.path.join(DETAIL, name)) < 60:
            continue
        out.append(name)
    return out


def run(gpu, k, n, batch):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = MODEL_HOME
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.set_active_adapters("specter2")
    model = model.cuda().eval().half()
    sep = tok.sep_token
    schema = pa.schema([("id", pa.int64()), ("has_abstract", pa.bool_()), ("emb", pa.list_(pa.float16(), 768))])

    def encode(texts):
        x = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            return model(**x).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16)

    n_done, t0 = 0, time.time()
    while True:
        todo = my_pending(k, n)
        if not todo:
            if scan_finished() and not my_pending(k, n):
                # 扫描已结束: 再等 90 秒确认没有刚写完的分片, 然后退出
                time.sleep(90)
                if not my_pending(k, n):
                    break
            time.sleep(30)
            continue
        for name in todo:
            try:
                t = pq.read_table(os.path.join(DETAIL, name), columns=["id", "title", "abstract"]).to_pydict()
            except Exception as e:                       # 文件可能还没写完, 下一轮再试
                print(f"[gpu{gpu}] skip {name}: {e}", flush=True)
                continue
            # 标题或摘要有一个非空就计算 (少数论文标题是空字符串但有摘要)
            rows = [(i, ti, ab) for i, ti, ab in zip(t["id"], t["title"], t["abstract"]) if ti or ab]
            texts = [(ti or "") + (sep + ab if ab else "") for _, ti, ab in rows]
            order = np.argsort([len(s) for s in texts])          # 按长度排序, 减少 padding
            embs = np.zeros((len(rows), 768), dtype=np.float16)
            i, b = 0, batch
            while i < len(order):
                idx = order[i:i + b]
                try:
                    embs[idx] = encode([texts[j] for j in idx])
                    i += len(idx)
                except torch.cuda.OutOfMemoryError:            # 卡被别人占用时自动减半批大小
                    torch.cuda.empty_cache()
                    b = max(8, b // 2)
                    print(f"[gpu{gpu}] OOM, batch -> {b}", flush=True)
            tbl = pa.Table.from_arrays([
                pa.array([r[0] for r in rows], pa.int64()),
                pa.array([bool(r[2]) for r in rows]),
                pa.FixedSizeListArray.from_arrays(pa.array(embs.reshape(-1), pa.float16()), 768),
            ], schema=schema)
            # 当场自查 (向量还在内存里, 几乎不花时间): 结果写入 _stats/, 供 check_progress.py 读取
            e32 = embs.astype(np.float32)
            norms = np.linalg.norm(e32, axis=1)
            stat = {"shard": name, "n": len(rows), "n_detail": len(t["id"]),
                    "n_nonfinite": int((~np.isfinite(e32)).any(axis=1).sum()),
                    "n_zero": int((norms < 1e-3).sum()),
                    "norm_min": float(norms.min()) if len(rows) else None,
                    "norm_max": float(norms.max()) if len(rows) else None,
                    "n_has_abstract": int(sum(bool(r[2]) for r in rows))}
            if stat["n_nonfinite"] or stat["n_zero"]:
                print(f"[gpu{gpu}] BADVEC {name}: {stat}", flush=True)
            with open(os.path.join(OUT, "_stats", name + ".json"), "w") as fh:
                json.dump(stat, fh)
            tmp = os.path.join(OUT, f".{name}.tmp")
            pq.write_table(tbl, tmp, compression="zstd")
            os.replace(tmp, os.path.join(OUT, name))
            n_done += len(rows)
            print(f"[gpu{gpu}] {name}: {len(rows):,} papers  total={n_done:,}  "
                  f"{n_done / (time.time() - t0):.0f} papers/s", flush=True)
    print(f"[gpu{gpu}] FINISHED {n_done:,} papers in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "_stats"), exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=run, args=(g, k, len(gpus), args.batch)) for k, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
