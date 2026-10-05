"""
【13】补算 SPECTER2 向量（早期分片按旧规则漏掉的"无标题但有摘要"的论文）          阶段 1：总库

用途
  【12】的早期分片只算了有标题的论文。本脚本对指定分片只计算缺失论文的向量，追加进原文件，并补写 _stats 统计；
  已有向量不变。

输入
  --list 指定的分片清单（由【14】完整检查输出的问题分片生成：/path/to/mpcc/logs/embed_redo.txt）
  union/detail/、union/specter2/

输出
  union/specter2/{分片}.parquet   原文件 + 补算的向量（先写临时文件再改名）
  union/specter2/_stats/{分片}.json

用法
  python 13_embed_patch.py --list /path/to/mpcc/logs/embed_redo_part_00 --gpu 0
  （2026-10-02 实际按 8 份清单在 8 张 GPU 上并行运行）

运行记录
  2026-10-02  153 个分片，补算 12,247 个向量（12,246 篇无标题有摘要 + 1 篇空字符串标题），无多余/错误向量；
              补算后【14】完整检查：277 个分片与论文逐条一致
  原文件名：03b_embed_patch.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

D = os.path.join(C.UNION_DIR, "detail")
E = os.path.join(C.UNION_DIR, "specter2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = os.path.join(C.ROOT, "models")
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

    def encode(texts):
        out = []
        for s in range(0, len(texts), 64):
            x = tok(texts[s:s + 64], padding=True, truncation=True, max_length=512, return_tensors="pt",
                    return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                out.append(model(**x).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out).astype(np.float16) if out else np.zeros((0, 768), np.float16)

    total = 0
    for name in open(args.list).read().split():
        old = pq.read_table(os.path.join(E, name))
        have = set(old.column("id").to_pylist())
        d = pq.read_table(os.path.join(D, name), columns=["id", "title", "abstract"]).to_pydict()
        rows = [(i, t, a) for i, t, a in zip(d["id"], d["title"], d["abstract"]) if (t or a) and i not in have]
        if rows:
            embs = encode([(t or "") + (sep + a if a else "") for _, t, a in rows])
            add = pa.Table.from_arrays([
                pa.array([r[0] for r in rows], pa.int64()),
                pa.array([bool(r[2]) for r in rows]),
                pa.FixedSizeListArray.from_arrays(pa.array(embs.reshape(-1), pa.float16()), 768),
            ], schema=old.schema)
            new = pa.concat_tables([old, add])
        else:
            new = old
        # 统计: 对整个分片 (旧 + 新) 重算
        e32 = np.asarray(new.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                         dtype=np.float32).reshape(-1, 768)
        norms = np.linalg.norm(e32, axis=1)
        stat = {"shard": name, "n": new.num_rows, "n_detail": len(d["id"]),
                "n_nonfinite": int((~np.isfinite(e32)).any(axis=1).sum()), "n_zero": int((norms < 1e-3).sum()),
                "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                "n_has_abstract": int(sum(new.column("has_abstract").to_pylist())), "patched": len(rows)}
        tmp = os.path.join(E, f".{name}.tmp")
        pq.write_table(new, tmp, compression="zstd")
        os.replace(tmp, os.path.join(E, name))
        with open(os.path.join(E, "_stats", name + ".json"), "w") as fh:
            json.dump(stat, fh)
        total += len(rows)
        print(f"{name}: +{len(rows)} -> {new.num_rows:,}", flush=True)
    print(f"PATCH DONE: {total:,} vectors added", flush=True)


if __name__ == "__main__":
    main()
