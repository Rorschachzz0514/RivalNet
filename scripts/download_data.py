"""Download the RivalBench data files into data/ and verify their checksums.

Usage
    python scripts/download_data.py            # all files (about 2 GB)
    python scripts/download_data.py --no-emb   # without the SPECTER2 embeddings (about 270 MB)

The files listed in data/MD5SUMS are fetched from BASE_URL and placed under data/
(data/<field>/..., data/embeddings/..., data/predictions/...).
"""
import argparse
import hashlib
import os
import sys
import urllib.request

# Set when the data are hosted (an anonymized link during review, a Zenodo DOI after publication).
BASE_URL = ""

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def entries(with_emb):
    out = []
    for line in open(os.path.join(DATA, "MD5SUMS"), encoding="utf-8"):
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip().lstrip("./")
        top = rel.split("/")[0]
        if top not in ("data", "embeddings", "predictions") or (top == "embeddings" and not with_emb):
            continue
        local = rel[len("data/"):] if top == "data" else rel          # data/ai/x -> ai/x; embeddings/x stays
        out.append((digest, rel, os.path.join(DATA, local)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emb", action="store_true", help="skip the SPECTER2 embeddings")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()
    if not args.base_url:
        sys.exit("The data location is not set yet: pass --base-url or see data/README.md.")
    bad = 0
    for digest, rel, local in entries(not args.no_emb):
        if os.path.exists(local) and md5(local) == digest:
            print(f"ok       {rel}")
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        print(f"download {rel}", flush=True)
        urllib.request.urlretrieve(f"{args.base_url.rstrip('/')}/{rel}", local)
        if md5(local) != digest:
            print(f"CHECKSUM MISMATCH {rel}")
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
