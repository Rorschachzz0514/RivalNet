"""Download the RivalBench data into data/ and verify checksums.

Usage
    python scripts/download_data.py            # data (about 240 MB) and SPECTER2 embeddings (about 1.6 GB)
    python scripts/download_data.py --no-emb   # data only

The data are distributed as two archives that unpack into data/ (data/<field>/..., data/predictions/...,
data/embeddings/...). Every unpacked file is checked against data/MD5SUMS.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

# Download locations (an anonymized link during review; a Zenodo DOI after publication).
ARCHIVES = {
    "rivalbench_data.zip": {"url": "https://osf.io/download/6ac3d8801e6b19c085496c55/?view_only=040742fe90034c9097abe380d4ebeedb", "md5": "ca21af3e40b1d79af122a5255aa4e8b5"},
    "rivalbench_embeddings.zip": {"url": "https://osf.io/download/6ac3dabbae478e70c7a2f986/?view_only=040742fe90034c9097abe380d4ebeedb", "md5": "5a8be3fad85736d0fe3ca53d4f445b4b"},
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, url, digest):
    path = os.path.join(DATA, name)
    if os.path.exists(path) and md5(path) == digest:
        print(f"ok        {name}")
        return path
    if not url:
        sys.exit(f"No download location set for {name}: pass --data-url / --emb-url, or place the file in data/.")
    print(f"download  {name}", flush=True)
    urllib.request.urlretrieve(url, path)
    if md5(path) != digest:
        sys.exit(f"Checksum mismatch for {name}; please download it again.")
    return path


def verify(with_emb):
    bad = 0
    for line in open(os.path.join(DATA, "MD5SUMS"), encoding="utf-8"):
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip().lstrip("./")
        top = rel.split("/")[0]
        if top == "data":
            local = os.path.join(DATA, rel[len("data/"):])
        elif top in ("predictions", "embeddings"):
            if top == "embeddings" and not with_emb:
                continue
            local = os.path.join(DATA, rel)
        else:
            continue
        if not os.path.exists(local) or md5(local) != digest:
            print(f"MISSING OR CORRUPT  {rel}")
            bad += 1
    print("all files verified" if not bad else f"{bad} files failed verification")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emb", action="store_true", help="skip the SPECTER2 embeddings")
    ap.add_argument("--data-url", default=ARCHIVES["rivalbench_data.zip"]["url"])
    ap.add_argument("--emb-url", default=ARCHIVES["rivalbench_embeddings.zip"]["url"])
    args = ap.parse_args()
    todo = [("rivalbench_data.zip", args.data_url)] + ([] if args.no_emb else [("rivalbench_embeddings.zip", args.emb_url)])
    for name, url in todo:
        path = fetch(name, url, ARCHIVES[name]["md5"])
        with zipfile.ZipFile(path) as z:            # archive paths start with data/
            keep = ("data/README.md", "data/DATASHEET.md", "data/MD5SUMS")   # the repository's own copies take precedence
            z.extractall(ROOT, members=[m for m in z.namelist() if m not in keep])
        print(f"unpacked  {name}")
    sys.exit(1 if verify(not args.no_emb) else 0)


if __name__ == "__main__":
    main()
