"""
【10】扫描 OpenAlex 论文全量快照，建立总库 union/detail 与 union/lite            阶段 1：总库

用途
  并行扫描 OpenAlex works.txt（2.83 TB，一行一篇论文的 JSON），只读一遍，把所有实验需要的字段一次取全，
  以后各实验只从总库切子集，不再重扫原始快照。

输入
  /data/openalex/jsonl/works.txt          OpenAlex 快照 RELEASE 2024-12-31（组内共享，只读）

输出（/path/to/mpcc/union/）
  detail/w{进程}_{分片}.parquet   6 个领域（CS 17、医学 27、数学 26、化学 16、物理 31、环境 23）、2016–2024 年
                                 发表的论文，不限类型/语言，66 列完整字段（每 10 万篇一个分片）
  lite/w{进程}_{分片}.parquet     全库每篇论文 17 列精简字段（方向、年份、逐年被引、作者；2015 年起另存参考文献列表）
  _worker_{进程}.json            每个进程处理的字节区间、行数、错误行数
  _summary.json                  全部汇总

用法
  python 10_scan_works.py                      全量（32 进程）
  python 10_scan_works.py --test 1000000000    测试：每个进程只读其分段前 1 GB，输出到 union_test/

运行记录
  2026-10-02  全量，32 进程，5.3 小时：262,630,159 篇（= 文件总行数），错误行 0；detail 26,329,455 篇（AI 1,502,914）
  发现的问题：快照部分区段的 subfield/field/domain ID 存成整数而非 URL → 解析函数兼容两种格式后重跑；
             第一版漏了物理/环境两个领域、2015 年前论文的作者、concepts/mesh/机构类型 → 补齐后重跑
  原文件名：01_scan_openalex.py
"""
import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


# ============================================================
# 小工具
# ============================================================
def oid(url):
    """'https://openalex.org/W123' / '.../A5' / '.../T10432' -> 123 / 5 / 10432; 其他格式返回 None.
    快照里部分区段的 ID 直接存成整数, 原样返回."""
    if url is None or url == "":
        return None
    if isinstance(url, int):
        return url
    try:
        return int(url.rsplit("/", 1)[1][1:])
    except (ValueError, IndexError):
        return None


def tail_int(url):
    """'https://openalex.org/subfields/1702' -> 1702; 快照里部分区段直接存整数 1702."""
    if url is None or url == "":
        return None
    if isinstance(url, int):
        return url
    try:
        return int(url.rsplit("/", 1)[1])
    except (ValueError, IndexError):
        return None


def abstract_text(inv):
    """把 abstract_inverted_index 还原成文本."""
    if not inv:
        return None
    pos = []
    for w, ps in inv.items():
        for p in ps:
            pos.append((p, w))
    pos.sort()
    return " ".join(w for _, w in pos)


ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z\-\.]+/\d{7}|\d{4}\.\d{4,5})")


def arxiv_id(o):
    """从 doi (10.48550/arxiv.XXXX) 或任一 location 的链接里提取 arXiv 编号 (去掉版本号)."""
    doi = (o.get("doi") or "").lower()
    if "10.48550/arxiv." in doi:
        return re.sub(r"v\d+$", "", doi.split("10.48550/arxiv.", 1)[1])
    for loc in o.get("locations") or []:
        for k in ("landing_page_url", "pdf_url"):
            m = ARXIV_RE.search((loc.get(k) or "").lower())
            if m:
                return m.group(1)
    return None


# ============================================================
# 两类输出的 schema
# ============================================================
L64, L32, LF, LS, L16 = pa.list_(pa.int64()), pa.list_(pa.int32()), pa.list_(pa.float32()), pa.list_(pa.string()), pa.list_(pa.int16())
LITE_SCHEMA = pa.schema([
    ("id", pa.int64()), ("year", pa.int16()), ("type", pa.string()),
    ("pt_topic", pa.int32()), ("pt_subfield", pa.int16()), ("pt_field", pa.int16()),
    ("topic_ids", L32), ("topic_scores", LF),
    ("n_refs", pa.int32()), ("refs", L64), ("author_ids", L64), ("source_id", pa.int64()),
    ("cby_year", L16), ("cby_count", L32), ("cited_by_count", pa.int32()),
    ("is_retracted", pa.bool_()), ("is_paratext", pa.bool_()),
])
DETAIL_SCHEMA = pa.schema([
    ("id", pa.int64()), ("doi", pa.string()), ("arxiv_id", pa.string()), ("pmid", pa.string()), ("mag", pa.string()),
    ("title", pa.string()), ("abstract", pa.string()), ("language", pa.string()),
    ("type", pa.string()), ("type_crossref", pa.string()),
    ("publication_date", pa.string()), ("publication_year", pa.int16()),
    ("created_date", pa.string()), ("updated_date", pa.string()),
    ("pt_topic", pa.int32()), ("pt_subfield", pa.int16()), ("pt_field", pa.int16()), ("pt_domain", pa.int16()), ("pt_score", pa.float32()),
    ("topic_ids", L32), ("topic_scores", LF), ("topic_subfields", L16), ("topic_fields", L16),
    ("keywords", LS), ("keyword_scores", LF),
    ("concept_ids", L64), ("concept_names", LS), ("concept_levels", L16), ("concept_scores", LF),
    ("mesh_names", LS), ("mesh_major", pa.list_(pa.bool_())),
    ("author_ids", L64), ("n_authors", pa.int32()),
    ("institution_ids", L64), ("institution_types", LS), ("n_institutions", pa.int32()),
    ("countries", LS), ("n_countries", pa.int32()),
    ("n_refs", pa.int32()), ("refs", L64),
    ("cited_by_count", pa.int32()), ("cby_year", L16), ("cby_count", L32),
    ("is_oa", pa.bool_()), ("oa_status", pa.string()),
    ("source_id", pa.int64()), ("source_name", pa.string()), ("source_type", pa.string()),
    ("primary_version", pa.string()), ("primary_is_oa", pa.bool_()),
    ("loc_source_ids", L64), ("loc_source_types", LS), ("loc_versions", LS), ("loc_is_oa", pa.list_(pa.bool_())),
    ("locations_count", pa.int32()), ("indexed_in", LS),
    ("n_grants", pa.int32()), ("funder_ids", L64),
    ("first_page", pa.string()), ("last_page", pa.string()),
    ("fwci", pa.float32()), ("cnp_value", pa.float32()), ("cnp_top1", pa.bool_()), ("cnp_top10", pa.bool_()),
    ("is_retracted", pa.bool_()), ("is_paratext", pa.bool_()),
])


# ============================================================
# 单条记录 -> 行
# ============================================================
def parse(o):
    wid = oid(o.get("id"))
    year = o.get("publication_year")
    pt = o.get("primary_topic") or {}
    pt_topic = oid(pt.get("id"))
    pt_sub = tail_int((pt.get("subfield") or {}).get("id"))
    pt_field = tail_int((pt.get("field") or {}).get("id"))
    topics = o.get("topics") or []
    t_ids = [oid(t.get("id")) for t in topics]
    t_sc = [t.get("score") for t in topics]
    cby = o.get("counts_by_year") or []
    keep_lists = year is not None and year >= C.REFS_YEAR_MIN
    refs = [oid(r) for r in o.get("referenced_works") or []] if keep_lists else None
    auths = o.get("authorships") or []
    a_ids = [oid((a.get("author") or {}).get("id")) for a in auths]
    pl = o.get("primary_location") or {}
    ps = pl.get("source") or {}
    allrow = {
        "id": wid, "year": year, "type": o.get("type"),
        "pt_topic": pt_topic, "pt_subfield": pt_sub, "pt_field": pt_field,
        "topic_ids": t_ids, "topic_scores": t_sc,
        "n_refs": o.get("referenced_works_count"),
        "refs": refs, "author_ids": a_ids, "source_id": oid(ps.get("id")),
        "cby_year": [c.get("year") for c in cby], "cby_count": [c.get("cited_by_count") for c in cby],
        "cited_by_count": o.get("cited_by_count"),
        "is_retracted": o.get("is_retracted"), "is_paratext": o.get("is_paratext"),
    }
    focal = None
    if pt_field in C.DETAIL_FIELDS and year is not None and C.DETAIL_YEAR_MIN <= year <= C.DETAIL_YEAR_MAX:
        ids = o.get("ids") or {}
        inst, inst_type = [], []
        for a in auths:
            for i in a.get("institutions") or []:
                v = oid(i.get("id"))
                if v is not None and v not in inst:
                    inst.append(v)
                    inst_type.append(i.get("type"))
        cons = o.get("concepts") or []
        mesh = o.get("mesh") or []
        locs = o.get("locations") or []
        oa = o.get("open_access") or {}
        cnp = o.get("citation_normalized_percentile") or {}
        bib = o.get("biblio") or {}
        grants = o.get("grants") or []
        kws = o.get("keywords") or []
        abst = abstract_text(o.get("abstract_inverted_index"))
        focal = {
            "id": wid, "doi": o.get("doi"), "arxiv_id": arxiv_id(o), "pmid": ids.get("pmid"), "mag": str(ids["mag"]) if ids.get("mag") else None,
            "title": o.get("title"), "abstract": abst, "language": o.get("language"),
            "type": o.get("type"), "type_crossref": o.get("type_crossref"),
            "publication_date": o.get("publication_date"), "publication_year": year,
            "created_date": o.get("created_date"), "updated_date": o.get("updated_date"),
            "pt_topic": pt_topic, "pt_subfield": pt_sub, "pt_field": pt_field,
            "pt_domain": tail_int((pt.get("domain") or {}).get("id")), "pt_score": pt.get("score"),
            "topic_ids": t_ids, "topic_scores": t_sc,
            "topic_subfields": [tail_int((t.get("subfield") or {}).get("id")) for t in topics],
            "topic_fields": [tail_int((t.get("field") or {}).get("id")) for t in topics],
            "keywords": [k.get("display_name") for k in kws], "keyword_scores": [k.get("score") for k in kws],
            "concept_ids": [oid(c.get("id")) for c in cons], "concept_names": [c.get("display_name") for c in cons],
            "concept_levels": [c.get("level") for c in cons], "concept_scores": [c.get("score") for c in cons],
            "mesh_names": [m.get("descriptor_name") for m in mesh], "mesh_major": [m.get("is_major_topic") for m in mesh],
            "author_ids": a_ids, "n_authors": len(auths),
            "institution_ids": inst, "institution_types": inst_type,
            "n_institutions": o.get("institutions_distinct_count"),
            "countries": sorted({c for a in auths for c in (a.get("countries") or [])}),
            "n_countries": o.get("countries_distinct_count"),
            "n_refs": o.get("referenced_works_count"),
            "refs": refs if refs is not None else [oid(r) for r in o.get("referenced_works") or []],
            "cited_by_count": o.get("cited_by_count"),
            "cby_year": allrow["cby_year"], "cby_count": allrow["cby_count"],
            "is_oa": oa.get("is_oa"), "oa_status": oa.get("oa_status"),
            "source_id": oid(ps.get("id")), "source_name": ps.get("display_name"), "source_type": ps.get("type"),
            "primary_version": pl.get("version"), "primary_is_oa": pl.get("is_oa"),
            "loc_source_ids": [oid((l.get("source") or {}).get("id")) for l in locs],
            "loc_source_types": [(l.get("source") or {}).get("type") for l in locs],
            "loc_versions": [l.get("version") for l in locs], "loc_is_oa": [l.get("is_oa") for l in locs],
            "locations_count": o.get("locations_count"), "indexed_in": o.get("indexed_in"),
            "n_grants": len(grants), "funder_ids": [oid(g.get("funder")) for g in grants],
            "first_page": bib.get("first_page"), "last_page": bib.get("last_page"),
            "fwci": o.get("fwci"), "cnp_value": cnp.get("value"),
            "cnp_top1": cnp.get("is_in_top_1_percent"), "cnp_top10": cnp.get("is_in_top_10_percent"),
            "is_retracted": o.get("is_retracted"), "is_paratext": o.get("is_paratext"),
        }
    return allrow, focal


# ============================================================
# worker: 处理一个字节区间
# ============================================================
class Buffer:
    def __init__(self, schema, outdir, tag):
        self.schema, self.outdir, self.tag = schema, outdir, tag
        self.rows, self.part = [], 0
        os.makedirs(outdir, exist_ok=True)

    def add(self, r):
        self.rows.append(r)
        if len(self.rows) >= C.FLUSH_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        t = pa.Table.from_pylist(self.rows, schema=self.schema)
        pq.write_table(t, os.path.join(self.outdir, f"{self.tag}_{self.part:05d}.parquet"), compression="zstd")
        self.part += 1
        self.rows = []


def worker(args):
    i, start, end, scan_dir = args
    allbuf = Buffer(LITE_SCHEMA, os.path.join(scan_dir, "lite"), f"w{i:03d}")
    focbuf = Buffer(DETAIL_SCHEMA, os.path.join(scan_dir, "detail"), f"w{i:03d}")
    n = bad = n_detail = n_ai = 0
    t0 = time.time()
    with open(C.WORKS_TXT, "rb", buffering=16 * 1024 * 1024) as f:
        f.seek(start)
        if start > 0:
            f.readline()                      # 跳过被切断的半行 (它属于上一个 worker)
        pos = f.tell()
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos += len(line)
            line = line.strip()
            if not line:
                continue
            try:
                o = orjson.loads(line)
                a, fo = parse(o)
            except Exception as e:
                bad += 1
                if bad <= 3:
                    print(f"[w{i:03d}] bad line at byte {pos}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            n += 1
            allbuf.add(a)
            if fo is not None:
                focbuf.add(fo)
                n_detail += 1
                n_ai += fo["pt_subfield"] == C.AI_SUBFIELD
            if n % 500_000 == 0:
                done = (pos - start) / max(end - start, 1)
                print(f"[w{i:03d}] {n:,} lines  {done:6.1%}  detail={n_detail:,} ai={n_ai:,} bad={bad}  "
                      f"{(pos - start) / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    allbuf.flush()
    focbuf.flush()
    stat = {"worker": i, "start": start, "end": end, "lines": n, "bad": bad, "detail": n_detail, "ai": n_ai,
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(scan_dir, f"_worker_{i:03d}.json"), "w") as fh:
        json.dump(stat, fh)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="测试模式: 每个 worker 只处理分段开头的这么多字节")
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    args = ap.parse_args()
    scan_dir = C.UNION_DIR + ("_test" if args.test else "")
    if os.path.exists(os.path.join(scan_dir, "lite")) and os.listdir(os.path.join(scan_dir, "lite")):
        sys.exit(f"{scan_dir} 已有输出, 为避免混入旧分片, 请先移走再运行")
    size = os.path.getsize(C.WORKS_TXT)
    step = size // args.workers
    jobs = []
    for i in range(args.workers):
        s = i * step
        e = size if i == args.workers - 1 else (i + 1) * step
        if args.test:
            e = min(e, s + args.test)
        jobs.append((i, s, e, scan_dir))
    print(f"file={size / 1e12:.2f} TB  workers={args.workers}  test={bool(args.test)}  out={scan_dir}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        stats = pool.map(worker, jobs, chunksize=1)
    tot = {k: sum(s[k] for s in stats) for k in ("lines", "bad", "detail", "ai")}
    tot["seconds"] = round(time.time() - t0, 1)
    tot["bytes"] = sum(s["end"] - s["start"] for s in stats)
    with open(os.path.join(scan_dir, "_summary.json"), "w") as fh:
        json.dump({"total": tot, "workers": stats}, fh, indent=1)
    print("DONE", json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()
