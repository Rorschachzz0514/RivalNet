"""
【11】把 OpenAlex 小型实体文件转成 parquet，放进总库 union/meta                    阶段 1：总库

用途
  期刊/会议（含逐年发文与被引，用于计算"截至 T−1 年"的期刊影响力）与方向层级的元数据。

输入
  /data/openalex/jsonl/{sources,topics,subfields,fields,domains}.txt

输出（/path/to/mpcc/union/meta/）
  sources.parquet        期刊/会议 260,811 个：名称、类型、出版商、国家、是否 OA/DOAJ/核心；
                         h_index_2024、mean_citedness_2yr_2024 为 2024 年底的值，含未来信息，只做描述
  sources_year.parquet   期刊/会议逐年发文量与被引量 2,028,880 行
  topics.parquet         方向 4,516 个：名称、简介、关键词、所属子领域/领域/大类
  subfields.parquet / fields.parquet / domains.parquet   层级名称

用法
  python 11_scan_meta.py      （单进程，几分钟）

运行记录
  2026-10-02  首次生成；后随失败的第一次扫描目录一起被移走并误删，同日重新生成，行数与首次完全一致
  原文件名：01b_scan_meta.py
"""
import os
import sys

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def oid(url):
    if isinstance(url, int):             # 快照里部分区段的 ID 直接存成整数
        return url
    try:
        return int(url.rsplit("/", 1)[1][1:])
    except (AttributeError, ValueError, IndexError):
        return None


def tail_int(url):
    if isinstance(url, int):
        return url
    try:
        return int(url.rsplit("/", 1)[1])
    except (AttributeError, ValueError, IndexError):
        return None


def lines(name):
    with open(os.path.join(C.OA_DIR, name), "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def write(rows, name, out):
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(out, name), compression="zstd")
    print(f"{name}: {len(rows):,} rows", flush=True)


def main():
    out = os.path.join(C.UNION_DIR, "meta")
    os.makedirs(out, exist_ok=True)

    src, src_year = [], []
    for o in lines("sources.txt"):
        sid = oid(o.get("id"))
        ss = o.get("summary_stats") or {}
        src.append({
            "source_id": sid, "name": o.get("display_name"), "type": o.get("type"), "issn_l": o.get("issn_l"),
            "publisher": o.get("host_organization_name"), "country": o.get("country_code"),
            "is_oa": o.get("is_oa"), "is_in_doaj": o.get("is_in_doaj"), "is_core": o.get("is_core"),
            "works_count": o.get("works_count"), "cited_by_count": o.get("cited_by_count"),
            "h_index_2024": ss.get("h_index"), "mean_citedness_2yr_2024": ss.get("2yr_mean_citedness"),
        })
        for c in o.get("counts_by_year") or []:
            src_year.append({"source_id": sid, "year": c.get("year"), "works_count": c.get("works_count"),
                             "oa_works_count": c.get("oa_works_count"), "cited_by_count": c.get("cited_by_count")})
    write(src, "sources.parquet", out)
    write(src_year, "sources_year.parquet", out)

    tps = []
    for o in lines("topics.txt"):
        tps.append({
            "topic_id": oid(o.get("id")), "name": o.get("display_name"), "description": o.get("description"),
            "keywords": o.get("keywords"),
            "subfield_id": tail_int((o.get("subfield") or {}).get("id")), "subfield": (o.get("subfield") or {}).get("display_name"),
            "field_id": tail_int((o.get("field") or {}).get("id")), "field": (o.get("field") or {}).get("display_name"),
            "domain_id": tail_int((o.get("domain") or {}).get("id")), "domain": (o.get("domain") or {}).get("display_name"),
            "works_count_2024": o.get("works_count"), "cited_by_count_2024": o.get("cited_by_count"),
        })
    write(tps, "topics.parquet", out)

    for name in ("subfields", "fields", "domains"):
        rows = [{"id": tail_int(o.get("id")), "name": o.get("display_name"),
                 "parent_id": tail_int(((o.get("field") or o.get("domain")) or {}).get("id"))}
                for o in lines(f"{name}.txt")]
        write(rows, f"{name}.parquet", out)


if __name__ == "__main__":
    main()
