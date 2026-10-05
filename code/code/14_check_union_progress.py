"""
【14】总库自查：运行中轻量检查 + 扫描结束后完整检查                               阶段 1：总库

用途
  在扫描/向量计算进行中反复检查已写出的分片，尽早发现问题；扫描结束后逐条读内容做完整检查。

检查内容
  扫描进行中（只读 parquet 元数据，不与扫描抢磁盘）：
    文件可读；id 无空值；年份/领域/逐年被引年份/方向分数在范围内；逐年被引非负；
    向量条数在合理范围内；向量当场统计（NaN/全零）
  扫描结束后（--full 或总库 _summary.json 存在时）：
    以上全部 + 参考文献列表完整、各列表字段长度一致、lite 中参考文献列表按年份规则存在、
    detail 论文都在 lite 中、向量与论文逐条一一对应、全局 ID 无重复；可选与 OpenAlex API 抽样比对
  已通过的分片记在 union/_checked.json，下次只查新分片；有时间预算（--budget），超时自动保存进度

输入
  union/detail、union/lite、union/specter2

输出
  标准输出（PROBLEM: … / RESULT: …）；union/_checked.json（进度记录）

用法
  python 14_check_union_progress.py --no-api            增量检查
  python 14_check_union_progress.py --full --budget 10800   全部重查

运行记录
  2026-10-02  运行中每 20 分钟自动检查；扫描后完整检查 RESULT: OK：detail 26,329,455 行、lite 262,630,159 行，
              全局重复 ID 均为 0；277 个向量分片逐条一致；API 抽样标题 27/30、年份 29/30、子领域 29/30 一致
  原文件名：check_progress.py
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

import duckdb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

D, L, E = (os.path.join(C.UNION_DIR, x) for x in ("detail", "lite", "specter2"))
STATE = os.path.join(C.UNION_DIR, "_checked.json")
problems = []


def bad(msg):
    problems.append(msg)
    print("PROBLEM:", msg, flush=True)


def stable(dirname):
    if not os.path.isdir(dirname):
        return []
    now = time.time()
    return sorted(f for f in os.listdir(dirname)
                  if f.endswith(".parquet") and now - os.path.getmtime(os.path.join(dirname, f)) > 60)


def lst(files, d):
    return "[" + ",".join(f"'{os.path.join(d, f)}'" for f in files) + "]"


def check_detail(c, files):
    r = c.sql(f"""
      select count(*) n, sum((id is null)::int) null_id,
             sum((publication_year not between {C.DETAIL_YEAR_MIN} and {C.DETAIL_YEAR_MAX})::int) bad_year,
             sum((pt_field not in ({",".join(map(str, sorted(C.DETAIL_FIELDS)))}))::int) bad_field,
             sum((len(cby_year) != len(cby_count))::int) cby_misaligned,
             sum((list_min(cby_year) < 2012 or list_max(cby_year) > 2025)::int) cby_out_of_range,
             sum((list_min(cby_count) < 0)::int) cby_negative,
             sum((len(topic_ids) > 3 or len(topic_ids) != len(topic_scores))::int) bad_topics,
             sum((list_min(topic_scores) < 0 or list_max(topic_scores) > 1)::int) bad_topic_score,
             avg((len(refs) = n_refs)::int) refs_complete, sum((len(refs) > n_refs)::int) refs_too_long,
             avg((title is not null)::int) has_title, avg((abstract is not null)::int) has_abstract,
             sum((len(institution_ids) != len(institution_types))::int) inst_misaligned,
             sum((len(concept_ids) != len(concept_scores))::int) concept_misaligned
      from read_parquet({lst(files, D)})""").df().iloc[0]
    print(f"detail 新分片 {len(files)}: n={int(r.n):,} refs_complete={r.refs_complete:.3f} "
          f"has_title={r.has_title:.3f} has_abstract={r.has_abstract:.3f}", flush=True)
    for k in ("null_id", "bad_year", "bad_field", "cby_misaligned", "cby_out_of_range", "cby_negative",
              "bad_topics", "bad_topic_score", "refs_too_long", "inst_misaligned", "concept_misaligned"):
        if r[k] > 0:
            bad(f"detail.{k} = {int(r[k])}")
    if r.refs_complete < 0.95:
        bad(f"detail 参考文献列表完整率只有 {r.refs_complete:.3f}")
    if r.has_title < 0.98:
        bad(f"detail 有标题比例只有 {r.has_title:.3f}")


def check_lite(c, files):
    r = c.sql(f"""
      select count(*) n, sum((id is null)::int) null_id, avg((year is null)::int) no_year,
             sum((year >= {C.REFS_YEAR_MIN} and refs is null)::int) missing_refs_list,
             sum((year < {C.REFS_YEAR_MIN} and refs is not null)::int) unexpected_refs_list,
             sum((len(cby_year) != len(cby_count))::int) cby_misaligned
      from read_parquet({lst(files, L)})""").df().iloc[0]
    print(f"lite 新分片 {len(files)}: n={int(r.n):,} no_year={r.no_year:.4f}", flush=True)
    for k in ("null_id", "missing_refs_list", "unexpected_refs_list", "cby_misaligned"):
        if r[k] > 0:
            bad(f"lite.{k} = {int(r[k])}")


def check_embed(files, scan_done):
    """返回本次确认通过的分片. 向量数值 (NaN/全零/长度) 由 03 步当场检查并写入 _stats/;
    这里只读 id 列核对向量与论文一一对应. 早期没有 _stats 的分片, 等扫描结束、磁盘空闲后再做一次全量数值检查."""
    ok, n, deferred = [], 0, 0
    for f in files:
        sp = os.path.join(E, "_stats", f + ".json")
        if not os.path.exists(sp) and not scan_done:
            deferred += 1
            continue
        got = pq.read_table(os.path.join(E, f), columns=["id"]).column("id").to_pylist()
        dt = pq.read_table(os.path.join(D, f), columns=["id", "title", "abstract"]).to_pydict()
        want = {i for i, t, a in zip(dt["id"], dt["title"], dt["abstract"]) if t or a}   # 与 03 步规则一致
        if set(got) != want or len(got) != len(want):
            bad(f"specter2/{f}: 向量 ID 与 detail 不一致 (向量 {len(got)}, 应有 {len(want)})")
            continue
        if os.path.exists(sp):
            s = json.load(open(sp))
            if s["n_nonfinite"] or s["n_zero"]:
                bad(f"specter2/{f}: NaN/Inf {s['n_nonfinite']} 个, 全零 {s['n_zero']} 个")
                continue
        else:                                     # 早期分片: 全量读向量检查一次
            emb = np.asarray(pq.read_table(os.path.join(E, f), columns=["emb"]).column("emb").combine_chunks()
                             .flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 768)
            norms = np.linalg.norm(emb, axis=1)
            if not np.isfinite(emb).all() or (norms < 1e-3).any():
                bad(f"specter2/{f}: 含 NaN/Inf 或全零向量")
                continue
        ok.append(f)
        n += len(got)
    print(f"specter2 新分片 {len(files)}: 通过 {len(ok)} 片 ({n:,} 个向量); "
          f"等扫描结束后再查 {deferred} 片 (早期分片, 需读全量向量)", flush=True)
    return ok


def col_stats(path):
    """读 parquet 文件尾部的列统计 (不读数据): {列路径: (null_count, min, max)}, 以及行数."""
    md = pq.ParquetFile(path).metadata
    out = {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for ci in range(g.num_columns):
            col = g.column(ci)
            st = col.statistics
            if st is None:
                continue
            p = col.path_in_schema
            nul, mn, mx = out.get(p, (0, None, None))
            nul += st.null_count or 0
            if st.has_min_max:
                mn = st.min if mn is None else min(mn, st.min)
                mx = st.max if mx is None else max(mx, st.max)
            out[p] = (nul, mn, mx)
    return md.num_rows, out


def light_check(new):
    """扫描进行中使用: 只读元数据, 不和扫描抢磁盘. 返回通过检查的分片."""
    ok = {"detail": [], "lite": [], "specter2": []}
    rng = lambda s, p: (s[p][1], s[p][2]) if p in s else (None, None)
    for f in new["detail"]:
        n0 = len(problems)
        n, s = col_stats(os.path.join(D, f))
        if s.get("id", (1,))[0]:
            bad(f"detail/{f}: id 有空值")
        y = rng(s, "publication_year")
        if y[0] is not None and (y[0] < C.DETAIL_YEAR_MIN or y[1] > C.DETAIL_YEAR_MAX):
            bad(f"detail/{f}: 年份越界 {y}")
        fl = rng(s, "pt_field")
        if fl[0] is not None and (fl[0] < min(C.DETAIL_FIELDS) or fl[1] > max(C.DETAIL_FIELDS)):
            bad(f"detail/{f}: 领域越界 {fl}")
        cy = rng(s, "cby_year.list.element")
        if cy[0] is not None and (cy[0] < 2012 or cy[1] > 2025):
            bad(f"detail/{f}: 逐年被引年份越界 {cy}")
        cc = rng(s, "cby_count.list.element")
        if cc[0] is not None and cc[0] < 0:
            bad(f"detail/{f}: 逐年被引为负 {cc}")
        ts = rng(s, "topic_scores.list.element")
        if ts[0] is not None and (ts[0] < 0 or ts[1] > 1):
            bad(f"detail/{f}: 方向分数越界 {ts}")
        if len(problems) == n0:
            ok["detail"].append(f)
    for f in new["lite"]:
        n0 = len(problems)
        n, s = col_stats(os.path.join(L, f))
        if s.get("id", (1,))[0]:
            bad(f"lite/{f}: id 有空值")
        cc = rng(s, "cby_count.list.element")
        if cc[0] is not None and cc[0] < 0:
            bad(f"lite/{f}: 逐年被引为负 {cc}")
        if len(problems) == n0:
            ok["lite"].append(f)
    for f in new["specter2"]:
        n0 = len(problems)
        n_emb = pq.ParquetFile(os.path.join(E, f)).metadata.num_rows
        n_det, s = col_stats(os.path.join(D, f))
        # 规则: 标题或摘要非空即有向量. 元数据只知道 null 数, 分不出空字符串, 只能给范围:
        #   下限 = 有标题的论文数 - 5 (空字符串标题), 上限 = min(总行数, 有标题数 + 有摘要数)
        n_title = n_det - s.get("title", (0,))[0]
        n_abs = n_det - s.get("abstract", (0,))[0]
        lo, hi = n_title - 5, min(n_det, n_title + n_abs)
        if not lo <= n_emb <= hi:
            bad(f"specter2/{f}: 向量 {n_emb} 条, 不在合理范围 [{lo}, {hi}] 内")
        sp = os.path.join(E, "_stats", f + ".json")
        if os.path.exists(sp):
            st = json.load(open(sp))
            if st["n_nonfinite"] or st["n_zero"]:
                bad(f"specter2/{f}: NaN/Inf {st['n_nonfinite']} 个, 全零 {st['n_zero']} 个")
        if len(problems) == n0:
            ok["specter2"].append(f)
    print(f"轻量检查 (只读元数据): detail {len(ok['detail'])}/{len(new['detail'])}, "
          f"lite {len(ok['lite'])}/{len(new['lite'])}, specter2 {len(ok['specter2'])}/{len(new['specter2'])} 通过", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--api-n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=780, help="时间预算 (秒), 用完后保存进度并结束")
    args = ap.parse_args()
    t0 = time.time()
    c = duckdb.connect()
    c.execute(f"SET temp_directory='{C.TMP_DIR}'")
    state = {"detail": [], "lite": [], "specter2": []}
    if os.path.exists(STATE) and not args.full:
        state = json.load(open(STATE))

    df, lf, ef = stable(D), stable(L), stable(E)
    scan_done = os.path.exists(os.path.join(C.UNION_DIR, "_summary.json"))

    # 扫描进行中: 只做元数据轻量检查 (不和扫描抢磁盘); 扫描结束后才做读内容的完整检查
    if not scan_done and not args.full:
        for k in ("light_detail", "light_lite", "light_specter2"):
            state.setdefault(k, [])
        newl = {"detail": [f for f in df if f not in set(state["light_detail"])],
                "lite": [f for f in lf if f not in set(state["light_lite"])],
                "specter2": [f for f in ef if f not in set(state["light_specter2"])]}
        ok = light_check(newl)
        for k in ok:
            state["light_" + k] = sorted(set(state["light_" + k]) | set(ok[k]))
        json.dump(state, open(STATE, "w"))
        print(f"RESULT: {'OK' if not problems else f'{len(problems)} PROBLEMS'}  ({time.time() - t0:.0f}s)  "
              f"[扫描进行中, 轻量检查; 累计 detail {len(state['light_detail'])} / lite {len(state['light_lite'])} / "
              f"specter2 {len(state['light_specter2'])} 片; 完整检查待扫描结束]", flush=True)
        sys.exit(1 if problems else 0)

    new = {"detail": [f for f in df if f not in set(state["detail"])],
           "lite": [f for f in lf if f not in set(state["lite"])],
           "specter2": [f for f in ef if f not in set(state["specter2"])]}
    print(f"分片总数 detail {len(df)} / lite {len(lf)} / specter2 {len(ef)}; "
          f"本次新查 {len(new['detail'])} / {len(new['lite'])} / {len(new['specter2'])}", flush=True)

    # 1. 新分片可读
    for d, key in ((D, "detail"), (L, "lite"), (E, "specter2")):
        for f in new[key]:
            try:
                pq.ParquetFile(os.path.join(d, f)).metadata
            except Exception as e:
                bad(f"无法读取 {d}/{f}: {e}")
    n_bad_before = len(problems)

    # 2-4. 新分片的字段与一致性: 分批检查, 每批通过即保存进度; 超出时间预算就停下, 剩余的下次再查
    over = lambda: time.time() - t0 > args.budget
    passed = {"detail": [], "lite": [], "specter2": []}

    def save(key, files):
        passed[key] += files
        state[key] = sorted(set(state[key]) | set(files))
        json.dump(state, open(STATE, "w"))

    for key, step, fn in (("detail", 10, lambda fs: check_detail(c, fs)),
                          ("lite", 100, lambda fs: check_lite(c, fs)),
                          ("specter2", 20, lambda fs: check_embed(fs, scan_done))):
        for s in range(0, len(new[key]), step):
            if over():
                break
            batch = new[key][s:s + step]
            before = len(problems)
            res = fn(batch)
            ok = res if key == "specter2" else batch      # check_embed 返回确认通过的分片 (早期分片可能延后)
            if len(problems) == before:
                save(key, ok)
    remaining = {k: len(new[k]) - len(passed[k]) for k in new}

    # 5. 全局 ID 重复 (只读 id 列); 时间不够就留到下次
    for name, d, files in (("detail", D, df), ("lite", L, lf)):
        if not files:
            continue
        if over() or any(remaining.values()):
            print(f"{name} 全局 ID 重复检查: 分片还没查完或时间预算用完, 下次再查", flush=True)
            continue
        n, nd = c.sql(f"select count(*), count(distinct id) from read_parquet({lst(files, d)}, union_by_name=true)").fetchone()
        print(f"{name} 全局: {n:,} 行, 重复 ID {n - nd:,}", flush=True)
        if n != nd:
            bad(f"{name} 有重复 ID: {n - nd} 行 (并行切分边界可能出错)")

    # 6. 已结束的 worker: 其 detail 论文都应在 lite 中
    for w in sorted({f.split("_")[0] for f in new["detail"]}):
        if os.path.exists(os.path.join(C.UNION_DIR, f"_worker_{w[1:]}.json")):
            miss = c.sql(f"""select count(*) from read_parquet('{D}/{w}_*.parquet') d
                             anti join read_parquet('{L}/{w}_*.parquet') l using (id)""").fetchone()[0]
            if miss:
                bad(f"{w}: {miss} 篇 detail 论文不在 lite 中")

    # 7. 与 OpenAlex API 抽样比对 (只抽新分片)
    if not args.no_api and new["detail"]:
        rows = c.sql(f"""select id, title, publication_year, pt_subfield from read_parquet({lst(new['detail'], D)})
                         using sample {args.api_n}""").fetchall()
        n = mt = my = ms = 0
        for i, t, y, s in rows:
            try:
                url = f"https://api.openalex.org/works/W{i}?select=id,title,publication_year,primary_topic"
                o = json.load(urllib.request.urlopen(url, timeout=20))
            except Exception:
                continue
            sub = ((o.get("primary_topic") or {}).get("subfield") or {}).get("id")
            sub = int(str(sub).rsplit("/", 1)[-1]) if sub else None
            n += 1
            mt += (t or "") == (o.get("title") or "")
            my += y == o.get("publication_year")
            ms += s == sub
            time.sleep(0.15)
        if n:
            print(f"API 抽样 {n} 篇 (API 为现行数据, 少量差异正常): 标题 {mt}/{n}, 年份 {my}/{n}, 子领域 {ms}/{n}", flush=True)
            if mt < 0.8 * n or my < 0.8 * n:
                bad("与 API 比对一致率过低, 解析可能有误")

    left = "; ".join(f"{k} 剩 {v} 片" for k, v in remaining.items() if v)
    print(f"RESULT: {'OK' if not problems else f'{len(problems)} PROBLEMS'}  ({time.time() - t0:.0f}s)"
          + (f"  [未查完, 下次继续: {left}]" if left else ""), flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
