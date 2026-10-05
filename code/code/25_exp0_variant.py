"""
【25】实验 0 子样本变体：只保留指定发表渠道（如期刊 + 会议）的论文，重建实验 0 所需的全部表          阶段 2：实验 0

用途
  从实验 0 子集中删除不属于指定渠道的论文——它们既不再作为焦点论文，也不再作为竞争对手、不计入方向供给：
    渠道 = 合并后代表记录（正式版优先）的 source_type（OpenAlex：journal / conference / book series /
           ebook platform / repository / 空）
    jc   journal + conference（主）
    jcb  journal + conference + book series（LNCS、CCIS 等会议论文集在 OpenAlex 中归为 book series）
  被引（因变量）仍按全部来源统计：删掉的是"竞争者"，不是"引用者"。
  重建：
    papers.parquet        只保留指定渠道的论文（is_sample / is_focal 标记不变）
    topic_year.parquet    n_ai_dedup 改为该渠道的样本论文数（原值保留为 n_ai_dedup_all）；其余列不变
                          （demand_* 是施引方规模；n_rule 是跨域方向全部论文数，未按渠道重算）
    pairs.parquet         只保留焦点与对手都在指定渠道的论文对
    sim_counts.parquet    调用【24】--keep：焦点与候选都限于指定渠道的样本论文，重新全量比较（GPU）
    paper_year*.parquet、citations.parquet、versions.parquet、_sim_calibration.json   符号链接到原子集
  自查：原 n_ai_dedup 能由 papers 重算复现；sim_counts 行数 = 保留的焦点论文数；变体的候选数 ≤ 原候选数。

输入
  subsets/exp0/（【20】–【24】）

输出
  subsets/exp0_{name}/   上述各表 + _keep_ids.parquet（保留的论文 id）+ _summary.json（各项计数）

用法
  python 25_exp0_variant.py --name jc  --types journal,conference --gpus 0,1,2,3,5,7
  python 25_exp0_variant.py --name jcb --types "journal,conference,book series" --gpus 0,1,2,3,5,7

运行记录
  2026-10-02  建立；jc 与 jcb 两个变体
"""
import argparse
import json
import os
import subprocess
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

SRC = os.path.join(C.SUBSET_DIR, "exp0")
LINKS = ["paper_year.parquet", "paper_year_in_corpus.parquet", "citations.parquet", "versions.parquet",
         "_sim_calibration.json", "emb_meta.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--types", required=True, help="逗号分隔的 source_type")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    t0 = time.time()
    types = [t.strip() for t in args.types.split(",")]
    out = os.path.join(C.SUBSET_DIR, f"exp0_{args.name}")
    os.makedirs(out, exist_ok=True)
    S = lambda n: os.path.join(SRC, n)
    O = lambda n: os.path.join(out, n)
    tl = ", ".join(f"'{t}'" for t in types)
    con = duckdb.connect()
    con.execute(f"SET threads=96; SET memory_limit='200GB'; SET temp_directory='/path/to/mpcc/tmp'")

    # 1. 论文
    con.execute(f"create table p as select * from read_parquet('{S('papers.parquet')}')")
    con.execute(f"create table keep as select paper_id from p where source_type in ({tl})")
    con.execute(f"copy (select * from p where paper_id in (select paper_id from keep)) to '{O('papers.parquet')}' (format parquet)")
    con.execute(f"copy keep to '{O('_keep_ids.parquet')}' (format parquet)")
    cnt = con.execute(f"""select count(*) n_all, count(*) filter (where is_sample) sample_all, count(*) filter (where is_focal) focal_all,
                                count(*) filter (where source_type in ({tl})) n_keep,
                                count(*) filter (where is_sample and source_type in ({tl})) sample_keep,
                                count(*) filter (where is_focal and source_type in ({tl})) focal_keep from p""").df().iloc[0].to_dict()
    print("论文:", cnt, flush=True)

    # 2. 方向 x 年份供给
    chk = con.execute(f"""with c as (select pt_topic as topic_id, first_public_year as yr, count(*) as n from p where is_sample group by 1, 2)
        select count(*) filter (where coalesce(c.n, 0) <> t.n_ai_dedup) from read_parquet('{S('topic_year.parquet')}') as t
        left join c on c.topic_id = t.topic_id and c.yr = t.year""").fetchone()[0]
    assert chk == 0, f"原 n_ai_dedup 无法由 papers 复现（{chk} 格不一致）"
    con.execute(f"""copy (
        with c as (select pt_topic as topic_id, first_public_year as yr, count(*) as n from p
                   where is_sample and source_type in ({tl}) group by 1, 2)
        select t.* exclude (n_ai_dedup), t.n_ai_dedup as n_ai_dedup_all,
               case when t.n_ai_dedup > 0 then coalesce(c.n, 0) else t.n_ai_dedup end as n_ai_dedup
        from read_parquet('{S('topic_year.parquet')}') as t left join c on c.topic_id = t.topic_id and c.yr = t.year
    ) to '{O('topic_year.parquet')}' (format parquet)""")

    # 3. 论文对
    con.execute(f"""copy (select * from read_parquet('{S('pairs.parquet')}')
        where focal_id in (select paper_id from keep) and comp_id in (select paper_id from keep))
        to '{O('pairs.parquet')}' (format parquet)""")
    cnt["pairs_all"] = con.execute(f"select count(*) from read_parquet('{S('pairs.parquet')}')").fetchone()[0]
    cnt["pairs_keep"] = con.execute(f"select count(*) from read_parquet('{O('pairs.parquet')}')").fetchone()[0]
    print(f"论文对: {cnt['pairs_all']:,} -> {cnt['pairs_keep']:,} ({time.time() - t0:.0f}s)", flush=True)

    # 4. 符号链接
    for n in LINKS:
        if not os.path.lexists(O(n)):
            os.symlink(S(n), O(n))

    # 5. 相似计数（GPU，调用【24】）
    r = subprocess.run([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "24_exp0_sim_counts.py"),
                        "--gpus", args.gpus, "--keep", O("_keep_ids.parquet"), "--out", out])
    assert r.returncode == 0, "【24】失败"
    n_sc = con.execute(f"select count(*) from read_parquet('{O('sim_counts.parquet')}')").fetchone()[0]
    assert n_sc == cnt["focal_keep"], f"sim_counts 行数 {n_sc} ≠ 保留的焦点论文 {cnt['focal_keep']}"
    bad = con.execute(f"""select count(*) from read_parquet('{O('sim_counts.parquet')}') as v
        join read_parquet('{S('sim_counts.parquet')}') as o using (focal_id)
        where v.n_cand_before > o.n_cand_before or v."n_ge0.95_before" > o."n_ge0.95_before" """).fetchone()[0]
    assert bad == 0, f"{bad} 篇论文的变体计数大于原计数"
    cnt.update(con.execute(f"""select avg(v."n_ge0.95_before") mean_prior95_keep, avg(o."n_ge0.95_before") mean_prior95_all_same_focal
        from read_parquet('{O('sim_counts.parquet')}') as v join read_parquet('{S('sim_counts.parquet')}') as o using (focal_id)""").df().iloc[0].to_dict())
    cnt.update({"name": args.name, "types": types, "seconds": round(time.time() - t0)})
    json.dump({k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in cnt.items()},
              open(O("_summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(f"VARIANT {args.name} DONE: {json.dumps(cnt, ensure_ascii=False, default=float)}", flush=True)


if __name__ == "__main__":
    main()
