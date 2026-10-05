"""RivalBench evaluation: the exact metrics and significance test used in the paper.

Usage
-----
    import pandas as pd
    from rivalbench_evaluate import evaluate, compare

    papers = pd.read_parquet("data/ai/papers.parquet")
    test = papers[papers.split_main == "test"]
    pred = pd.Series(..., index=test.paper_id)          # predicted log(1 + citations in years Y+1..Y+3)
    print(evaluate(test, pred))                           # MALE, RMSLE, rho_sub, NDCG@10, rho_all
    print(compare(test, pred_a, pred_b))                  # differences a - b with 95% intervals (subtopic bootstrap)

Definitions (identical to the paper)
-----------------------------------
- Target: y = cites_3y (citations received in calendar years Y+1..Y+3); predictions are on the log(1 + y) scale.
- MALE / RMSLE: mean absolute / root mean squared error of the prediction against log(1 + y).
- rho_sub: Spearman correlation within each evaluation subtopic (column `eval_subtopic`) with at least 10 papers
  (0 when predictions or targets are constant), averaged with weights equal to subtopic size.
- NDCG@10: within each evaluation subtopic with at least 20 papers and nonzero citations, gain = y, averaged
  over subtopics; ties broken by a fixed random jitter (seed 0).
- rho_all: Spearman correlation over all papers.
- Significance: 1,000 resamples of whole evaluation subtopics (multinomial weights, seed 2026).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEY = "eval_subtopic"


def _frame(papers, pred):
    d = papers[["paper_id", KEY, "cites_3y"]].copy()
    d["pred"] = pd.Series(pred).reindex(d.paper_id).to_numpy()
    if d.pred.isna().any():
        raise ValueError(f"{int(d.pred.isna().sum())} papers have no prediction")
    d["y"] = d.cites_3y.astype(float)
    d["ly"] = np.log1p(d.y)
    return d


def group_rho(d, key=KEY, min_n=10):
    out = []
    for g, x in d.groupby(key, sort=False):
        if len(x) < min_n:
            continue
        if x.pred.nunique() <= 1 or x.y.nunique() <= 1:
            r = 0.0
        else:
            r = spearmanr(x.pred, x.y).statistic
        out.append((g, len(x), 0.0 if np.isnan(r) else r))
    return pd.DataFrame(out, columns=["g", "n", "rho"])


def ndcg10(d, key=KEY, min_n=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _, x in d.groupby(key, sort=False):
        if len(x) < min_n or x.y.sum() == 0:
            continue
        tie = rng.random(len(x)) * 1e-9
        top = np.argsort(-(x.pred.to_numpy() + tie))[:10]
        disc = 1 / np.log2(np.arange(2, 12))
        dcg = (x.y.to_numpy()[top] * disc[:len(top)]).sum()
        idcg = (np.sort(x.y.to_numpy())[::-1][:10] * disc[:len(top)]).sum()
        vals.append(dcg / idcg)
    return float(np.mean(vals))


def evaluate(papers, pred):
    """papers: rows of papers.parquet to evaluate on; pred: Series indexed by paper_id (log scale)."""
    d = _frame(papers, pred)
    e = d.pred - d.ly
    rs = group_rho(d)
    return {"MALE": float(e.abs().mean()), "RMSLE": float(np.sqrt((e ** 2).mean())),
            "rho_sub": float(np.average(rs.rho, weights=rs.n)), "NDCG@10": ndcg10(d),
            "rho_all": float(spearmanr(d.pred, d.y).statistic)}


def compare(papers, pred_a, pred_b, n_boot=1000, seed=2026):
    """Differences a - b in MALE and rho_sub, with 95% intervals from resampling whole subtopics."""
    subs = np.sort(papers[KEY].unique())
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(subs), np.full(len(subs), 1 / len(subs)), size=n_boot).astype(float)

    def agg(pred):
        d = _frame(papers, pred)
        g = d.assign(ae=(d.pred - d.ly).abs()).groupby(KEY).agg(se=("ae", "sum"), n=("ae", "count")).reindex(subs).fillna(0)
        rs = group_rho(d).set_index("g")
        rn = (rs.rho * rs.n).reindex(subs).fillna(0).to_numpy()
        nr = rs.n.reindex(subs).fillna(0).to_numpy()
        return (W @ g.se.to_numpy()) / (W @ g.n.to_numpy()), (W @ rn) / (W @ nr)

    ma, ra = agg(pred_a)
    mb, rb = agg(pred_b)
    dm, dr = ma - mb, ra - rb
    return {"dMALE": float(dm.mean()), "dMALE_95ci": (float(np.quantile(dm, .025)), float(np.quantile(dm, .975))),
            "drho_sub": float(dr.mean()), "drho_sub_95ci": (float(np.quantile(dr, .025)), float(np.quantile(dr, .975)))}
