# RivalBench data

RivalBench accompanies the paper *Scooped or Spotlighted? Rival-Aware Citation Prediction for New Papers on the Scholarly Web*.
It provides, for three disciplines, every new paper's **competition set** (its most similar contemporaries visible at prediction time),
publication-time features, citation targets, temporal splits, near-identical **twin pairs**, SPECTER2 embeddings,
the test predictions of all methods in the paper, and the exact evaluation code.

## Where to get the files

The data are hosted separately from this repository at https://osf.io/nmd5x/overview?view_only=040742fe90034c9097abe380d4ebeedb (an anonymized link during review; a Zenodo record with a DOI will follow after publication).
Two archives: `rivalbench_data.zip` (240 MB: papers, rivals, twins, in-corpus targets, predictions) and `rivalbench_embeddings.zip` (1.6 GB, optional). Run `python scripts/download_data.py` from the repository root, or download the archives manually, put them in `data/`, and run the script to unpack and verify them; checksums are in `MD5SUMS`.

## Task

For a paper first made public in year *Y*, predict at the end of *Y* (time *T*) the number of citations it receives in calendar
years *Y*+1 to *Y*+3, using only information available at *T*. Predictions are made on the log(1 + citations) scale.
Methods are compared by absolute error (MALE, RMSLE) and, most importantly, by how well they rank papers **within a subtopic**
(within-subtopic Spearman and NDCG@10), because papers in the same subtopic and year compete for the same citations.

## Contents

After `python scripts/download_data.py`, the files below are in this folder (`data/`). Fields: `ai`, `oncology`, `applied_math`.

| Path | Rows (AI / oncology / applied math) | Description |
|---|---|---|
| `<field>/papers.parquet` | 313,260 / 155,569 / 72,259 | One row per paper: identifiers, splits, subtopic and cell, publication-time features, targets |
| `<field>/rivals.parquet` | 50 per paper | Competition set: the 50 most similar papers from years *Y*−2 to *Y* visible at *T*, with relation features |
| `<field>/twins.parquet` | 7,963 / 11,534 / 2,420 | Near-identical paper pairs by disjoint teams (similarity ≥ 0.97, 1–183 days apart, no mutual citation) |
| `ai/cites_incorpus.parquet` | 313,260 | Citations counted within the corpus (the target of graph-based baselines) |
| `embeddings/<field>_specter2.npy` | 732,846 / 320,260 / 133,978 | SPECTER2 embeddings (float16, 768-d) of all focal papers and rivals; row → paper in `<field>_specter2_ids.parquet` |
| `predictions/ai_test_allsource.parquet` | | Test predictions (seed ensembles) of RivalNet and all baselines in the paper's main table |
| `predictions/ai_test_incorpus.parquet` | | Same for the in-corpus target |
| `MD5SUMS` | | Checksums of all data files |

Titles and abstracts are not redistributed; they can be retrieved from OpenAlex with the work IDs (`paper_id` is the numeric part of the
OpenAlex ID, e.g. `2890749849` → `https://openalex.org/W2890749849`).

## Splits

- `split_main`: train = 2017–2019, val = 2020, test = 2021 (the paper's main protocol). The validation year is used for early stopping and
  model selection; the test year should be evaluated once.
- `split_shift`: everything shifted back by one year (train = 2017–2018, val = 2019, test = 2020), the paper's second test year.
  Note that 2020 is the main validation year, so hyperparameters tuned on `split_main` have seen it.

## Key columns of `papers.parquet`

- **Identifiers**: `paper_id`, `field`, `year` (*Y*), `first_public_date`, `date_precision`, `topic`, `topic2` (OpenAlex topics).
- **Grouping**: `subtopic` (k-means cluster of SPECTER2 embeddings fitted on papers up to 2019, K = 2000; used as model input),
  `cell` = subtopic × year (`subtopic` · 10000 + `year`), `eval_subtopic` (clustering of all papers, K = 2000; **used only for evaluation**).
- **Targets**: `cites_y1`, `cites_y2`, `cites_y3` (citations in years *Y*+1, *Y*+2, *Y*+3), `cites_3y` (their sum; the main target).
  `cites_y0` (citations in year *Y*) is **not** available at *T* and must not be used as a feature for cold-start prediction.
- **Paper features** (available at *T*): team size (`n_authors`, `n_institutions`, `n_countries`), `n_refs`, open access, abstract and funding
  indicators, `pub_month`, `venue_at_T` / `source_at_T` (a paper whose formal version appears after *Y* counts as a preprint),
  `has_preprint_at_T`, topic match scores, author track records (`auth_*`, `first_auth_*`, `share_new_authors`), venue history (`venue_*`),
  and, for AI only, the quality of cited references (`ref_*`).
- **Demand of the topic and subtopic**: supply (`*_supply_Y`, `_Ym1`, `_Ym2`: papers published), citation inflow (`*_inflow_*`), citing demand
  (`topic_dem_*`), references written (`sub_refs_Y`), and citation levels of earlier cohorts (`*_age1_mean`, `*_y3_mean`).
- **Competition counts**: `c_m1_95`, `c_m2_95`, `c_m1_90`, `c_m2_90` (papers from years *Y*−1 / *Y*−2 with SPECTER2 similarity ≥ 0.95 / 0.90),
  `c_pb365_95` (similar papers in the 365 days before publication, exact dates only), `rival_cites_m1(_max)` (their citations by *T*),
  `s3_m1` (bibliographically coupled papers from *Y*−1), `n_preempted` and `preempt_min_days` (earlier near-identical papers, as in the twin
  definition), `n_cand_ym1` (corpus papers from *Y*−1, the denominator of the similarity counts).
  **Caveat**: the corpus starts in 2016, so counts that look back one or two years are truncated for 2017 (and 2018) papers. Models that use
  raw counts and are trained on few years can extrapolate badly; we log-transform heavy-tailed counts for MLP baselines.

## Columns of `rivals.parquet`

`focal_id`, `rank` (1 = most similar), `rival_id`, `sim` (cosine similarity of SPECTER2 embeddings), `year_offset` (rival year − focal year,
in {−2, −1, 0}), `lead_days` (rival date − focal date in days; missing if a date is known only to the year), `rival_cites_at_T`,
`rival_venue`, `shared_author`, `bib_coupled`, `focal_cites_rival`, `rival_cites_focal` (citations by *T*).

## Evaluating a method

```python
import pandas as pd
from rivalbench_evaluate import evaluate, compare

papers = pd.read_parquet("data/ai/papers.parquet")
test = papers[papers.split_main == "test"]
P = pd.read_parquet("data/predictions/ai_test_allsource.parquet")
rivalnet = P[P.method == "RivalNet"].set_index("paper_id").pred
print(evaluate(test, rivalnet))
# {'MALE': 0.6745, 'RMSLE': 0.8580, 'rho_sub': 0.6440, 'NDCG@10': 0.7204, 'rho_all': 0.7162}
mlp_ra = P[P.method == "Retrieval-augmented MLP"].set_index("paper_id").pred
print(compare(test, rivalnet, mlp_ra))
# dMALE -0.0376 [-0.0410, -0.0343], drho_sub +0.0455 [+0.0427, +0.0484]
```

These numbers reproduce Table 3 of the paper from the released files alone.

## Provenance and licenses

- Source: an OpenAlex snapshot (CC0). Embeddings: SPECTER2 (`allenai/specter2_base` with the proximity adapter, Apache-2.0) on title and abstract.
- RivalBench data: CC BY 4.0. Code: MIT. *(To be confirmed by the authors before the public release.)*
- See `../DATASHEET.md` for motivation, composition, collection process, intended uses, and limitations.
