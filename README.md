# RivalNet and RivalBench

Code and benchmark for the paper **Scooped or Spotlighted? Rival-Aware Citation Prediction for New Papers on the Scholarly Web**.

New papers compete for attention: a paper's citations depend on the contemporaries it competes with. This repository contains

- **RivalNet**, a rival-aware citation predictor: each new paper attends to its own rivals (the most similar papers visible at
  prediction time) with learned, signed effects and a null rival that acts as an outside option; it conditions on the demand of its
  subtopic and is trained with a within-subtopic Dirichlet-multinomial share likelihood;
- **RivalBench**, the data to study and evaluate rival-aware prediction: competition sets, near-identical twin pairs, publication-time
  features, and temporal splits for 541K papers in three disciplines (AI, oncology, applied mathematics), together with the test
  predictions of all methods in the paper;
- the full pipeline that builds the data from an OpenAlex snapshot, the pre-registered evidence tests, all baselines, and all analyses.

## Data

The data are hosted separately as two archives: `rivalbench_data.zip` (240 MB) and the optional `rivalbench_embeddings.zip` (1.6 GB, SPECTER2 embeddings); see [`data/README.md`](data/README.md).
They can be browsed and downloaded at https://osf.io/nmd5x/overview?view_only=040742fe90034c9097abe380d4ebeedb.

```bash
python scripts/download_data.py            # downloads both archives, unpacks them into data/, verifies checksums
python scripts/download_data.py --no-emb   # without the embeddings
```

The task, splits, columns, and caveats are documented in [`data/README.md`](data/README.md) and [`DATASHEET.md`](DATASHEET.md).

## Evaluate a method

[`rivalbench_evaluate.py`](rivalbench_evaluate.py) implements exactly the metrics and the significance test of the paper.

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

Running this on the released files reproduces Table 3 of the paper.

## Repository layout

| Path | Contents |
|---|---|
| `code/code/` | Data pipeline (scripts `10`–`35`): scanning an OpenAlex snapshot, SPECTER2 embeddings, de-duplication and cleaning, subtopic clustering, similarity counts, prediction samples, competition sets, reference and institution features. `00_脚本清单.md` lists every script. |
| `code/exp0_analysis/` | Pre-registered evidence tests (shared demand, twins, visibility, cross-domain crowding) and Figure 1. |
| `code/exp1/` | Feature baselines (LightGBM, MLP, NAIP), the evaluation functions, baseline tuning, and the second test year. |
| `code/exp2/` | **RivalNet** (`02_mpcnet.py`; named MPC-Net in the code, final configuration `autosearch/configs/AS_r06_1.json`), ablations, hyperparameter search, final evaluation, comparison with all baselines, kNN baselines, twin counterfactuals, and the level-invariance check. |
| `code/exp4`–`code/exp9/` | Mechanism analyses, placebo rivals and event study, cross-field replication, use cases, robustness, efficiency. |
| `code/baselines/`, `code/third_party/` | Scripts that run HINTS, H2CGL, DPPDCC, and PLM-FT on our data. Only our data preparation and compatibility scripts are included; the official implementations must be obtained from their authors. |
| `scripts/download_data.py` | Data download with checksum verification. |
| `rivalbench_evaluate.py` | Metrics and subtopic-bootstrap test of the paper. |

Most scripts begin with a docstring that states their purpose, inputs, outputs, and usage. Comments are partly in Chinese.
Absolute paths have been replaced by `/path/to/mpcc` (data root), `/path/to/conda/envs/mpcc`, and `/path/to/envs`; set them to your
own locations.

## Environment

Python 3.11; see [`requirements.txt`](requirements.txt). RivalNet trains in about seven minutes on one RTX 4090 (5.8M parameters,
peak memory 4.9 GB). The graph baselines need their own environments (DGL 2.4 for H2CGL and DPPDCC; TensorFlow 1.15 for HINTS).

## Reproducing RivalNet

```bash
cd code/exp2
python 01_build_tensors.py --v6                                   # tensors from the prediction samples and competition sets
python 02_mpcnet.py --configs AS_r06_1 --seeds 1,2,3 --gpus 0,1,2 # ensemble A (seeds 4-10 in run_final.sh)
bash run_final.sh                                                  # remaining seeds of A, then ensemble B (retrained with 2020)
python 08_final_eval.py                                            # one-time test evaluation
python 12_baseline_table.py                                        # comparison with all baselines
```

## Licenses

Code: MIT (see [`LICENSE`](LICENSE)). Data: CC BY 4.0, derived from OpenAlex (CC0). SPECTER2: Apache-2.0.
