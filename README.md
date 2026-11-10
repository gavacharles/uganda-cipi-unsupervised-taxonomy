# uganda-cipi-unsupervised-taxonomy

Python code, outputs and manuscript for an unsupervised machine-learning taxonomy of Uganda's construction input price dynamics, compared with administrative trade classifications.

## Data

Input is `data/processed/panel_v1.0.parquet` from [cio_pipeline-2](https://github.com/gavacharles/cio_pipeline-2): monthly, 2017-07 to 2026-04, the 18 disaggregated CIPI sub-indices.

## Contents

| Path | What it is |
|---|---|
| `src/run_taxonomy.py` | Full analysis: DTW distances, DTW-DBA k-means, k-medoids and Ward robustness, k selection, normalisation sensitivity, temporal stability |
| `src/trade_mapping.csv` | Administrative benchmark (trade group and ISIC proxy) used for ARI comparisons |
| `tests/test_dba_convergence.py` | Synthetic DBA convergence test |
| `outputs/manifest.json` | Run metadata and summary metrics |
| `outputs/tables/` | k-selection grid, divergence by normalisation, cluster table, migration table, estimator agreement |
| `outputs/figures/` | Heatmap, manifold, centroid profiles, robustness and stability figures |
| `manuscript/paper.md` | Manuscript draft |

## Running

Use an environment with the cio_pipeline-2 dependencies installed and the panel available:

```
python src/run_taxonomy.py
python -m pytest -q
```
