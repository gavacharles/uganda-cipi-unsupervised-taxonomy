# Paper 3 — Unsupervised Taxonomy of CIPI Dynamics (Independent Build)

This folder is a standalone paper workspace built from the CIO pipeline outputs, without modifying the main pipeline code.

## Scope
- Data source: `cio_pipeline-2/data/processed/panel_v1.0.parquet`
- Frequency: monthly
- Realized sample in panel: 2017-07 to 2026-04
- Materials used: disaggregated CIPI block only (18 sub-indices)

## Reproducible workflow
1. Run analysis:
   - `"/Users/charlesgava/Documents/Projects /Paper 3- Unsupervised Taxonomy/cio_pipeline-2/.venv/bin/python" src/run_taxonomy.py`
2. Run synthetic DBA test:
   - `"/Users/charlesgava/Documents/Projects /Paper 3- Unsupervised Taxonomy/cio_pipeline-2/.venv/bin/python" -m pytest -q`

## What is generated
- `outputs/manifest.json` — main run metadata + headline metrics
- `outputs/tables/k_selection_grid.csv` — silhouette + gap for k/radius/normalization sweep
- `outputs/tables/taxonomy_divergence_by_normalization.csv` — ARI vs trade and ISIC proxies
- `outputs/tables/clusters_with_trade_mapping.csv` — empirical taxonomy table
- `outputs/tables/cluster_migration.csv` — temporal stability migration output
- `outputs/tables/estimator_agreement.csv` — triangulation agreement (k-means/k-medoids/Ward)
- `outputs/figures/cluster_heatmap.png` — co-movement heatmap
- `outputs/figures/tsne_manifold.png` — main manifold figure
- `outputs/figures/centroid_profiles.png` — cluster centroid profiles
- `outputs/figures/tsne_umap_appendix.png` — t-SNE/UMAP robustness figure

## Methods implemented
- Distance: DTW with Sakoe-Chiba radius sweep (`3, 6, 12`)
- Clustering: DTW-DBA `TimeSeriesKMeans`; robustness via PAM k-medoids and Ward-on-MDS(DTW)
- Selection: joint silhouette and gap statistic sweep
- Normalization sensitivity: returns z-score, levels z-score, detrended levels z-score
- Stability: first-half vs second-half clustering with label remapping and migration table

## Note on classification benchmark
`src/trade_mapping.csv` provides the administrative benchmark used for ARI checks (`trade_group` and `isic_proxy`) so the taxonomy divergence can be quantified consistently.
