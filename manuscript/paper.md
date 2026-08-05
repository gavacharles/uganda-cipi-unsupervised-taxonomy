# Beyond Trade Classifications: An Unsupervised Machine Learning Taxonomy of Construction Input Price Dynamics in Uganda

## Abstract
Construction price systems in emerging economies exhibit non-linear co-movement that administrative categories do not necessarily capture. Using Uganda’s disaggregated Construction Input Price Index (CIPI) block from the CIO panel, this paper estimates a data-driven taxonomy of material-price dynamics with Dynamic Time Warping (DTW), DBA-based time-series k-means, and manifold visualization. The analysis compares normalization choices (z-scored log returns, normalized log levels, detrended normalized log levels), evaluates clustering robustness with k-medoids and hierarchical Ward-on-MDS(DTW), and quantifies divergence from administrative classifications using the adjusted Rand index (ARI). The joint silhouette-gap sweep selects a compact partition (`k=4`) under normalized levels with DTW radius 12 months. Divergence from administrative groupings is non-trivial: ARI against trade groups is `0.210` in the selected setup, and as low as `0.088` under return normalization. Temporal stability is partial (half-sample ARI `0.351`, migration rate `27.8%`). Findings support procurement designs based on volatility clusters rather than inherited trade sections.

---

## 1. Introduction
Conventional construction-input grouping frameworks (trade sections, ISIC-like categories, and bill-of-quantities headings) are administrative conveniences. They are not validated as representations of how prices co-move through time. If escalation clauses, framework contracts, and hedging attention are anchored to misaligned groupings, procurement risk management inherits avoidable basis risk.

This paper asks whether Ugandan CIPI sub-indices form coherent dynamic clusters that cut across administrative classes. The contribution is applied: a practical clustering blueprint for procurement risk segmentation.

---

## 2. Research Questions and Hypotheses
### Research Questions
1. Do Ugandan construction input prices form statistically coherent clusters based on dynamic behavior rather than administrative class?
2. How strongly does the empirical taxonomy diverge from administrative groupings?
3. Is cluster membership stable through time, or do materials migrate across regimes?

### Hypotheses
- **H1**: The optimal cluster count is small (`k=3..5`), and agreement with administrative classes is weak (ARI near zero).
- **H2**: Fuel-linked and import-linked inputs form distinct high-shock regimes, while domestic aggregates sit in a lower-volatility regime.

---

## 3. Data
- Source panel: `panel_v1.0.parquet` from the CIO pipeline.
- Inputs used: 18 disaggregated CIPI sub-indices only.
- Frequency: monthly.
- Realized sample available in panel: 2017-07 to 2026-04 (106 months).

Although the study concept targets 2015–2025, empirical execution uses the available panel window.

---

## 4. Methodology
### 4.1 Feature constructions
Three alternatives are estimated:
1. `returns_z`: z-scored log returns.
2. `levels_z`: z-scored log levels.
3. `detrended_levels_z`: z-scored detrended log levels.

### 4.2 Distance and clustering
- Distance metric: DTW with Sakoe-Chiba radius sweep (`3, 6, 12`).
- Primary estimator: `TimeSeriesKMeans` with DTW and DBA barycenters.
- Robustness estimators:
  - PAM k-medoids on DTW matrix,
  - hierarchical Ward on MDS embedding of DTW distances.

### 4.3 Model selection and validation
- `k` sweep: `k in {3,4,5}`.
- Joint decision rule: silhouette + gap statistic.
- Divergence metric: adjusted Rand index against two administrative benchmarks (`trade_group`, `isic_proxy`).
- Stability: first-half vs second-half clustering, Hungarian label remapping, migration table.

---

## 5. Results
### 5.1 Optimal partition and divergence
The joint sweep selects:
- normalization: `levels_z`
- DTW radius: `12`
- clusters: `k=4`

Headline divergence (selected specification):
- ARI vs trade grouping: `0.210`
- ARI vs ISIC proxy grouping: `0.160`

Normalization sensitivity matters materially:
- Under `returns_z`, ARI vs trade falls to `0.088` (near-zero alignment).

Interpretation: the taxonomy is not a relabeling of trade classes; alignment is weak-to-moderate and specification-dependent.

### 5.2 Regime structure and cross-trade clustering
Empirical clusters mix categories that administrative systems keep separate. Example patterns include:
- mixed clusters containing masonry, labour, plumbing and carpentry,
- metals splitting into distinct dynamic groups rather than one monolithic “metal trade” block,
- an earthworks-heavy cluster that still absorbs fuel- and metal-linked series in shock episodes.

This supports the “cross-trade co-movement” claim.

### 5.3 Estimator triangulation
Pairwise ARI across estimators:
- DBA k-means vs k-medoids: `0.644`
- DBA k-means vs Ward-on-MDS(DTW): `0.809`
- k-medoids vs Ward-on-MDS(DTW): `0.525`

Conclusion: broad, not perfect, agreement.

### 5.4 Temporal stability
Half-sample stability is partial:
- Stability ARI: `0.351`
- Migration rate: `27.8%` (5 of 18 materials)

So regimes are persistent for a majority, but not fixed.

### 5.5 DTW value-add
Euclidean baseline silhouette scores are lower (`0.305–0.361`) than the selected DTW solution (`0.474`), supporting DTW’s incremental value under lagged co-movement.

---

## 6. Figures
Main figures generated:
1. `outputs/figures/cluster_heatmap.png` (co-movement intensity heatmap)
2. `outputs/figures/tsne_manifold.png` (2D manifold of DTW geometry)
3. `outputs/figures/centroid_profiles.png` (cluster centroid profiles over time)
4. `outputs/figures/optimal_partition_and_divergence.png` (joint `k` selection profile + ARI divergence by normalization)
5. `outputs/figures/regime_structure_cross_trade.png` (cluster composition by trade + cluster×trade matrix)
6. `outputs/figures/estimator_triangulation.png` (pairwise estimator ARI + average co-assignment map)
7. `outputs/figures/temporal_stability.png` (half-sample migration matrix + migration rates)

Appendix robustness figure:
- `outputs/figures/tsne_umap_appendix.png`

---

## 7. Discussion and Procurement Implications
The results indicate that administrative categories are an imperfect hedge map for price risk. Inputs that sit in different trades can share common dynamic shocks, while nominal trade siblings can decouple.

**Actionable implication:** procurement officers should complement or partially replace trade-based escalation buckets with volatility-cluster buckets when:
- drafting escalation clauses,
- structuring framework contract lots,
- targeting price-risk monitoring/hedging attention.

A practical path is to maintain trade sections for operational contracting, but overlay cluster IDs as risk tags in contract management systems.

---

## 8. Limitations
1. Sample starts in 2017 in the current panel build.
2. Administrative benchmark is implemented through a transparent proxy mapping file.
3. DTW-Ward is operationalized via MDS embedding before Ward linkage.

---

## 9. Reproducibility
- Script: `src/run_taxonomy.py`
- Synthetic DBA unit test: `tests/test_dba_convergence.py`
- Run metadata: `outputs/manifest.json`

